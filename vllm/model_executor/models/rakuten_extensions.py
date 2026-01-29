import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm._aiter_ops import rocm_aiter_ops
from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.deep_gemm import fp8_mqa_logits, fp8_paged_mqa_logits
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

from .interfaces import MixtureOfExperts, SupportsEagle, SupportsLoRA, SupportsPP
from .utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops as ops

try:
  import autort, os, random
except:
  raise Exception('Failed to import autort, please install:\npython3 -m pip install --no-build-isolation https://github.com/microsoft/antares/releases/download/v0.9.6/autort-0.9.6.4.5+cuda.zip')

@torch.compiler.disable(recursive=True)
def get_inflight_index_map(positions, buffer_count=32):
  forward_context = get_forward_context()
  if not hasattr(get_inflight_index_map, 'inflight_table_map'):
    inflight_table_map = torch.full([192000], -1, dtype=torch.int32, device=positions.device)
    inflight_table_map[0] = 0
    get_inflight_index_map.inflight_table_map = inflight_table_map
    get_inflight_index_map.inflight_map_fn = torch.compiler.disable(autort.export(name=f'inflight_map_fn', dev=positions.device.index, source=r'''
@DEF_FUNC: query_start_loc:int32[N], positions:int64[P], block_table:int32[N, L], inflight_table_map:int32[NUMBLOCKS] -> current_map:int32[N]
@DEF_BIND: ~%~:1
@DEF_EXTRA: world_rank:int32, buffer_count:int32

void main() {
  for (int n = 0; n < size_of_N(); ++n) {
    int left = positions(query_start_loc(n)), right = positions(query_start_loc(n + 1) - 1) + 1;
    if ((left == 0 && right == 1) || block_table(n, 0) < 0) { // warmup
      current_map(n) = 0; continue;
    }
    // printf("debug[gpu-%d, sample-%d/%d]: positions=[%d .. %d)\n", world_rank, n, int(size_of_N()), left, right); continue;

#define TABLE_COUNTER()      (inflight_table_map(0))
#define LEADING_BLOCK_ID(n)  inflight_table_map(block_table(n, 0) + 1)

    if (left == 0) {
      // prefill update mapping
      int curr_addr = atomicAdd(&TABLE_COUNTER(), 1) % buffer_count;
      int prev_addr = LEADING_BLOCK_ID(n);
      LEADING_BLOCK_ID(n) = curr_addr;
      if (world_rank == 0)
        printf("[Inflight-Batch-Logging] New Request => [gpu-%d, sample-%d/%d]: forwarding positions=[%d .. %d), sample-%d register to new cache_idx-%d (previous-idx:%d)\n",
          world_rank, n, int(size_of_N()), left, right, n, curr_addr, prev_addr);
    } else {
      // non-leading-prefill or decode query mapping
      if (world_rank == 0) {
        if (left + 1 < right)
          printf("[Inflight-Batch-Logging] Old Request => [gpu-%d, sample-%d/%d]: forwarding positions=[%d .. %d), sample-%d should use cache_idx-%d\n",
            world_rank, n, int(size_of_N()), left, right, n, LEADING_BLOCK_ID(n));
        if (LEADING_BLOCK_ID(n) < 0)
          printf("  [error found] inflight batching has no mapping index.\n");
      }
    }
    current_map(n) = LEADING_BLOCK_ID(n);
  }
}'''))

  else:
    inflight_table_map = get_inflight_index_map.inflight_table_map

  if forward_context.attn_metadata is None:
    return [], None
  else:
    attn_metadata = forward_context.attn_metadata.get('model.layers.0.self_attn.indexer.k_cache', None)
    if attn_metadata is not None:
      query_start_loc = attn_metadata.query_start_loc[:-1]
      world_rank = int(torch.distributed.get_rank())

      map_array_1d = []
      req_offset = 0
      # NOTE: num_decodes IS BEFORE num_decodes in query_start_loc
      if attn_metadata.num_decodes > 0:
        req_offset += attn_metadata.num_decodes
        block_table = attn_metadata.decode.block_table
        assert block_table.size(0) == attn_metadata.num_decodes
        map_array_1d.append(get_inflight_index_map.inflight_map_fn(query_start_loc[:req_offset],
          positions, block_table, inflight_table_map, extra=[world_rank, buffer_count]))

      if attn_metadata.num_prefills > 0:
        for chunk in attn_metadata.prefill.chunks:
          block_table = chunk.block_table
          for i in range(block_table.size(0)):
            req_offset += 1
            map_array_1d.append(get_inflight_index_map.inflight_map_fn(query_start_loc[req_offset - 1:req_offset],
              positions, block_table[i:i + 1], inflight_table_map, extra=[world_rank, buffer_count]))

      assert attn_metadata.num_prefills + attn_metadata.num_decodes == req_offset
      return map_array_1d, attn_metadata
    else:
      return [], None
