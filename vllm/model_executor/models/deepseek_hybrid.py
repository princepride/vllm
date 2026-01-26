# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs
from torch import nn
from typing import Iterable, Tuple

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM, 
    DeepseekV2Model,
)
from vllm.model_executor.models.interfaces import HasInnerState
from vllm.model_executor.models.utils import maybe_prefix
from vllm.compilation.decorators import support_torch_compile # Needed for decorator

logger = init_logger(__name__)

# --- FLA PATCHES START ---

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["K", "H"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'N'])
def chunk_kda_fwd_kernel_intra_token_parallel_patched(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr, 
    IS_VARLEN: tl.constexpr,
):
    i_tg, i_hg = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = 0
        left, right = 0, N

        # Unrolled binary search (max B=2^32)
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left

        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T

    if i_t >= T:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC

    q += bos * H*K
    k += bos * H*K
    g += bos * H*K
    Aqk += bos * H*BT
    Akk += bos * H*BC
    beta += bos * H

    # BK is now passed as a parameter
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, BK)
    m_h = (i_hg * BH + o_h) < H
    m_k = o_k < K

    p_q = tl.make_block_ptr(q + i_t * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_t * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_t * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta + i_t * H, (H,), (1,), (i_hg * BH,), (BH,), (0,))
    
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_k = b_k * tl.load(p_beta, boundary_check=(0,)).to(tl.float32)[:, None]

    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        p_kj = tl.make_block_ptr(k + j * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        p_gj = tl.make_block_ptr(g + j * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        
        b_kj = tl.load(p_kj, boundary_check=(0, 1)).to(tl.float32)
        b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)

        b_kgj = tl.where(m_k[None, :], b_kj * exp2(b_g - b_gj), 0.0)
        
        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
        b_Akk = tl.where(j < i_t, tl.sum(b_k * b_kgj, axis=1), 0.0)

        tl.store(Aqk + i_t * H*BT + (i_hg * BH + o_h) * BT + j % BT, b_Aqk.to(Aqk.dtype.element_ty), mask=m_h)
        tl.store(Akk + i_t * H*BC + (i_hg * BH + o_h) * BC + j % BC, b_Akk.to(Akk.dtype.element_ty), mask=m_h)


def chunk_kda_fwd_intra_token_parallel_patched(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
) -> None:
    """
    Patched version that computes BK outside the kernel.
    """
    B, T, H, K = q.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size
    BK = triton.next_power_of_2(K) 

    def grid(meta): return (B * T, triton.cdiv(H, meta['BH']))
    chunk_kda_fwd_kernel_intra_token_parallel_patched[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        N=N,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        IS_VARLEN=cu_seqlens is not None
    )
    return Aqk, Akk


def apply_fla_patches():
    """Apply fla patches."""
    global _patches_applied
    if '_patches_applied' in globals() and _patches_applied:
        return

    try:
        import fla.ops.kda.chunk_intra_token_parallel as original_module
        import fla.ops.kda.chunk_intra as chunk_intra_module

        original_module.chunk_kda_fwd_kernel_intra_token_parallel = chunk_kda_fwd_kernel_intra_token_parallel_patched
        original_module.chunk_kda_fwd_intra_token_parallel = chunk_kda_fwd_intra_token_parallel_patched

        if hasattr(chunk_intra_module, 'chunk_kda_fwd_intra_token_parallel'):
            chunk_intra_module.chunk_kda_fwd_intra_token_parallel = chunk_kda_fwd_intra_token_parallel_patched
            
        # Patch get_multiprocessor_count to avoid Dynamo errors
        import fla.modules.convolution
        import fla.utils
        
        def patched_get_multiprocessor_count(device_idx):
            import torch
            try:
                if device_idx is None:
                     device_idx = torch.cuda.current_device()
                return torch.cuda.get_device_properties(device_idx).multi_processor_count
            except:
                return 80 # Fallback default
                
        fla.utils.get_multiprocessor_count = patched_get_multiprocessor_count
        fla.modules.convolution.get_multiprocessor_count = patched_get_multiprocessor_count
            
        
        # Patch chunk_kda to remove @torch.compiler.disable decorator
        import fla.layers.kda
        if hasattr(fla.layers.kda.chunk_kda, '__wrapped__'):
             fla.layers.kda.chunk_kda = fla.layers.kda.chunk_kda.__wrapped__
             logger.info("Unwrapped chunk_kda to enable torch.compile")

        # Patch chunk_kda to remove @torch.compiler.disable decorator
        import fla.layers.kda
        if hasattr(fla.layers.kda.chunk_kda, '__wrapped__'):
             fla.layers.kda.chunk_kda = fla.layers.kda.chunk_kda.__wrapped__
             logger.info("Unwrapped chunk_kda to enable torch.compile")

        # Full replacement patch for kda_gate_fwd_kernel to fix NameError: softplus
        try:
            import fla.ops.kda.gate
            import triton
            import triton.language as tl
            from fla.utils import autotune_cache_kwargs, IS_AMD
            
            BT_LIST_AUTOTUNE = [32, 64, 128]
            NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]

            @triton.heuristics({
                "HAS_BIAS": lambda args: args["dt_bias"] is not None,
                "HAS_BETA": lambda args: args["beta"] is not None,
            })
            @triton.autotune(
                configs=[
                    triton.Config({"BT": BT}, num_warps=num_warps, num_stages=num_stages)
                    for BT in BT_LIST_AUTOTUNE
                    for num_warps in NUM_WARPS_AUTOTUNE
                    for num_stages in [2, 3]
                ],
                key=["H", "D"],
                **autotune_cache_kwargs,
            )
            @triton.jit
            def kda_gate_fwd_kernel_patched(
                g,
                A_log,
                dt_bias,
                beta,
                yg,
                yb,
                T,
                H: tl.constexpr,
                D: tl.constexpr,
                BT: tl.constexpr,
                BD: tl.constexpr,
                HAS_BIAS: tl.constexpr,
                HAS_BETA: tl.constexpr,
            ):
                # Explicitly import softplus inside the kernel if possible, or assume it's in scope
                # However, triton.jit doesn't support imports inside kernels.
                # We rely on the fact that this function is defined within a scope where 
                # we can control globals or it will pick up tl.math.softplus if we use it directly.
                # But to be safe, let's use tl.math.log(1 + tl.math.exp(x)) or try to reference tl.math.softplus? 
                # Actually, standard softplus in triton is tl.softplus? No, it's often custom.
                # The original code used `from fla.ops.utils.softplus import softplus`.
                # We will re-implement a simple softplus here to be self-contained.
                
                i_t, i_h = tl.program_id(0), tl.program_id(1)
            
                b_A = tl.load(A_log + i_h).to(tl.float32)
            
                p_g = tl.make_block_ptr(g + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
                p_yg = tl.make_block_ptr(yg + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
                # [BT, BD]
                b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
                if HAS_BIAS:
                    p_b = tl.make_block_ptr(dt_bias, (H * D,), (1,), (i_h * D,), (BD,), (0,))
                    b_g = b_g + tl.load(p_b, boundary_check=(0,)).to(tl.float32)
                
                # Inline softplus implementation: log(1 + exp(x))
                # Or use tl.extra.cuda.libdevice.log1p(tl.exp(x)) for better precision?
                # Simple version:
                b_softplus_g = tl.log(1 + tl.exp(b_g))
                
                b_yg = -tl.exp(b_A) * b_softplus_g
                tl.store(p_yg, b_yg.to(p_yg.dtype.element_ty), boundary_check=(0, 1))
            
                if HAS_BETA:
                    p_b = tl.make_block_ptr(beta + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
                    p_yb = tl.make_block_ptr(yb + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
                    b_yb = tl.sigmoid(tl.load(p_b, boundary_check=(0,)).to(tl.float32))
                    tl.store(p_yb, b_yb.to(p_yb.dtype.element_ty), boundary_check=(0,))

            fla.ops.kda.gate.kda_gate_fwd_kernel = kda_gate_fwd_kernel_patched
            logger.info("Successfully redefined and patched kda_gate_fwd_kernel")
            
        except Exception as e:
            logger.warning(f"Failed to redefine kda_gate_fwd_kernel: {e}")
             
    except ImportError as e:
        logger.warning(f"Failed to apply fla patches: {e}")
    except Exception as e:
        logger.warning(f"Error applying fla patches: {e}")

    globals()['_patches_applied'] = True

# --- FLA PATCHES END ---

apply_fla_patches()
from fla.layers.kda import KimiDeltaAttention


class DeepseekHybridDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer=None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            config=config,
            topk_indices_buffer=topk_indices_buffer,
        )
        
        self.kda_attn = None
        if config is None:
            config = vllm_config.model_config.hf_config
            
        kda_residual_last_n_layers = getattr(config, "kda_residual_last_n_layers", 0)
        
        if kda_residual_last_n_layers > 0:
            total_layers = config.num_hidden_layers
            if self.layer_idx >= total_layers - kda_residual_last_n_layers:
                self.kda_attn = KimiDeltaAttention(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    head_dim=getattr(config, 'kda_head_dim', 256),
                    expand_v=getattr(config, 'kda_expand_v', 2),
                    use_short_conv=getattr(config, 'kda_use_short_conv', True),
                    conv_size=getattr(config, 'kda_conv_size', 4),
                    conv_bias=getattr(config, 'kda_conv_bias', False),
                    norm_eps=config.rms_norm_eps,
                    layer_idx=self.layer_idx,
                )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        # Self Attention
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_kwargs = {
            "positions": positions,
            "hidden_states": hidden_states,
        }
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
            
        attn_output = self.self_attn(**attn_kwargs)

        if self.kda_attn is not None:
             # KDA expects 3D input (batch, seq, dim)
             # vLLM hidden_states is 2D (batch*seq, dim)
             # We unsqueeze to (1, total_tokens, dim)
             hidden_states_3d = hidden_states.unsqueeze(0)
             
             kda_output, _, _ = self.kda_attn(
                 hidden_states_3d,
                 attention_mask=None,
                 past_key_value=None,
                 use_cache=False,
                 output_attentions=False,
             )
             
             # Squeeze back to 2D
             kda_output = kda_output.squeeze(0)
             attn_output = attn_output + kda_output
             
        hidden_states = attn_output

        from vllm.model_executor.models.deepseek_v2 import DeepseekAttention
        
        if (
            not isinstance(self.self_attn, DeepseekAttention)
            and hidden_states.dtype == torch.float16
        ):
             hidden_states *= 1.0 / self.routed_scaling_factor
             if self.layer_idx == 0:
                 residual *= 1.0 / self.routed_scaling_factor

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP

        if isinstance(self.mlp, DeepseekV2MLP) and hidden_states.dtype == torch.float16:
            hidden_states *= 1.0 / self.routed_scaling_factor

        return hidden_states, residual

@support_torch_compile
class DeepseekHybridModel(DeepseekV2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Avoid attribute errors by redundant initialization
        self.compiled = False
        self.compilation_config = vllm_config.compilation_config
        
        # Call parent initializer to setup all attributes and standard layers
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        
        config = vllm_config.model_config.hf_config
        
        # Logic to replace hybrid layers
        kda_residual_last_n_layers = getattr(config, "kda_residual_last_n_layers", 0)
        
        if kda_residual_last_n_layers > 0:
            total_layers = config.num_hidden_layers
            
            # Recreate buffer for new layers if needed (logic matches DeepseekV2Model)
            topk_indices_buffer = None
            if hasattr(self, 'is_v32') and self.is_v32:
                topk_tokens = config.index_topk
                from vllm.platforms import current_platform
                device = current_platform.device_type
                # Note: This creates a new buffer. 
                # If we want to share the exact same buffer object as original layers,
                # we could try to extract it from self.layers[0]. 
                # BUT DeepseekV2DecoderLayer does not store it as a public attribute in 
                # standard implementation (it's closed over or stored in internal backend).
                # However, creating a duplicate buffer is safe logic-wise, just small memory overhead.
                topk_indices_buffer = torch.empty(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    topk_tokens,
                    dtype=torch.int32,
                    device=device,
                )

            for i in range(total_layers):
                if i >= total_layers - kda_residual_last_n_layers:
                    # Replace with Hybrid Layer
                    layer_prefix = f"{prefix}.layers.{i}"
                    
                    # Clean up old layer registration to avoid Duplicate layer name error
                    # Recursively remove any module that registered itself in static_forward_context
                    # This handles Attention, MLA, FusedMoE, and any other components
                    old_layer = self.layers[i]
                    context = vllm_config.compilation_config.static_forward_context
                    for module in old_layer.modules():
                        if hasattr(module, "layer_name"):
                             layer_name = module.layer_name
                             if layer_name in context:
                                 del context[layer_name]

                    new_layer = DeepseekHybridDecoderLayer(
                        vllm_config, 
                        layer_prefix, 
                        config=config, 
                        topk_indices_buffer=topk_indices_buffer
                    )
                    self.layers[i] = new_layer


class DeepseekHybridForCausalLM(DeepseekV2ForCausalLM, HasInnerState):
    model_cls = DeepseekHybridModel
    
    def set_moe_parameters(self):
        from vllm.model_executor.models.utils import PPMissingLayer
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
        
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            if hasattr(layer, "mlp") and isinstance(layer.mlp, DeepseekV2MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Separate KDA weights to handle them manually
        kda_weights = {}

        def filter_weights(weights_iter):
            for name, w in weights_iter:
                if "kda" in name:
                    kda_weights[name] = w
                else:
                    yield name, w

        # Call parent to load standard weights
        super().load_weights(filter_weights(weights))

        # Process and load KDA weights manually
        from collections import defaultdict
        grouped_weights = defaultdict(dict)

        for name, weight in kda_weights.items():
            if name.endswith(".weight_scale_inv"):
                base_name = name[:-17]
                grouped_weights[base_name]["scale"] = weight
            elif name.endswith(".weight"):
                base_name = name[:-7]
                grouped_weights[base_name]["weight"] = weight
            else:
                grouped_weights[name]["weight"] = weight

        for base_name, tensors in grouped_weights.items():
            if "weight" not in tensors:
                continue

            weight = tensors["weight"]
            scale = tensors.get("scale", None)

            # Dequantize if scale is provided
            if scale is not None:
                # Handle block quantization (block_size=128)
                if weight.dim() == scale.dim():
                    for dim in range(weight.dim()):
                         if weight.shape[dim] == scale.shape[dim] * 128:
                             scale = torch.repeat_interleave(scale, 128, dim=dim)

                weight = weight.float() * scale.float()

            # Format: model.layers.X.kda.PARAM
            parts = base_name.split(".")
            try:
                # Check for "kda" existence in parts
                if "kda" not in parts: 
                    continue

                kda_index = parts.index("kda")
                if kda_index < 2:
                    continue
                
                # Assume layer index is before "kda" and "layers" is before that
                # e.g. ...layers.55.kda...
                layer_idx_str = parts[kda_index - 1]
                if not layer_idx_str.isdigit():
                    continue
                layer_idx = int(layer_idx_str)

                if not (0 <= layer_idx < len(self.model.layers)):
                    continue

                layer = self.model.layers[layer_idx]
                if not hasattr(layer, "kda_attn") or layer.kda_attn is None:
                    continue
                
                # vLLM implementation calls it kda_attn, user snippet called it kda
                target_module = layer.kda_attn
                
                # Parameter name is everything after kda
                param_parts = parts[kda_index + 1:]
                param_name = ".".join(param_parts)
                
                target_param = None

                # Mapping logic from user snippet
                if param_name == "g_a_proj":
                    target_param = target_module.g_proj[0].weight
                elif param_name == "g_b_proj":
                    target_param = target_module.g_proj[1].weight
                elif param_name == "f_a_proj":
                    target_param = target_module.f_proj[0].weight
                elif param_name == "f_b_proj":
                    target_param = target_module.f_proj[1].weight
                else:
                     # recursive lookup or direct
                     curr = target_module
                     found = True
                     for p in param_parts:
                         if hasattr(curr, p):
                             curr = getattr(curr, p)
                         else:
                             found = False
                             break
                     if found:
                         if isinstance(curr, torch.nn.Parameter):
                             target_param = curr
                         elif isinstance(curr, torch.nn.Module):
                             target_param = curr.weight
                
                if target_param is not None:
                    # Match dtype and device
                    weight = weight.to(dtype=target_param.dtype, device=target_param.device)

                    # Handle shape mismatch if any
                    if weight.shape != target_param.shape:
                        if weight.numel() == target_param.numel():
                            weight = weight.view(target_param.shape)

                    target_param.data.copy_(weight)

            except Exception as e:
                logger.warning(f"Failed to load KDA weight {name}: {e}")
                continue
