from transformers import DeepseekV2Config

class DeepseekHybridConfig(DeepseekV2Config):
    model_type = "custom_deepseek_v3"

    def __init__(
        self,
        kda_residual_last_n_layers=0,
        linear_attn_config=None,
        **kwargs
    ):
        self.kda_residual_last_n_layers = kda_residual_last_n_layers
        self.linear_attn_config = linear_attn_config
        super().__init__(**kwargs)
