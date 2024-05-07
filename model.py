from cmath import cos
import math
from time import perf_counter  
import warnings  
from typing import List, Optional, Tuple, Union 
  
import torch 
import torch.nn.functional as F
import torch.utils.checkpoint 
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import(
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from config import LlamaConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input 

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

def _get_unpad_data(attention_mask):
    seqlen_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(seqlen_in_batch.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlen_in_batch.max().item()
    cu_seqlen = F.pad(torch.cumsum(seqlen_in_batch, dim=0, dtype=torch.int32), (1,2))
    return (
        indices,
        cu_seqlen,
        max_seqlen_in_batch
    )

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_eps = eps 
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.int32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_eps)
        return self.weight * hidden_states.to(input_dtype)

class LlamaFixedRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps 
    
    def _norm(self, x:torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x:torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)
ALL_LAYERNORM_LAYERS.append(LlamaFixedRMSNorm)

class LlamaRotaryEmbeddings(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0) -> None:
        super().__init__()
        self.dim = dim 
        self.max_position_embeddings = max_position_embeddings
        self.base = base 
        self.scaling_factor = scaling_factor
        self.inv_frequency = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_frequency", self.inv_frequency, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(0, self.max_seq_len_cached, device=device)
        t = t / scaling_factor
        freqs = torch.outer(t, self.inv_frequency)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)
    
    @property
    def sin_cached(self):
        logger.warning_once(
            "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self.sin_cached
    
    @property
    def cos_cached(self):
        logger.warning_once(
            "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self.cos_cached

    @torch.no_grad
    def forward(self, x, position_ids):
        inv_frequency_expanded = self.inv_frequency[None, :, None].float().expanded(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[None, :, None].float()
        device_dtype = x.device.dtype 
        device_dtype = device_dtype if isinstance(device_dtype, str) and device_dtype != "mps" else "cpu"
        with torch.autocast(device_type=device_dtype, enabled=False):
            freqs = (inv_frequency_expanded.float() @ position_ids_expanded.float()).transpose(1,2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class LlamaLinearScalingRotaryEmbeddings(LlamaRotaryEmbeddings):
    def forward(self, x, position_ids):
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin 

class LlamaDynamicNTKScalingRotaryEmbeddings(LlamaRotaryEmbeddings):
    def forward(self, x, position_ids):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * ((self.scaling_factor ** (seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)) ** (self.dim / (self.dim - 2)))
            inv_frequency = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(x.device) / self.dim))
            self.register_buffer("inv_frequency", inv_frequency, persistent=False)
            cos, sin = super().forward(x, position_ids)
        return cos, sin 

ALL_ROTARY_EMBEDDINGS_CLASSES = {
    "rotart" : LlamaRotaryEmbeddings,
    "linear": LlamaLinearScalingRotaryEmbeddings,
    "dynamic" : LlamaDynamicNTKScalingRotaryEmbeddings
}

def percompute_theta_pos_frequncies(head_dim: int, seq_len:int, device: str, theta: float=10000.0):
    assert head_dim % 2 == 0 , "dimensions must be divisble by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim))
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embedding(x:torch.Tensor, device: str,freqs_complex: int):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = torch.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x:torch.Tensor, n_rep: int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape 
    if n_rep == 1:
        return x 
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    ) 

class LlamaScalableGroupedQueryAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config 
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_heads_groups = self.num_heads // self.num_kv_heads
        self.max_position_embedding = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True 

        if(self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.query_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.key_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.value_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()
    
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.emb_rope = LlamaRotaryEmbeddings(
                self.head_dim,
                max_position_embeddings=self.max_position_embedding,
                base=self.rope_theta
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.emb_rope = LlamaLinearScalingRotaryEmbeddings(
                    self.head_dim, 
                    max_position_embeddings=self.max_position_embedding,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta
                )
            elif scaling_type == "dynamic":
                self.emb_rope = LlamaDynamicNTKScalingRotaryEmbeddings(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embedding,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta
                )
            else:
                raise ValueError(f"Unknow scaling type of RoPE: {scaling_type}")
