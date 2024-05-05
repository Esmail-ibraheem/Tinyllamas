import math
from time import perf_counter
import warnings
from typing import List, Optional, Tuple, Union 
   
import torch  
import torch.nn.functional as F
import torch.utils.checkpoint  
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss 
 
from configuration_llama import LlamaConfig

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

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input 

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(seqlens_in_batch.flatten(), as_tuple=False).flatten()
    max_seqlens_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1,0))
    return (
        indices,
        cu_seqlens,
        max_seqlens_in_batch
    )

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_eps = eps 
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype 
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_eps).type_as()
        return self.weight * hidden_states.to(input_dtype)

class LlamaChatRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps 
    
    def _norm(self, x:torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x:torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)
ALL_LAYERNORM_LAYERS.append(LlamaChatRMSNorm)

class LlamaRotaryEmbeddings(nn.Module):
    def __init__(self, dim, max_position_embedding=2048, base=10000, device=None, scaling_factor=1.0) -> None:
        super().__init__()
        self.dim = dim 
        self.max_position_embedding = max_position_embedding
        self.scaling_factor = scaling_factor
        self.base = base 
        self.inv_frequency = 1.0 / (self.base ** (torch.arange(0, dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_frequency", self.inv_frequency, persistent=False)
        self.max_seq_len_cached = max_position_embedding
        t = torch.arange(0, max_position_embedding, device=device, dtype=torch.int64)
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
            "The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self.cos_cached

    @torch.no_grad
    def forward(self, x, position_ids):
        inv_frequence_expanded = self.inv_frequency[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[None, :, None].float()
        device_type = x.device.type 
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_frequence_expanded.float() @ position_ids_expanded.float()).transpose(1,2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class LlamaLinearScalingRotaryEmbeddings(LlamaRotaryEmbeddings):
    def forward(self, x, position_ids):
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin
