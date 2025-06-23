"""
GeneralMHA class
Return transformer model with MHA
"""
from typing import Optional, Tuple
import os

import torch
import torch.nn as nn
import torchtune.modules as ttm

from torchtune.modules import RMSNorm
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
from torchtune.models.qwen2._positional_embeddings import Qwen2RotaryPositionalEmbeddings
from torchtune.modules import RotaryPositionalEmbeddings
from xotorch.inference.shard import Shard
from xotorch.inference.torch.models.llm_utils import (
  layer_mlp,
  ShardTransformerDecoder
)

from xotorch.helpers import DEBUG

def GeneralMHA(
    config: dict,
    shard: Shard
):
  use_tied = False
  attn_bias = config.get("attn_bias", False)
  output_bias = config.get("attn_bias", False)

  if DEBUG >= 2: print(f"GeneralMHA loading model {shard.model_id}")

  if "llama" in shard.model_id or "Llama" in shard.model_id:
    # rope scaling config
    rope = Llama3ScaledRoPE(
      dim=config["head_dim"],
      max_seq_len=config["max_seq_len"],
      base=config["rope_base"],
      scale_factor=config["rope_scaling_factor"],
    )

    # tied needed for 3.2 llama models
    if "3.2" in shard.model_id:
      use_tied = True
  elif "qwen" in shard.model_id or "Qwen" in shard.model_id:
    # rope scaling config
    rope = Qwen2RotaryPositionalEmbeddings(
      dim=config["head_dim"],
      max_seq_len=config["max_seq_len"],
      base=config["rope_base"]
    )
    attn_bias = True
    output_bias = False

    # tied needed for 0.5B qwen models
    if "0.5B" in shard.model_id or "0.5b" in shard.model_id:
      use_tied = True
  else:
    rope = RotaryPositionalEmbeddings(
      dim=config["head_dim"],
      max_seq_len=config["max_seq_len"],
      base=config["rope_base"]
    )
  
  if DEBUG >= 4:
    print(f"model_id: {shard.model_id}")
    print(f"rope: {rope}")
    print(f"attn_bias: {attn_bias}")
    print(f"output_bias: {output_bias}")
    print(f"use_tied: {use_tied}")

  # hack to align sharded weights with layers
  # fill unused layer positions with None
  layers = [None for _ in range(shard.n_layers)]

  # build layers
  for i in range(shard.start_layer, shard.end_layer + 1):
    self_attn = ttm.MultiHeadAttention(
      embed_dim=config["embed_dim"],
      num_heads=config["num_heads"],
      num_kv_heads=config["num_kv_heads"],
      head_dim=config["head_dim"],
      q_proj=nn.Linear(
        config["embed_dim"],
        config["num_heads"]*config["head_dim"],
        bias=attn_bias,
      ),
      k_proj=nn.Linear(
        config["embed_dim"],
        config["num_kv_heads"]*config["head_dim"],
        bias=attn_bias,
      ),
      v_proj=nn.Linear(
        config["embed_dim"],
        config["num_kv_heads"]*config["head_dim"],
        bias=attn_bias,
      ),
      output_proj=nn.Linear(
        config["embed_dim"],
        config["embed_dim"],
        bias=output_bias,
      ),
      max_seq_len=config["max_seq_len"],
      attn_dropout=config["attn_dropout"],
      pos_embeddings=rope,
    )

    mlp = layer_mlp(
      dim=config["embed_dim"],
      hidden_dim=config["intermediate_dim"],
    )

    layer = ttm.TransformerSelfAttentionLayer(
      attn=self_attn,
      mlp=mlp,
      sa_norm=RMSNorm(config["embed_dim"], eps=config["norm_eps"]),
      mlp_norm=RMSNorm(config["embed_dim"], eps=config["norm_eps"]),
    )

    layers[i] = layer

  layers = nn.ModuleList(layers)

  tok_embeddings = nn.Embedding(config["vocab_size"], config["embed_dim"])
  if use_tied:
    output_proj = ttm.TiedLinear(tok_embeddings)
  else:
    output_proj = nn.Linear(config["embed_dim"], config["vocab_size"], bias=False)

  norm = RMSNorm(config["embed_dim"], eps=config["norm_eps"])

  return ShardTransformerDecoder(
    tok_embeddings=tok_embeddings,
    shard=shard,
    layers=layers,
    max_seq_len=config["max_seq_len"],
    num_heads=config["num_heads"],
    head_dim=config["head_dim"],
    norm=norm,
    output=output_proj,
    num_layers=config["num_layers"],
  )

class ShardedGeneralModel(nn.Module):
  def __init__(
    self,
    config: dict,
    shard: Shard,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = None,
    use_cache: Optional[bool] = False,
    max_generated_tokens: int = 1024
  ):
    super(ShardedGeneralModel, self).__init__()

    self.shard = shard
    self.config = config
    
    # Check if we should force FP16
    force_fp16 = bool(os.getenv("TORCH_FORCE_FP16", "False").lower() == "true")
    
    # Determine the appropriate dtype
    if dtype is None:
      if force_fp16 and (device is not None and device.type == 'cuda'):
        self.dtype = torch.float16
        if DEBUG >= 2:
          print(f"Forcing FP16 for model weights")
      else:
        self.dtype = torch.bfloat16
    else:
      self.dtype = dtype
    
    self.device = device if device is not None else torch.device("cpu")
    self.max_seq_len = self.config["max_seq_len"]
    self.use_cache = use_cache
    
    # Create the model with the determined dtype
    self.model = GeneralMHA(
      config,
      self.shard
    ).to(
      dtype=self.dtype,
      device=self.device
    )
    
    # Ensure KV cache uses the same dtype
    if hasattr(self.model, 'caches_are_enabled') and self.model.caches_are_enabled():
      if DEBUG >= 3:
        print(f"Resetting KV cache with dtype: {self.dtype}")
      self.model.reset_caches()
    
    # Apply monkey patch to fix dtype mismatch in KV cache
    if force_fp16 and self.device.type == 'cuda':
      self._patch_kv_cache_for_fp16()
      if DEBUG >= 2:
        print("Applied FP16 patch to KV cache")
        
  def _patch_kv_cache_for_fp16(self):
    """
    Monkey patch the KV cache update method to ensure dtype consistency.
    This is needed because the torchtune library's KV cache might use BF16
    while our model is using FP16, causing dtype mismatches.
    """
    import types
    from functools import wraps
    
    # Find all attention layers with KV cache
    for name, module in self.model.named_modules():
      if hasattr(module, 'kv_cache') and hasattr(module.kv_cache, 'update'):
        original_update = module.kv_cache.update
        
        # Create a wrapper that ensures dtype consistency
        @wraps(original_update)
        def patched_update(self, k_val, v_val):
          # Convert inputs to match cache dtype if needed
          k_cache_dtype = self.k_cache.dtype
          v_cache_dtype = self.v_cache.dtype
          
          if k_val.dtype != k_cache_dtype:
            if DEBUG >= 3:
              print(f"Converting k_val from {k_val.dtype} to {k_cache_dtype}")
            k_val = k_val.to(dtype=k_cache_dtype)
            
          if v_val.dtype != v_cache_dtype:
            if DEBUG >= 3:
              print(f"Converting v_val from {v_val.dtype} to {v_cache_dtype}")
            v_val = v_val.to(dtype=v_cache_dtype)
            
          # Call the original update method with dtype-matched tensors
          return original_update(k_val, v_val)
        
        # Replace the original update method with our patched version
        module.kv_cache.update = types.MethodType(patched_update, module.kv_cache)
        
        if DEBUG >= 3:
          print(f"Patched KV cache in {name}")

    if DEBUG >= 4:
      print("ShardedGeneralModel called")
      print(f"self.model {self.model}")

    # keep track of current position in generation
    self.max_generated_tokens = max_generated_tokens

  def generate(
    self,
    tokens: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    input_pos: Optional[torch.Tensor] = None,
    hidden_state: Optional[torch.Tensor] = None,
    curr_pos: Optional[int] = 0
  ) -> Tuple[
    Optional[torch.Tensor],
    torch.Tensor,
  ]:
    """
    Generate logits and/or hidden_states from llama model

    Args
      tokens (torch.Tensor) - tokens from prompt tokenization and generation
      hidden_state (torch.Tensor, optional) - hidden state from last activated hidden layer, if any
    """
    if DEBUG >= 4:
      print("generate called")
      print(f"tokens: {tokens}")
      if mask is not None:
        print(f"mask: {mask.size()}")
        print(f"input_pos: {input_pos.size()}") 
      print(f"hidden_state: {hidden_state}")
      print(f"curr_pos: {curr_pos}")
      print(f"cached? {self.model.caches_are_enabled()}")

    model_hs = None
    model_logits = None

    self.model.output_hidden_states = [self.shard.end_layer]

    if curr_pos > 0:
      if self.model.caches_are_enabled():
        input_pos = input_pos[:, curr_pos].contiguous()
        mask = mask[:, curr_pos, None, :].contiguous()
      else:
        input_pos = input_pos[:, :curr_pos + 1]
        mask = mask[:, :curr_pos + 1, :curr_pos + 1]
    else:
      _, tklng = tokens.size()

      if self.model.caches_are_enabled():
        mask = mask[:, :tklng]
      else:
        mask = mask[:, :tklng, :tklng]

      input_pos = input_pos[:, :tklng].squeeze()

    if DEBUG >= 4:
      print("model_input")
      if tokens is not None:
        print(f"tokens: {tokens}")
      if hidden_state is not None:
        print(f"hidden_state: {hidden_state}")
      print(f"mask: {mask}")
      print(f"input_pos: {input_pos}")

    with torch.no_grad():
      model_output = self.model(
        tokens=tokens,
        mask=mask,
        input_pos=input_pos,
        hidden_state=hidden_state,
        dtype=self.dtype
      )

    if self.shard.is_last_layer():
      model_logits = model_output
    else:
      model_hs = model_output

    if DEBUG >= 4:
      print(f"model_hs\n{model_hs}\nmodel_logits\n{model_logits}")

    return model_hs, model_logits
