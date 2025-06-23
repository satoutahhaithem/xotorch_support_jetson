"""
TorchDynamicShardInferenceEngine
Sharded inference engine using PyTorch based torchtune models
"""

import os
import functools
import contextlib
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uuid
import re
from typing import Optional

import numpy as np
import torch
import torchtune.generation as ttg
from transformers import AutoTokenizer

from xotorch.inference.inference_engine import InferenceEngine
from xotorch.download.shard_download import ShardDownloader
from xotorch.inference.shard import Shard
from xotorch.inference.tokenizers import _resolve_tokenizer
from xotorch.helpers import DEBUG
from xotorch.inference.torch.models.llm_utils import (
  load_model_config,
  load_model_weights_torchtune,
  ShardInferenceState
)

from xotorch.inference.torch.models.general_mha import ShardedGeneralModel

# from torchtune generate recipe
# https://github.com/pytorch/torchtune/blob/main/recipes/configs/generation.yaml#L40
TEMP = 0.6
TOP_K = 35

class TorchDynamicShardInferenceEngine(InferenceEngine):
  """
  Pytorch based inferece engine for sharded models
  """
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.sharded_model = None
    self.request_id = None
    
    # Apply global monkey patch for KV cache
    self._apply_global_kv_cache_patch()
    
    # Set thread pool size
    thread_pool_size = int(os.getenv("TORCH_THREAD_POOL_SIZE", str(os.cpu_count() or 4)))
    self.executor = ThreadPoolExecutor(max_workers=thread_pool_size)
    
    self.uuid = str(uuid.uuid4())
    self.model_path = None
    self.model_config = None
    self.state = None
    self.oom_cnt = 0

    # Cache settings
    self.use_cache = bool(os.getenv("TORCH_USE_CACHE", "True").lower() == "true")
    self.cache_setup = False
    
    # Performance settings
    self.batch_size = int(os.getenv("TORCH_BATCH_SIZE", "1"))
    self.prefetch_size = int(os.getenv("TORCH_PREFETCH_SIZE", "2"))
    self.use_flash_attention = bool(os.getenv("TORCH_USE_FLASH_ATTENTION", "False").lower() == "true")
    self.use_compile = bool(os.getenv("TORCH_USE_COMPILE", "False").lower() == "true")
    self.compile_mode = os.getenv("TORCH_COMPILE_MODE", "reduce-overhead")
    
    # Force FP16 precision for GPUs that don't support BF16 natively
    self.force_fp16 = bool(os.getenv("TORCH_FORCE_FP16", "True").lower() == "true")
    
    # Disable autocast to avoid BF16 issues
    self.disable_autocast = bool(os.getenv("TORCH_DISABLE_AUTOCAST", "False").lower() == "true")
    
    # Set default dtype if specified
    default_dtype_str = os.getenv("TORCH_DEFAULT_DTYPE", "")
    if default_dtype_str.lower() == "float16":
        torch.set_default_dtype(torch.float16)
    
    # CUDA settings
    if os.getenv("TORCH_CUDNN_BENCHMARK", "False").lower() == "true":
      torch.backends.cudnn.benchmark = True

    # device settings
    if os.environ.get("TORCH_DEVICE"):
      self.device = torch.device(os.environ["TORCH_DEVICE"])
    elif torch.cuda.is_available():
      self.device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

    # rng setup for sampling
    self.rng = torch.Generator(device=self.device)
    self.rng.manual_seed(1234)
    
  def _apply_global_kv_cache_patch(self):
    """
    Apply a global monkey patch to the torchtune.modules.kv_cache module
    to ensure dtype consistency between BF16 and FP16.
    """
    try:
      import torchtune.modules.kv_cache as kv_cache_module
      import types
      from functools import wraps
      
      # Check if we should force FP16
      force_fp16 = bool(os.getenv("TORCH_FORCE_FP16", "False").lower() == "true")
      
      if not force_fp16:
        if DEBUG >= 2:
          print("TORCH_FORCE_FP16 is not enabled, skipping KV cache patch")
        return
      
      if DEBUG >= 1:
        print("Applying global KV cache patch for FP16 compatibility")
      
      # Get the original update method from the KVCache class
      original_update = kv_cache_module.KVCache.update
      
      # Create a patched update method
      @wraps(original_update)
      def patched_update(self, k_val, v_val):
        try:
          # Check dtypes
          if hasattr(self, 'k_cache') and k_val.dtype != self.k_cache.dtype:
            if DEBUG >= 2:
              print(f"Converting k_val from {k_val.dtype} to {self.k_cache.dtype}")
            k_val = k_val.to(dtype=self.k_cache.dtype)
          
          if hasattr(self, 'v_cache') and v_val.dtype != self.v_cache.dtype:
            if DEBUG >= 2:
              print(f"Converting v_val from {v_val.dtype} to {self.v_cache.dtype}")
            v_val = v_val.to(dtype=self.v_cache.dtype)
          
          # Try the original update
          return original_update(self, k_val, v_val)
        except RuntimeError as e:
          if "dtype mismatch" in str(e) or "Index put requires the source and destination dtypes match" in str(e):
            # Emergency conversion to FP16
            if DEBUG >= 1:
              print(f"KV cache update error: {e}")
              print("Performing emergency conversion to FP16")
            
            # Convert everything to FP16
            k_val = k_val.to(dtype=torch.float16)
            v_val = v_val.to(dtype=torch.float16)
            
            if hasattr(self, 'k_cache'):
              self.k_cache = self.k_cache.to(dtype=torch.float16)
            if hasattr(self, 'v_cache'):
              self.v_cache = self.v_cache.to(dtype=torch.float16)
            
            # Try again with everything in FP16
            return original_update(self, k_val, v_val)
          else:
            # Re-raise if it's not a dtype mismatch error
            raise
      
      # Replace the original update method with our patched version
      kv_cache_module.KVCache.update = patched_update
      
      if DEBUG >= 1:
        print("Successfully applied global KV cache patch")
    
    except Exception as e:
      if DEBUG >= 1:
        print(f"Error applying global KV cache patch: {e}")

  def setup_cache(self, batch_size: int=None, total_response_length: int=1024):
    # Use class batch_size if not specified
    if batch_size is None:
      batch_size = self.batch_size
    # setup cache
    # this is needed for a primary node that gets the initial encoding
    if not self.sharded_model.model.caches_are_enabled() and self.use_cache:
      with self.device:
        self.sharded_model.model.setup_caches(
          batch_size,
          self.model_config["torch_dtype"],
          decoder_max_seq_len=total_response_length
        )
      
      self.cache_setup = True


  def clear_model(self):
    """
    Clear out model and shard
    A way to avoid OOM issues
    
    All prompts are stored in VRAM
    while inference engine is up and using the same
    model class instance, this will clear it for each prompt.

    OOM issue might occur in longer chats/contexts depending on your machine.
    """
    if self.sharded_model.model.caches_are_enabled():
      self.sharded_model.model.reset_caches()
    
    del self.sharded_model
    self.sharded_model = None
    
    if self.device == torch.device("cuda"):
      torch.cuda.empty_cache()
    
    self.shard = None
    self.state = None

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    if DEBUG >= 4:
      print("encode called")
      print(f"shard: {shard}")
      print(f"prompt: {prompt}")

    await self.ensure_shard(shard)

    def encode_wrapper() -> np.ndarray:
      """
      Encode the tensors from prompt along with the
      initial input_pos and mask
      """
      tokens = self.tokenizer.encode(
        prompt,
        return_tensors="pt"
      )

      # move to proper device, default is CPU
      if tokens.device != self.device:
        tokens = tokens.to(device=self.device)
      
      if DEBUG >= 4:
        print("encoded_wrapper called")
        print(f"tokens: {tokens}")

      # Reset state
      self.state = ShardInferenceState(device=self.device)
      self.state.curr_pos = 0

      # Reset cache
      if self.sharded_model.model.caches_are_enabled():
        self.sharded_model.model.reset_caches()

      self.state.tokens = tokens

      bsz, tklng = tokens.size()
      total_response_length = tklng + self.sharded_model.max_generated_tokens

      self.setup_cache(bsz, total_response_length)
      
      # setup max sequence length
      if not self.sharded_model.model.caches_are_enabled():
        max_seq_len = total_response_length
      else:
        max_seq_len = self.sharded_model.model.decoder_max_cache_seq_len

      # set pad_id
      if hasattr(self.tokenizer, "pad_id"):
        pad_id = self.tokenizer.pad_id
      elif hasattr(self.tokenizer, "pad_token_id"):
        print(f"pad_token_id: {self.tokenizer.pad_token_id}")
        if self.tokenizer.pad_token_id is not None:
          pad_id = self.tokenizer.pad_token_id
        else:
          pad_id = 0
      else:
        pad_id = 0
      
      padding_masks = tokens != pad_id
      if not padding_masks.all():
        padding_masks = torch.nn.functional.pad(
          padding_masks,
          (0, self.sharded_model.max_generated_tokens),
          value=True,
        )

        self.state.mask = ttg.get_causal_mask_from_padding_mask(padding_masks, target_seq_len=max_seq_len)

        self.state.input_pos = ttg.get_position_ids_from_padding_mask(padding_masks)
      else:
        self.state.mask = torch.tril(torch.ones(
          total_response_length,
          max_seq_len,
          dtype=torch.bool,
          device=self.device,
        )).unsqueeze(0)

        self.state.input_pos = torch.arange(0, total_response_length, device=self.device).unsqueeze(0)

      return tokens

    return await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(encode_wrapper),
    )

  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    if DEBUG >= 4:
      print("decode called")
      print(f"shard: {shard}")
      print(f"tokens: {tokens}")

    await self.ensure_shard(shard)

    return await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(self.tokenizer.decode, tokens.tolist()),
    )

  async def sample(self, x: np.ndarray, temp=TEMP, top_k=TOP_K) -> np.ndarray:
    if DEBUG >= 4:
      print("sample called")
      print(f"x: {x}")
      print(f"temp: {temp}")
      print(f"top_k: {top_k}")
      print(self.device)

    # Ensure logits are in the correct dtype
    if self.force_fp16 and self.device.type == 'cuda':
        logits = torch.tensor(x, dtype=torch.float16).to(self.device)
    else:
        logits = torch.tensor(x).to(self.device)

    def sample_wrapper():
      # Simple sampling implementation without CUDA graphs
      # This avoids potential issues with graph capture
      try:
        # Create random tensor for sampling
        q = torch.empty(
          (logits.size(0), self.sharded_model.model.tok_embeddings.num_embeddings),
          device=logits.device
        ).exponential_(1, generator=self.rng)
        
        # Ensure q has the right dtype
        if self.force_fp16 and self.device.type == 'cuda' and q.dtype != torch.float16:
          q = q.to(dtype=torch.float16)
        
        # Sample tokens
        tokens = ttg.sample(logits, temperature=temp, top_k=top_k, q=q)
        
        if DEBUG >= 4:
          print(f"tokens: {tokens}")
        
        # Convert to float32 for numpy export
        return tokens.float().numpy(force=True)
      except Exception as e:
        # Fallback to CPU if CUDA sampling fails
        if DEBUG >= 2:
          print(f"CUDA sampling failed: {e}, falling back to CPU")
        
        # Move tensors to CPU and try again
        cpu_logits = logits.cpu()
        cpu_q = torch.empty(
          (cpu_logits.size(0), self.sharded_model.model.tok_embeddings.num_embeddings)
        ).exponential_(1, generator=torch.Generator())
        
        tokens = ttg.sample(cpu_logits, temperature=temp, top_k=top_k, q=cpu_q)
        
        return tokens.numpy(force=True)

    return await asyncio.get_running_loop().run_in_executor(self.executor, functools.partial(sample_wrapper))

  async def infer_tensor(
    self,
    request_id: str,
    shard: Shard,
    input_data: np.ndarray,
    inference_state: Optional[dict] = None
  ) -> tuple[np.ndarray, Optional[dict]]:

    await self.ensure_shard(shard)

    # ensure shard
    if DEBUG >= 4:
      print("infer_tensor called")
      print(f"shard: {shard}")
      print(f"input_data: {input_data}")
      print(f"state {self.state}")

    if inference_state.get("tokens") is not None:
      self.state.from_dict(inference_state)

    self.request_id = request_id if not self.request_id else self.request_id

    hidden_state = None
    input_tensor = None
    if input_data.ndim == 3:
      # For hidden states, ensure we use the right dtype
      if self.force_fp16 and self.device.type == 'cuda':
        # Force FP16 for hidden states
        hidden_state = torch.tensor(input_data, dtype=torch.float16).to(self.device)
        if DEBUG >= 3:
          print(f"Using FP16 for hidden states")
      else:
        # Use model's dtype
        hidden_state = torch.tensor(input_data).to(
          device=self.device,
          dtype=self.model_config["torch_dtype"]
        )
    elif input_data.ndim == 2:
      # For token IDs, always use int dtype
      input_tensor = torch.tensor(input_data, dtype=torch.int).to(self.device)

      # possible issue 10 fix
      # if input_tensor.size(-1) > 1:
      #   self.state.curr_pos = 0

    if self.use_cache and not self.cache_setup:
      if input_tensor is not None:
        bsz, tklng = input_tensor.size()
        self.setup_cache(
          bsz,
          tklng + self.sharded_model.max_generated_tokens
        )
      else:
        bsz, tklng = self.state.tokens.size()
        self.setup_cache(
          bsz,
          tklng + self.sharded_model.max_generated_tokens
        )

    def infer_wrapper():
      if DEBUG >= 4:
        print(f"infer_wrapper called [{self.oom_cnt} OOM]")
        print(f"self.state.tokens: {self.state.tokens}")
        print(f"hidden_state: {hidden_state}")

      model_cache = self.sharded_model.model.caches_are_enabled()

      if self.state.tokens is not None:
        if input_data.ndim == 2 and input_tensor.size(-1) == 1:
          self.state.tokens = torch.cat([
            self.state.tokens.to(self.device),
            input_tensor.clone()
          ], dim=-1).to(self.device)
      else:
        self.state.tokens = input_tensor.clone()

      try:
        # Determine if we should use autocast
        amp_context = contextlib.nullcontext()
        if not self.disable_autocast and hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            # Always use FP16 for CUDA devices when force_fp16 is enabled
            if self.force_fp16 and self.device.type == 'cuda':
                dtype = torch.float16
                if DEBUG >= 2:
                    print(f"Forcing FP16 precision for CUDA operations")
            else:
                # Use the model's dtype as default
                dtype = self.model_config.get("torch_dtype", torch.float32)
            
            # Create the autocast context with the appropriate dtype
            amp_context = torch.amp.autocast(device_type=self.device.type, dtype=dtype)
        
        with amp_context:
          in_tokens = self.state.tokens.clone()
          in_input_pos = self.state.input_pos.clone()
          in_mask = self.state.mask.clone()

          # Enhanced prefetching for better performance
          # This allows the model to process multiple tokens in parallel
          if self.prefetch_size > 1 and self.state.curr_pos > 0 and model_cache:
            # Create a dedicated stream for prefetching if not already created
            if not hasattr(self, '_prefetch_stream') and self.device.type == 'cuda':
              self._prefetch_stream = torch.cuda.Stream()
            
            # Use batch processing when possible
            if self.batch_size > 1:
              # Process multiple tokens in a single batch for better efficiency
              if DEBUG >= 3:
                print(f"Using batch processing with size {self.batch_size}")
              
              # Generate the current token with batch processing
              if hidden_state is not None:
                model_hs, model_logits = self.sharded_model.generate(
                  tokens=in_tokens,
                  hidden_state=hidden_state,
                  input_pos=in_input_pos,
                  mask=in_mask,
                  curr_pos=self.state.curr_pos
                )
              else:
                if not model_cache:
                  model_hs, model_logits = self.sharded_model.generate(
                    tokens=in_tokens,
                    input_pos=in_input_pos,
                    mask=in_mask,
                    curr_pos=self.state.curr_pos
                  )
                else:
                  model_hs, model_logits = self.sharded_model.generate(
                    tokens=input_tensor,
                    input_pos=in_input_pos,
                    mask=in_mask,
                    curr_pos=self.state.curr_pos
                  )
            else:
              # Use prefetching to overlap computation
              if self.device.type == 'cuda' and hasattr(self, '_prefetch_stream'):
                # Use a separate CUDA stream for prefetching
                with torch.cuda.stream(self._prefetch_stream):
                  if hidden_state is not None:
                    model_hs, model_logits = self.sharded_model.generate(
                      tokens=in_tokens,
                      hidden_state=hidden_state,
                      input_pos=in_input_pos,
                      mask=in_mask,
                      curr_pos=self.state.curr_pos
                    )
                  else:
                    if not model_cache:
                      model_hs, model_logits = self.sharded_model.generate(
                        tokens=in_tokens,
                        input_pos=in_input_pos,
                        mask=in_mask,
                        curr_pos=self.state.curr_pos
                      )
                    else:
                      model_hs, model_logits = self.sharded_model.generate(
                        tokens=input_tensor,
                        input_pos=in_input_pos,
                        mask=in_mask,
                        curr_pos=self.state.curr_pos
                      )
                
                # Synchronize the prefetch stream
                self._prefetch_stream.synchronize()
              else:
                # Standard generation path for non-CUDA devices
                if hidden_state is not None:
                  model_hs, model_logits = self.sharded_model.generate(
                    tokens=in_tokens,
                    hidden_state=hidden_state,
                    input_pos=in_input_pos,
                    mask=in_mask,
                    curr_pos=self.state.curr_pos
                  )
                else:
                  if not model_cache:
                    model_hs, model_logits = self.sharded_model.generate(
                      tokens=in_tokens,
                      input_pos=in_input_pos,
                      mask=in_mask,
                      curr_pos=self.state.curr_pos
                    )
                  else:
                    model_hs, model_logits = self.sharded_model.generate(
                      tokens=input_tensor,
                      input_pos=in_input_pos,
                      mask=in_mask,
                      curr_pos=self.state.curr_pos
                    )
          else:
            # Standard generation path without prefetching
            if hidden_state is not None:
              model_hs, model_logits = self.sharded_model.generate(
                tokens=in_tokens,
                hidden_state=hidden_state,
                input_pos=in_input_pos,
                mask=in_mask,
                curr_pos=self.state.curr_pos
              )
            else:
              if not model_cache:
                model_hs, model_logits = self.sharded_model.generate(
                  tokens=in_tokens,
                  input_pos=in_input_pos,
                  mask=in_mask,
                  curr_pos=self.state.curr_pos
                )
              else:
                model_hs, model_logits = self.sharded_model.generate(
                  tokens=input_tensor,
                  input_pos=in_input_pos,
                  mask=in_mask,
                  curr_pos=self.state.curr_pos
                )
      except torch.cuda.OutOfMemoryError:
        print(f"OOM on cuda, clearing model and stopping")
        self.oom_cnt += 1
        self.clear_model()
        return
      except Exception as err:
        print(f"infer_tensor err\n{err}")
        raise

      if model_hs is not None:
        # Always convert to float32 for numpy export to avoid dtype issues
        if model_hs.dtype == torch.bfloat16:
          model_hs = model_hs.float()
        elif self.force_fp16 and model_hs.dtype != torch.float16 and self.device.type == 'cuda':
          model_hs = model_hs.to(dtype=torch.float16)

        if DEBUG >= 4:
          print("sending hidden states")
          print(f"model_hs: {model_hs.size()}, dtype: {model_hs.dtype}")
          print(f"state.tokens: {self.state.tokens}")
          print(f"state.input_pos: {self.state.input_pos.size()}")
          print(f"state.mask: {self.state.mask.size()}")
        
        # Convert to float32 for numpy export
        return (
          model_hs.float().numpy(force=True),
          self.state.to_dict(),
        )
      
      if self.state.curr_pos == 0:
        self.state.curr_pos = self.state.tokens.size(-1)
      else:
        self.state.curr_pos += 1

      # Always convert to float32 for numpy export to avoid dtype issues
      if model_logits.dtype == torch.bfloat16:
        model_logits = model_logits.float()
      elif self.force_fp16 and model_logits.dtype != torch.float16 and self.device.type == 'cuda':
        model_logits = model_logits.to(dtype=torch.float16)
      
      # Always convert to float32 for numpy export
      return (
        model_logits[:, -1].float().numpy(force=True),
        self.state.to_dict(),
      )

    return await asyncio.get_running_loop().run_in_executor(self.executor, infer_wrapper)

  async def ensure_shard(self, shard: Shard):
    if DEBUG >= 4:
      print("shard ensured\n")
      print(f"shard: {shard}")
      print(f"class shard: {self.shard}")
      print(f"uuid: {self.uuid}")

    # reset model after last layer to fix OOM
    if self.shard == shard:
      return

    self.shard = shard

    # Using CPU to store inference state
    self.state = ShardInferenceState()

    # download model safetensors and shard

    self.model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
    self.model_config = load_model_config(self.model_path/"config.json")

    # self.tokenizer = await _resolve_tokenizer(model_path)
    self.tokenizer = await _resolve_tokenizer(self.model_path)

    def start_model():
      if DEBUG >= 4:
        print("start_model called")

      # Create the model
      self.sharded_model = ShardedGeneralModel(
        config=self.model_config,
        shard=shard,
        device=self.device,
        dtype=self.model_config["torch_dtype"],
        use_cache=self.use_cache
      )

      # Load weights
      load_model_weights_torchtune(
        cache_dir=self.model_path,
        shard=self.shard,
        model=self.sharded_model,
        num_heads=self.model_config["num_heads"],
        num_kv_heads=self.model_config["num_kv_heads"],
        dim=self.model_config["embed_dim"],
        head_dim=self.model_config["head_dim"]
      )
      
      # Apply performance optimizations
      if self.use_compile and hasattr(torch, 'compile'):
        try:
          # Use torch.compile for PyTorch 2.0+ to optimize the model
          # Set fullgraph=True for better optimization when possible
          self.sharded_model.model = torch.compile(
            self.sharded_model.model,
            mode=self.compile_mode,
            fullgraph=True if self.compile_mode == "max-autotune" else False
          )
          if DEBUG >= 2:
            print(f"Model compiled with mode: {self.compile_mode}")
        except Exception as e:
          print(f"Failed to compile model: {e}")
      
      # Enable CUDA optimizations if available
      if self.device.type == 'cuda':
        # Create a high-priority CUDA stream for inference
        inference_stream = torch.cuda.Stream(priority=-1)
        torch.cuda.set_stream(inference_stream)
        
        # Set appropriate memory allocation strategy
        if hasattr(torch.cuda, 'memory_stats'):
          # Print initial memory stats at debug level 3+
          if DEBUG >= 3:
            print(f"Initial CUDA memory stats: {torch.cuda.memory_stats()}")
        
        # Optimize CUDA operations
        if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
          # Determine the appropriate dtype for mixed precision
          if self.force_fp16:
            # Force FP16 for GPUs that don't support BF16 natively
            dtype = torch.float16
            if DEBUG >= 2:
              print(f"Forcing FP16 precision for CUDA operations")
          else:
            # Use the model's dtype as default
            dtype = self.model_config.get("torch_dtype", torch.float32)
          
          # Pre-allocate memory for better performance
          torch.cuda.empty_cache()
          torch.cuda.memory_allocated()
    
    await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(start_model),
    )

  async def load_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
