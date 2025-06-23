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
    
    # Optimize thread pool size for better parallelism
    thread_pool_size = int(os.getenv("TORCH_THREAD_POOL_SIZE", str(os.cpu_count() or 4)))
    self.executor = ThreadPoolExecutor(max_workers=thread_pool_size)
    
    self.uuid = str(uuid.uuid4())
    self.model_path = None
    self.model_config = None
    self.state = None
    self.oom_cnt = 0

    # Enhanced cache settings
    self.use_cache = bool(os.getenv("TORCH_USE_CACHE", "True").lower() == "true")
    self.cache_setup = False
    
    # Performance optimization settings
    self.batch_size = int(os.getenv("TORCH_BATCH_SIZE", "4"))
    self.prefetch_size = int(os.getenv("TORCH_PREFETCH_SIZE", "8"))
    self.use_flash_attention = bool(os.getenv("TORCH_USE_FLASH_ATTENTION", "True").lower() == "true")
    self.use_compile = bool(os.getenv("TORCH_USE_COMPILE", "True").lower() == "true")
    self.compile_mode = os.getenv("TORCH_COMPILE_MODE", "max-autotune")
    
    # Force FP16 precision for GPUs that don't support BF16 natively
    self.force_fp16 = bool(os.getenv("TORCH_FORCE_FP16", "False").lower() == "true")
    
    # Enable CUDA benchmark for kernel optimization
    if os.getenv("TORCH_CUDNN_BENCHMARK", "True").lower() == "true":
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
      # Use CUDA graphs for repeated sampling operations if available
      use_cuda_graph = (self.device.type == 'cuda' and
                        hasattr(torch.cuda, 'CUDAGraph') and
                        x.shape == getattr(self, '_last_sample_shape', None))
      
      # Initialize variables for CUDA graph
      static_input = False
      graph = None
      
      # Try to use CUDA graph for better performance
      if use_cuda_graph:
        static_input = True
        # Create a new graph if we don't have one yet
        if not hasattr(self, '_sample_graph'):
          try:
            # Create a stream specifically for the graph
            if not hasattr(self, '_graph_stream'):
              self._graph_stream = torch.cuda.Stream()
            
            # Use the dedicated stream for graph capture
            with torch.cuda.stream(self._graph_stream):
              # Pre-allocate tensors for graph capture
              self._logits_clone = logits.clone()
              self._q_tensor = torch.empty(
                (logits.size(0), self.sharded_model.model.tok_embeddings.num_embeddings),
                device=logits.device,
                dtype=torch.float32  # Keep q as float32 for numerical stability
              )
              
              # Create and capture the graph
              self._sample_graph = torch.cuda.CUDAGraph()
              with torch.cuda.graph(self._sample_graph):
                # Generate random values
                self._q_tensor.exponential_(1, generator=self.rng)
                # Sample tokens
                self._sample_output = ttg.sample(
                  self._logits_clone,
                  temperature=temp,
                  top_k=top_k,
                  q=self._q_tensor
                )
              
              # Store the graph for reuse
              graph = self._sample_graph
              
              if DEBUG >= 3:
                print("CUDA graph captured successfully for sampling")
          except Exception as e:
            if DEBUG >= 2:
              print(f"CUDA graph capture failed: {e}")
            static_input = False
            # Clean up failed graph attempt
            if hasattr(self, '_sample_graph'):
              del self._sample_graph
        else:
          # We already have a graph, use it
          graph = self._sample_graph
      
      # Execute the sampling operation
      if static_input and graph is not None:
        # Update the input tensor with current logits
        self._logits_clone.copy_(logits)
        
        # Reuse captured graph for better performance
        with torch.cuda.stream(self._graph_stream):
          graph.replay()
          # Wait for the graph to complete
          self._graph_stream.synchronize()
        
        # Get the result
        tokens = self._sample_output
      else:
        # Standard sampling path when graph is not available
        q = torch.empty(
          (logits.size(0), self.sharded_model.model.tok_embeddings.num_embeddings),
          device=logits.device
        ).exponential_(1, generator=self.rng)
        
        tokens = ttg.sample(logits, temperature=temp, top_k=top_k, q=q)
        
        # Store shape for future optimizations
        self._last_sample_shape = x.shape
      
      if DEBUG >= 4:
        print(f"tokens: {tokens}")

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
      hidden_state = torch.tensor(input_data).to(
        device=self.device,
        dtype=self.model_config["torch_dtype"]
      )
    elif input_data.ndim == 2:
      input_tensor = torch.tensor(input_data).to(
        device=self.device,
        dtype=torch.int
      )

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
        # Use torch.amp for mixed precision if available
        amp_context = contextlib.nullcontext()
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            # Determine the appropriate dtype based on hardware capabilities
            if self.force_fp16 and self.device.type == 'cuda':
                # Force FP16 for GPUs that don't support BF16 natively
                dtype = torch.float16
            else:
                # Use the model's dtype as default
                dtype = self.model_config.get("torch_dtype", torch.float32)
                
            # Create the autocast context with the appropriate dtype
            amp_context = torch.amp.autocast(device_type=self.device.type, dtype=dtype)
            
            if DEBUG >= 2:
                print(f"Using mixed precision with dtype: {dtype}")
        
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
        # Handle different dtypes properly
        if model_hs.dtype == torch.bfloat16 or (self.force_fp16 and model_hs.dtype != torch.float16):
          # Convert to appropriate dtype for numpy export
          if self.force_fp16:
            model_hs = model_hs.to(dtype=torch.float16)
          else:
            model_hs = model_hs.float()

        if DEBUG >= 4:
          print("sending hidden states")
          print(f"model_hs: {model_hs.size()}, dtype: {model_hs.dtype}")
          print(f"state.tokens: {self.state.tokens}")
          print(f"state.input_pos: {self.state.input_pos.size()}")
          print(f"state.mask: {self.state.mask.size()}")
        
        return (
          model_hs.numpy(force=True),
          self.state.to_dict(),
        )
      
      if self.state.curr_pos == 0:
        self.state.curr_pos = self.state.tokens.size(-1)
      else:
        self.state.curr_pos += 1

      # Handle different dtypes properly
      if model_logits.dtype == torch.bfloat16 or (self.force_fp16 and model_logits.dtype != torch.float16):
        # Convert to appropriate dtype for numpy export
        if self.force_fp16:
          model_logits = model_logits.to(dtype=torch.float16)
        else:
          model_logits = model_logits.float()
      
      # Always ensure we have a proper dtype for numpy export
      return (
        model_logits[:, -1].numpy(force=True),
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
