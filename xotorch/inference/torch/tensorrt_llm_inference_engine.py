"""
TensorRTLLMInferenceEngine
Optimized inference engine using NVIDIA's TensorRT-LLM for faster inference on NVIDIA GPUs
"""

import os
import functools
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uuid
import re
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import torch

from xotorch.inference.inference_engine import InferenceEngine
from xotorch.download.shard_download import ShardDownloader
from xotorch.inference.shard import Shard
from xotorch.inference.tokenizers import _resolve_tokenizer
from xotorch.helpers import DEBUG
from xotorch.inference.torch.models.llm_utils import (
  load_model_config,
  ShardInferenceState
)

# Check if tensorrt_llm is available
try:
    import tensorrt_llm
    import tensorrt_llm.runtime
    TENSORRT_LLM_AVAILABLE = True
except ImportError:
    TENSORRT_LLM_AVAILABLE = False
    print("TensorRT-LLM not found. Please install it with: pip install tensorrt-llm")

# Default sampling parameters
TEMP = 0.6
TOP_K = 35

class TensorRTLLMInferenceEngine(InferenceEngine):
    """
    TensorRT-LLM based inference engine for optimized performance on NVIDIA GPUs
    """
    def __init__(self, shard_downloader: ShardDownloader):
        if not TENSORRT_LLM_AVAILABLE:
            raise ImportError("TensorRT-LLM is required but not installed. Please install it with: pip install tensorrt-llm")
        
        self.shard = None
        self.shard_downloader = shard_downloader
        self.model = None
        self.engine = None
        self.request_id = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.uuid = str(uuid.uuid4())
        self.model_path = None
        self.model_config = None
        self.state = None
        self.tokenizer = None
        
        # device settings - TensorRT-LLM requires CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT-LLM requires CUDA-capable GPU")
        
        self.device = torch.device("cuda")
        
        # rng setup for sampling
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(1234)
    
    def clear_model(self):
        """
        Clear out model and shard
        A way to avoid OOM issues
        """
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.engine is not None:
            del self.engine
            self.engine = None
        
        torch.cuda.empty_cache()
        
        self.shard = None
        self.state = None
    
    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """
        Encode the prompt into tokens
        """
        if DEBUG >= 4:
            print("encode called")
            print(f"shard: {shard}")
            print(f"prompt: {prompt}")
        
        await self.ensure_shard(shard)
        
        def encode_wrapper() -> np.ndarray:
            """
            Encode the prompt into tokens
            """
            tokens = self.tokenizer.encode(
                prompt,
                return_tensors="pt"
            )
            
            # move to proper device
            if tokens.device != self.device:
                tokens = tokens.to(device=self.device)
            
            if DEBUG >= 4:
                print("encoded_wrapper called")
                print(f"tokens: {tokens}")
            
            # Reset state
            self.state = ShardInferenceState(device=self.device)
            self.state.curr_pos = 0
            self.state.tokens = tokens
            
            return tokens.cpu().numpy()
        
        return await asyncio.get_running_loop().run_in_executor(
            self.executor,
            functools.partial(encode_wrapper),
        )
    
    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """
        Decode tokens back to text
        """
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
        """
        Sample from logits
        """
        if DEBUG >= 4:
            print("sample called")
            print(f"x: {x}")
            print(f"temp: {temp}")
            print(f"top_k: {top_k}")
        
        logits = torch.tensor(x).to(self.device)
        
        def sample_wrapper():
            # TensorRT-LLM sampling
            sampling_config = tensorrt_llm.runtime.SamplingConfig(
                temperature=temp,
                top_k=top_k,
                random_seed=1234
            )
            
            # Convert logits to probabilities and sample
            probs = torch.softmax(logits / temp, dim=-1)
            
            # Apply top-k
            if top_k > 0:
                values, indices = torch.topk(probs, top_k)
                mask = torch.zeros_like(probs).scatter_(-1, indices, 1)
                probs = probs * mask
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sample from the distribution
            tokens = torch.multinomial(probs, num_samples=1, generator=self.rng)
            
            if DEBUG >= 4:
                print(f"tokens: {tokens}")
            
            return tokens.cpu().numpy()
        
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, 
            functools.partial(sample_wrapper)
        )
    
    async def infer_tensor(
        self,
        request_id: str,
        shard: Shard,
        input_data: np.ndarray,
        inference_state: Optional[dict] = None
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Run inference using TensorRT-LLM
        """
        await self.ensure_shard(shard)
        
        if DEBUG >= 4:
            print("infer_tensor called")
            print(f"shard: {shard}")
            print(f"input_data: {input_data}")
            print(f"state {self.state}")
        
        if inference_state and inference_state.get("tokens") is not None:
            self.state.from_dict(inference_state)
        
        self.request_id = request_id if not self.request_id else self.request_id
        
        def infer_wrapper():
            if DEBUG >= 4:
                print(f"infer_wrapper called")
                print(f"self.state.tokens: {self.state.tokens}")
            
            # Convert input data to the format expected by TensorRT-LLM
            if input_data.ndim == 2:
                input_tensor = torch.tensor(input_data).to(
                    device=self.device,
                    dtype=torch.int32
                )
                
                if self.state.tokens is not None:
                    if input_data.shape[-1] == 1:
                        self.state.tokens = torch.cat([
                            self.state.tokens.to(self.device),
                            input_tensor.clone()
                        ], dim=-1).to(self.device)
                else:
                    self.state.tokens = input_tensor.clone()
            
            try:
                # Run inference with TensorRT-LLM
                if self.state.curr_pos == 0:
                    # First inference pass - process the entire input
                    input_lengths = torch.tensor([self.state.tokens.shape[1]], dtype=torch.int32, device=self.device)
                    
                    # Run TensorRT-LLM inference
                    output = self.engine.run(
                        self.state.tokens,
                        input_lengths,
                        streaming=True
                    )
                    
                    # Get the output logits for the next token prediction
                    logits = output.logits
                    
                    # Update position counter
                    self.state.curr_pos = self.state.tokens.shape[1]
                    
                    return (
                        logits[:, -1].float().cpu().numpy(),
                        self.state.to_dict(),
                    )
                else:
                    # Subsequent inference - process just the new token
                    new_token = self.state.tokens[:, -1:]
                    
                    # Run TensorRT-LLM inference for the next token
                    output = self.engine.run_next(new_token)
                    
                    # Get the output logits
                    logits = output.logits
                    
                    # Update position counter
                    self.state.curr_pos += 1
                    
                    return (
                        logits[:, -1].float().cpu().numpy(),
                        self.state.to_dict(),
                    )
                
            except Exception as err:
                print(f"TensorRT-LLM inference error: {err}")
                raise
        
        return await asyncio.get_running_loop().run_in_executor(self.executor, infer_wrapper)
    
    async def ensure_shard(self, shard: Shard):
        """
        Ensure the model shard is loaded
        """
        if DEBUG >= 4:
            print("shard ensured\n")
            print(f"shard: {shard}")
            print(f"class shard: {self.shard}")
            print(f"uuid: {self.uuid}")
        
        # If the shard is already loaded, return
        if self.shard == shard:
            return
        
        self.shard = shard
        
        # Using CPU to store inference state
        self.state = ShardInferenceState()
        
        # Download model safetensors and shard
        self.model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
        self.model_config = load_model_config(self.model_path/"config.json")
        
        # Load tokenizer
        self.tokenizer = await _resolve_tokenizer(self.model_path)
        
        def start_model():
            if DEBUG >= 4:
                print("start_model called")
            
            # Create TensorRT-LLM engine directory if it doesn't exist
            trt_engine_path = self.model_path / "tensorrt_llm_engines"
            trt_engine_path.mkdir(exist_ok=True)
            
            engine_name = f"{shard.model_id.replace('/', '_')}_shard_{shard.start_layer}_{shard.end_layer}.engine"
            engine_path = trt_engine_path / engine_name
            
            # Check if engine already exists
            if engine_path.exists():
                if DEBUG >= 2:
                    print(f"Loading existing TensorRT-LLM engine from {engine_path}")
                
                # Load the engine
                self.engine = tensorrt_llm.runtime.GenerationSession.load(str(engine_path))
            else:
                if DEBUG >= 2:
                    print(f"Building new TensorRT-LLM engine for {shard.model_id}")
                
                # Convert model to TensorRT-LLM format
                # This is a simplified version - actual implementation would need to handle
                # different model architectures and configurations
                
                # Determine model architecture
                if "llama" in shard.model_id.lower():
                    model_type = "llama"
                elif "mistral" in shard.model_id.lower():
                    model_type = "mistral"
                elif "qwen" in shard.model_id.lower():
                    model_type = "qwen"
                else:
                    model_type = "auto"
                
                # Build TensorRT-LLM engine
                builder = tensorrt_llm.Builder()
                builder_config = builder.create_builder_config(
                    precision="float16",  # Use FP16 for better performance
                    max_batch_size=1,
                    max_input_len=self.model_config["max_seq_len"],
                    max_output_len=1024,
                )
                
                # Create TensorRT-LLM model
                self.model = tensorrt_llm.models.LLMForCausalLM.from_hugging_face(
                    model_dir=str(self.model_path),
                    model_type=model_type,
                    dtype="float16",
                )
                
                # Build and save engine
                engine_buffer = builder.build_engine(self.model, builder_config)
                with open(str(engine_path), "wb") as f:
                    f.write(engine_buffer)
                
                # Load the engine
                self.engine = tensorrt_llm.runtime.GenerationSession.load(str(engine_path))
            
            # Configure the engine
            self.engine.setup(
                batch_size=1,
                max_sequence_length=self.model_config["max_seq_len"],
            )
        
        await asyncio.get_running_loop().run_in_executor(
            self.executor,
            functools.partial(start_model),
        )
    
    async def load_checkpoint(self, shard: Shard, path: str):
        """
        Load a checkpoint
        """
        await self.ensure_shard(shard)