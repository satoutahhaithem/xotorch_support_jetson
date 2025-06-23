import asyncio
import time
import uuid
from xotorch.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine
from xotorch.inference.shard import Shard
from xotorch.helpers import AsyncCallbackSystem
from xotorch.download.new_shard_download import new_shard_downloader

async def main():
    # Create the shard downloader
    shard_downloader = new_shard_downloader()
    
    # Create the inference engine
    inference_engine = TorchDynamicShardInferenceEngine(shard_downloader)
    
    # Define the shard
    shard = Shard(model_id="llama-3.2-1b", start_layer=0, end_layer=15, n_layers=16)
    
    # Define the prompt
    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 23 Jun 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello how are you<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # Generate tokens
    request_id = str(uuid.uuid4())
    
    # Start timing
    start_time = time.time()
    
    # Process the prompt
    tokens = []
    token_callback = AsyncCallbackSystem()
    callback = token_callback.register("token_callback")
    
    def on_token(request_id, new_tokens, is_finished):
        if isinstance(new_tokens, list):
            tokens.extend(new_tokens)
            current_time = time.time()
            elapsed = current_time - start_time
            if elapsed > 0:
                print(f"Generated {len(tokens)} tokens in {elapsed:.2f} seconds ({len(tokens)/elapsed:.2f} tokens/sec)")
    
    # Register the observer
    callback.on_next(on_token)
    
    # Ensure the shard is loaded
    await inference_engine.ensure_shard(shard)
    
    # Process the prompt
    print("Starting token generation...")
    
    # First, process the prompt
    result, inference_state = await inference_engine.infer_prompt(request_id, shard, prompt, {})
    
    # Then, generate tokens one by one
    is_finished = False
    max_tokens = 100  # Generate up to 100 tokens
    
    while not is_finished and len(tokens) < max_tokens:
        # Sample a token
        token = await inference_engine.sample(result)
        
        # Convert to integer and add to tokens list
        token_value = int(token.item())
        tokens.append(token_value)
        
        # Trigger the callback
        token_callback.trigger_all(request_id, [token_value], is_finished)
        
        # Check if we're done
        if token_value == inference_engine.tokenizer.eos_token_id:
            is_finished = True
            break
        
        # Process the next token
        forward = token.reshape(1, -1)
        result, inference_state = await inference_engine.infer_tensor(request_id, shard, forward, inference_state)
    
    # Calculate final speed
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Final: Generated {len(tokens)} tokens in {elapsed:.2f} seconds ({len(tokens)/elapsed:.2f} tokens/sec)")

if __name__ == "__main__":
    asyncio.run(main())