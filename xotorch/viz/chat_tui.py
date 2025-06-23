import time
import asyncio
import uuid
import traceback

# DIRECT DEBUG OUTPUT - THIS SHOULD ALWAYS PRINT
print("\n\n")
print("*"*80)
print("CHAT TUI MODULE LOADED")
print("*"*80)
print("\n\n")

from xotorch.helpers import is_port_available, find_available_port
# Disable DEBUG flag completely
DEBUG = 0
from xotorch.topology.device_capabilities import UNKNOWN_DEVICE_CAPABILITIES
from xotorch.models import build_base_shard, get_repo
from xotorch.inference.tokenizers import resolve_tokenizer

async def run_chat_tui(args, api, node):
  # Try to start the API server with fallback to an available port if needed
  try:
    # First check if the port is available
    if not is_port_available(args.chatgpt_api_port):
      print(f"Warning: Port {args.chatgpt_api_port} is already in use.")
      new_port = find_available_port("0.0.0.0")
      print(f"Using alternative port: {new_port}")
      args.chatgpt_api_port = new_port
      
    # Start the API server as a non-blocking task
    api_task = asyncio.create_task(api.run(port=args.chatgpt_api_port))
  except Exception as e:
    print(f"Note: API server not started: {e}")
    print("Continuing in TUI-only mode (no web interface available)")
  
  # Set Llama 1B as the default model
  default_model = args.default_model or "llama-3.2-1b"
  
  print("\n╔══════════════════════════════════════════════════════════════╗")
  print("║             XOTORCH TERMINAL INTERFACE                       ║")
  print("╠══════════════════════════════════════════════════════════════╣")
  print("║ Type your prompt after the '>' prompt below                  ║")
  print("║ Commands: 'exit' to quit, 'model <name>' to switch models    ║")
  print("║ Supported models: llama-3.2-1b, llama-3.2-3b, llama-3-8b     ║")
  print("╚══════════════════════════════════════════════════════════════╝")
  print("")
  print("═══════════════════════════")
  print(f"Model: {default_model}")
  print("═══════════════════════════")
  
  
  current_model = default_model
  tokens_per_second = 0.0
  last_token_count = 0
  start_time = None
  
  while True:
    try:
      user_input = input("\n> ")
      
      if user_input.lower() == 'exit':
        print("Exiting...")
        break
      elif user_input.lower().startswith('model '):
        model_name = user_input[6:].strip()
        current_model = model_name
        print(f"Set model to: {current_model}")
        continue
      
      if not user_input.strip():
        continue
      
      # Reset token timing metrics
      last_token_count = 0
      start_time = time.time()
      
      tflops = node.topology.nodes.get(node.id, UNKNOWN_DEVICE_CAPABILITIES).flops.fp16
      print(f"\n▶ Processing prompt with model: {current_model}")
      print(f"▶ System performance: {tflops:.2f} TFLOPS")
      print(f"▶ Starting generation...\n")
      
      # Create a custom callback to track tokens per second
      request_id = str(uuid.uuid4())
      callback_id = f"tui-token-speed-{request_id}"
      callback = node.on_token.register(callback_id)
      
      # Force print the callback registration
      print(f"\n*** REGISTERED CALLBACK: {callback_id} ***\n", flush=True)
      
      # Define token speed tracking callback
      async def track_token_speed():
        nonlocal tokens_per_second, last_token_count
        tokens = []
        full_response = ""
        
        def on_token(_request_id, _tokens, _is_finished):
          nonlocal tokens_per_second, last_token_count, start_time, full_response
          
          # FORCE PRINT EVERYTHING
          print("\n" + "="*50)
          print(f"TOKEN CALLBACK RECEIVED:")
          print(f"Request ID: {_request_id}")
          print(f"Tokens: {_tokens}")
          print(f"Is Finished: {_is_finished}")
          print("="*50 + "\n")
          
          tokens.extend(_tokens)
          
          # Calculate tokens per second
          # current_time = time.time()
          # elapsed = current_time - start_time
          # if elapsed > 0:
          #   tokens_per_second = 5 / elapsed # since updating display every 5 tokens
          
          # Try to decode and display the latest tokens
          try:
            if _tokens:
              # Convert float tokens to integers if needed
              int_tokens = [int(t) for t in _tokens]
              
              tokenizer = node.inference_engine.tokenizer
              try:
                # Force print raw tokens for debugging
                print(f"\n[TOKENS: {int_tokens}]", flush=True)
                
                new_text = tokenizer.decode(int_tokens)
                
                # Force print decoded text
                print(f"\n[TEXT: '{new_text}']", flush=True)
                
                # Add to full response
                full_response += new_text
              except Exception as e:
                print(f"\nError decoding tokens: {e}")
                print(f"Token values: {int_tokens}")
              
              # Update the stats line occasionally but only in debug mode
              if DEBUG and len(tokens) % 10 == 0:  # Update every 10 tokens
                current_time = time.time()
                elapsed = current_time - start_time
                if elapsed > 0:
                  tokens_per_second = len(tokens) / elapsed
          except Exception as e:
            if DEBUG >= 2:
              print(f"\nError decoding tokens: {e}")
          
          # Force print when the callback is called
          print(f"\n*** ON_TOKEN CALLBACK CALLED: {_request_id}, {len(_tokens)} tokens, is_finished={_is_finished} ***\n", flush=True)
          
          # Always return False to keep the callback active until we're done
          if _is_finished:
            print(f"\n*** FINISHED GENERATING TOKENS ***\n", flush=True)
            return True
          return False
        
        try:
          await callback.wait(on_token, timeout=300)
          
          # Show completion
          print("\n\n✓ Generation complete")
          print("===================\n")

          current_time = time.time()
          elapsed = current_time - start_time
          if elapsed > 0:
            tokens_per_second = len(tokens) / elapsed # since updating display every 5 tokens
          
          # Display final stats
          tflops = node.topology.nodes.get(node.id, UNKNOWN_DEVICE_CAPABILITIES).flops.fp16
          print(f"Final stats: {len(tokens)} tokens | {tokens_per_second:.2f} tokens/sec | {tflops:.2f} TFLOPS\n")
          
          # Write the response to a file for debugging
          with open("/tmp/xotorch_response.txt", "w") as f:
            f.write(f"Response length: {len(full_response)} characters\n")
            f.write(f"Response: {full_response}\n")
            f.write(f"Character codes: {' '.join([str(ord(c)) for c in full_response[:100]])}\n")
            f.write(f"Tokens: {tokens}\n")
          
          # Display the full response with VERY CLEAR formatting
          print("\n\n")
          print("="*80)
          print("="*30 + " FULL AI RESPONSE " + "="*30)
          print("="*80)
          print(f"Response length: {len(full_response)} characters")
          print("RESPONSE TEXT:")
          print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
          print(full_response)
          print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
          
          # Print the response with character codes for debugging
          print("\nRESPONSE CHARACTER CODES:")
          print(" ".join([f"{ord(c):d}" for c in full_response[:100]]))
          print("="*80 + "\n")
          
          # Print the file location
          print(f"\nResponse written to /tmp/xotorch_response.txt for debugging\n")
        except Exception as e:
          print(f"\n\nError tracking token speed: {e}")
        finally:
          node.on_token.deregister(callback_id)
      
      # Start token speed tracking in background
      token_speed_task = asyncio.create_task(track_token_speed())
      
      try:
        # Process the prompt
        shard = build_base_shard(current_model, node.inference_engine.__class__.__name__)
        if not shard:
          print(f"Error: Unsupported model '{current_model}'")
          continue
          
        tokenizer = await resolve_tokenizer(get_repo(shard.model_id, node.inference_engine.__class__.__name__))
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": user_input}], tokenize=False, add_generation_prompt=True)
        
        # Add direct print statement to show we're about to process the prompt
        print("\n\nDIRECT DEBUG: About to process prompt with request_id:", request_id)
        
        # Process the prompt
        result = await node.process_prompt(shard, prompt, request_id=request_id)
        
        # Add direct print statement to show the result
        print("\n\nDIRECT DEBUG: Result from process_prompt:", result)
        
        # Wait for token speed tracking to complete
        await token_speed_task
        
      except Exception as e:
        print(f"\nError processing prompt: {str(e)}")
        traceback.print_exc()
    
    except KeyboardInterrupt:
      print("\nExiting...")
      break
    except Exception as e:
      print(f"Error: {str(e)}")
      traceback.print_exc()