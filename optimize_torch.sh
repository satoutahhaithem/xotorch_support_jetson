#!/bin/bash

# Script to optimize PyTorch for better inference performance with XOTorch

echo "Optimizing PyTorch for XOTorch inference..."

# Check if running on a system with NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Optimizing for CUDA..."
    
    # Get GPU information
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    echo "Detected GPU: $GPU_INFO"
    
    # Get total GPU memory in MB
    TOTAL_GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    echo "Total GPU memory: ${TOTAL_GPU_MEM}MB"
    
    # Reserve 90% of GPU memory for PyTorch
    PYTORCH_MEM=$((TOTAL_GPU_MEM * 90 / 100))
    echo "Reserving ${PYTORCH_MEM}MB for PyTorch operations"
    
    # Set CUDA memory pool settings
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"
    
    # Set NVIDIA GPU to maximum performance mode if possible
    if command -v nvidia-smi &> /dev/null; then
        echo "Setting GPU to maximum performance mode..."
        sudo nvidia-smi -pm 1  # Set persistent mode
        
        # Get the highest supported memory and graphics clocks
        MEM_CLOCK=$(nvidia-smi -q -d SUPPORTED_CLOCKS | grep -A 1 "Memory" | tail -n 1 | awk '{print $1}' | tr -d ',')
        GPU_CLOCK=$(nvidia-smi -q -d SUPPORTED_CLOCKS | grep -A 1 "Graphics" | tail -n 1 | awk '{print $1}' | tr -d ',')
        
        if [ ! -z "$MEM_CLOCK" ] && [ ! -z "$GPU_CLOCK" ]; then
            echo "Setting application clocks to Memory: $MEM_CLOCK MHz, Graphics: $GPU_CLOCK MHz"
            sudo nvidia-smi -ac $MEM_CLOCK,$GPU_CLOCK
        else
            echo "Could not determine supported clock speeds. Skipping clock configuration."
        fi
    fi
elif [ "$(uname)" == "Darwin" ] && [ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple"* ]; then
    echo "Apple Silicon detected. Optimizing for MPS..."
    
    # Get the total memory in MB
    TOTAL_MEM_MB=$(($(sysctl -n hw.memsize) / 1024 / 1024))
    
    # Calculate 80% of total memory
    MPS_MEM=$((TOTAL_MEM_MB * 80 / 100))
    echo "Reserving ${MPS_MEM}MB for MPS operations"
    
    # Set environment variables for MPS
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Disable high watermark
    export PYTORCH_MPS_ALLOCATOR_MEMORYLESS=1  # Enable memoryless mode
else
    echo "No GPU detected. Optimizing for CPU..."
    
    # Get number of CPU cores
    NUM_CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    echo "Detected $NUM_CORES CPU cores"
    
    # Set number of threads for PyTorch
    export OMP_NUM_THREADS=$NUM_CORES
    export MKL_NUM_THREADS=$NUM_CORES
fi

# Create a .xotorch_torch_config file with optimized settings
echo "Creating optimized PyTorch configuration..."
cat > ~/.xotorch_torch_config << EOF
# XOTorch optimized configuration for PyTorch
TORCH_USE_CACHE=true
TOKENIZERS_PARALLELISM=true
# Use max-autotune for better runtime performance
TORCH_COMPILE_MODE=max-autotune
TORCH_USE_COMPILE=true
TORCH_USE_FLASH_ATTENTION=true
# Increase batch size for better GPU utilization
TORCH_BATCH_SIZE=4
# Increase prefetch size for better pipelining
TORCH_PREFETCH_SIZE=8
# Optimize memory allocation
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.6
# Force FP16 precision since BF16 is not supported on Quadro RTX 6000
TORCH_FORCE_FP16=true
# Optimize thread pool for GPU operations
TORCH_THREAD_POOL_SIZE=16
# Enable CUDA kernel fusion
TORCH_CUDNN_BENCHMARK=true
EOF

echo "Adding configuration to .bashrc for automatic loading..."
if ! grep -q "source ~/.xotorch_torch_config" ~/.bashrc; then
    echo "# Load XOTorch optimized PyTorch configuration" >> ~/.bashrc
    echo "if [ -f ~/.xotorch_torch_config ]; then" >> ~/.bashrc
    echo "    source ~/.xotorch_torch_config" >> ~/.bashrc
    echo "fi" >> ~/.bashrc
fi

echo "Optimization complete! Please restart your terminal or run 'source ~/.bashrc' to apply changes."
echo "Run XOTorch with: xot --inference-engine torch <model_name>"