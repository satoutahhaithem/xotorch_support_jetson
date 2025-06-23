#!/bin/bash

# Script to optimize NVIDIA GPUs for better inference performance with TensorRT-LLM and XOTorch

echo "Optimizing NVIDIA GPU for TensorRT-LLM inference with XOTorch..."

# Check if running on a system with NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "This script is intended for systems with NVIDIA GPUs only."
    exit 1
fi

# Check if TensorRT-LLM is installed
if ! pip list | grep -q tensorrt-llm; then
    echo "TensorRT-LLM is not installed. Installing it now..."
    pip install tensorrt-llm
fi

# Set environment variables for XOTorch with TensorRT-LLM
echo "Setting environment variables..."
export TORCH_DTYPE="float16"
export TORCH_USE_CACHE="true"
export TOKENIZERS_PARALLELISM="true"
export TENSORRT_LLM_VERBOSE="0"  # Set to higher values for more verbose output

# Configure CUDA for better performance
echo "Configuring CUDA settings..."

# Get GPU information
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "Detected GPU: $GPU_INFO"

# Get total GPU memory in MB
TOTAL_GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
echo "Total GPU memory: ${TOTAL_GPU_MEM}MB"

# Optimize memory allocation
echo "Optimizing memory settings..."

# Reserve 90% of GPU memory for TensorRT-LLM
TENSORRT_MEM=$((TOTAL_GPU_MEM * 90 / 100))
echo "Reserving ${TENSORRT_MEM}MB for TensorRT-LLM operations"

# Set CUDA memory pool settings
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"

# Set TensorRT-LLM specific optimizations
export TENSORRT_LLM_BUILDER_OPTIMIZATION_LEVEL=3  # Maximum optimization level
export TENSORRT_LLM_ENABLE_PROFILING=0  # Disable profiling for better performance
export TENSORRT_LLM_BUILDER_WORKSPACE_SIZE=$((TENSORRT_MEM * 1024 * 1024))  # Set workspace size in bytes

# Create a .xotorch_tensorrt_config file with optimized settings
echo "Creating optimized TensorRT-LLM configuration..."
cat > ~/.xotorch_tensorrt_config << EOF
# XOTorch optimized configuration for TensorRT-LLM
TORCH_DTYPE=float16
TORCH_USE_CACHE=true
TOKENIZERS_PARALLELISM=true
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
TENSORRT_LLM_VERBOSE=0
TENSORRT_LLM_BUILDER_OPTIMIZATION_LEVEL=3
TENSORRT_LLM_ENABLE_PROFILING=0
TENSORRT_LLM_BUILDER_WORKSPACE_SIZE=${TENSORRT_MEM}
EOF

echo "Adding configuration to .bashrc for automatic loading..."
if ! grep -q "source ~/.xotorch_tensorrt_config" ~/.bashrc; then
    echo "# Load XOTorch optimized TensorRT-LLM configuration" >> ~/.bashrc
    echo "if [ -f ~/.xotorch_tensorrt_config ]; then" >> ~/.bashrc
    echo "    source ~/.xotorch_tensorrt_config" >> ~/.bashrc
    echo "fi" >> ~/.bashrc
fi

# Set NVIDIA GPU to maximum performance mode if possible
if command -v nvidia-smi &> /dev/null; then
    echo "Setting GPU to maximum performance mode..."
    sudo nvidia-smi -pm 1  # Set persistent mode
    sudo nvidia-smi -ac $(nvidia-smi -q -d SUPPORTED_CLOCKS | grep -A 1 "Memory" | tail -n 1 | awk '{print $1}'),$(nvidia-smi -q -d SUPPORTED_CLOCKS | grep -A 1 "Graphics" | tail -n 1 | awk '{print $1}')
fi

echo "Optimization complete! Please restart your terminal or run 'source ~/.bashrc' to apply changes."
echo "Run XOTorch with: xot --inference-engine tensorrt <model_name>"