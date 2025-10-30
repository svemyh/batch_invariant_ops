#!/bin/bash
set -e  # Exit on error

echo "=========================================="
echo "Batch Invariant Ops - GPU Setup Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Step 1: Check GPU
echo "Step 1: Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    print_success "GPU detected"
else
    print_error "nvidia-smi not found. Is NVIDIA driver installed?"
    exit 1
fi
echo ""

# Step 2: Check Python version
echo "Step 2: Checking Python version..."
PYTHON_CMD=""

# Try different Python commands
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v $cmd &> /dev/null; then
        VERSION=$($cmd --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo $VERSION | cut -d. -f1)
        MINOR=$(echo $VERSION | cut -d. -f2)

        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_CMD=$cmd
            print_success "Found $cmd (version $VERSION)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    print_error "Python 3.10+ not found. Please install Python 3.10 or higher."
    exit 1
fi
echo ""

# Step 3: Create virtual environment
echo "Step 3: Creating virtual environment..."
if [ -d "venv" ]; then
    print_info "Virtual environment already exists. Removing..."
    rm -rf venv
fi

$PYTHON_CMD -m venv venv
print_success "Virtual environment created"
echo ""

# Step 4: Activate virtual environment and install dependencies
echo "Step 4: Installing dependencies..."
source venv/bin/activate

print_info "Upgrading pip..."
pip install --upgrade pip --quiet

print_info "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
print_success "PyTorch installed"

print_info "Installing Triton..."
pip install triton --quiet
print_success "Triton installed"

print_info "Installing batch_invariant_ops package..."
pip install -e . --quiet
print_success "Package installed"
echo ""

# Step 5: Verify installation
echo "Step 5: Verifying installation..."

print_info "Checking PyTorch and CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    print_success "PyTorch with CUDA is working"
else
    print_error "PyTorch CUDA not available"
    exit 1
fi

print_info "Checking Triton..."
python -c "import triton; print(f'Triton: {triton.__version__}')"
print_success "Triton is working"

print_info "Checking batch_invariant_ops..."
python -c "from batch_invariant_ops import set_batch_invariant_mode; print('Import successful')"
print_success "batch_invariant_ops is working"
echo ""

# Step 6: Run tests
echo "Step 6: Running batch invariance test..."
echo ""
print_info "This will take 10-30 seconds..."
echo ""

python test_batch_invariance.py

echo ""
print_success "Setup complete! Tests passed successfully."
echo ""

# Print next steps
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "To activate the environment in a new shell:"
echo "  source venv/bin/activate"
echo ""
echo "To run the test again:"
echo "  python test_batch_invariance.py"
echo ""
echo "To run vLLM deterministic inference test (requires additional setup):"
echo "  pip install httpx vllm"
echo "  vllm serve Qwen/Qwen3-8B --enforce-eager  # In separate terminal"
echo "  python deterministic_vllm_inference.py"
echo ""
echo "See GPU_SETUP_GUIDE.md for detailed documentation."
echo ""
