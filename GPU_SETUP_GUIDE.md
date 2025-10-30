# GPU Setup Guide for Batch Invariant Ops (A100 80GB)

Simple instructions to clone, setup, and test batch invariance on a rented GPU A100 80GB.

---

## Prerequisites

- **GPU**: NVIDIA A100 80GB (or any CUDA-capable GPU)
- **OS**: Linux (Ubuntu 20.04/22.04 recommended)
- **Internet connection** for downloading dependencies

---

## Step 1: Verify GPU Availability

```bash
# Check if NVIDIA GPU is available
nvidia-smi

# You should see your A100 GPU listed
```

Expected output: GPU information showing A100 with 80GB memory.

---

## Step 2: Install Python 3.10+

```bash
# Check Python version (must be 3.10 or higher)
python3 --version

# If Python < 3.10, install Python 3.10
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# Verify installation
python3.10 --version
```

---

## Step 3: Clone the Repository

```bash
# Clone the repository
git clone <your-repo-url> batch_invariant_ops
cd batch_invariant_ops

# Verify you're in the correct directory
ls -la
# You should see: README.md, pyproject.toml, test_batch_invariance.py, etc.
```

---

## Step 4: Create Virtual Environment

```bash
# Create virtual environment with Python 3.10+
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (prompt should show (venv))
which python
# Should show: /path/to/batch_invariant_ops/venv/bin/python
```

---

## Step 5: Install PyTorch with CUDA Support

```bash
# Install PyTorch with CUDA 12.1 (adjust CUDA version if needed)
# For A100, CUDA 11.8+ or 12.x is recommended
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch installation with CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch: 2.x.x+cu121
CUDA Available: True
CUDA Version: 12.1
```

---

## Step 6: Install Triton and Project Dependencies

```bash
# Install Triton (GPU kernel library)
pip install triton

# Install the batch_invariant_ops package in editable mode
pip install -e .

# Verify installation
python -c "from batch_invariant_ops import set_batch_invariant_mode; print('Installation successful!')"
```

---

## Step 7: Run Basic Batch Invariance Test

```bash
# Run the main test script
python test_batch_invariance.py
```

### Expected Output

```
Standard PyTorch:
Batch Deterministic: False run-to-run max/min/diff <non-zero values> for torch.float32 in 10 iterations
Batch Deterministic: False run-to-run max/min/diff <non-zero values> for torch.bfloat16 in 10 iterations

Batch-Invariant Mode:
Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for torch.float32 in 10 iterations
Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for torch.bfloat16 in 10 iterations
```

✅ **Success Criteria**: Batch-Invariant Mode shows `Batch Deterministic: True` with `0.0/0.0/0.0` differences.

---

## Step 8: (Optional) Test vLLM Deterministic Inference

This step requires additional setup for vLLM integration testing.

### Install Additional Dependencies

```bash
# Install httpx for async HTTP requests
pip install httpx

# Install vLLM (requires PyTorch 2.9.0+ and specific PR #24583)
# Note: As of writing, this requires a custom vLLM build
pip install vllm
```

### Run vLLM Server

```bash
# In a separate terminal, start vLLM server
vllm serve Qwen/Qwen3-8B --enforce-eager

# This will download the model (first time) and start server on localhost:8000
```

### Run Deterministic Inference Test

```bash
# In the original terminal, run the test
python deterministic_vllm_inference.py
```

Expected output: `Total samples: 1000, Unique samples: 1` (deterministic behavior)

---

## Troubleshooting

### Issue: `CUDA out of memory`
**Solution**: The A100 80GB should have sufficient memory. If you encounter this, check:
```bash
nvidia-smi  # Check GPU memory usage
# Kill other processes using GPU if needed
```

### Issue: `No module named 'triton'`
**Solution**:
```bash
pip install --upgrade triton
```

### Issue: `torch.cuda.is_available()` returns `False`
**Solution**: Reinstall PyTorch with correct CUDA version:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: `ImportError: cannot import name 'set_batch_invariant_mode'`
**Solution**: Reinstall the package:
```bash
pip install -e . --force-reinstall --no-deps
```

### Issue: Python version < 3.10
**Solution**: The code uses `match/case` statements (Python 3.10+ feature). You must use Python 3.10 or higher.

---

## Quick Verification Commands

```bash
# 1. Check Python version
python --version  # Should be 3.10+

# 2. Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 3. Check Triton
python -c "import triton; print(f'Triton: {triton.__version__}')"

# 4. Check package installation
python -c "from batch_invariant_ops import set_batch_invariant_mode; print('Package OK')"

# 5. Check GPU
nvidia-smi
```

---

## Performance Notes for A100 80GB

- **Test execution time**: ~10-30 seconds for basic test
- **Memory usage**: ~8-12 GB VRAM for test_batch_invariance.py
- **Expected throughput**: Very high due to A100's compute capabilities
- **Triton kernels**: Optimized for NVIDIA GPUs, should show significant performance

---

## Summary of Commands (Quick Reference)

```bash
# Clone and setup
git clone <repo-url> batch_invariant_ops && cd batch_invariant_ops
python3.10 -m venv venv && source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton
pip install -e .

# Run basic test
python test_batch_invariance.py

# Expected result: Batch Deterministic: True in Batch-Invariant Mode
```

---

## What the Test Validates

The test verifies that batch size does not affect computation results:

1. **Standard PyTorch**: Computing with batch_size=1 vs batch_size=2048 gives different results (non-deterministic)
2. **Batch-Invariant Mode**: Computing with any batch size gives identical results (deterministic)

This is critical for:
- Reproducible ML research
- Deterministic model inference
- Debugging numerical stability issues
- Ensuring consistent outputs across different batch configurations

---

## Additional Resources

- Blog post: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- GitHub Issues: Report any problems with the test execution
- A100 Optimization Guide: NVIDIA's documentation for optimal GPU utilization

---

## License

MIT License - See LICENSE file for details

---

**Last Updated**: 2025-01-30
**Tested On**: Ubuntu 22.04, Python 3.10, PyTorch 2.5+, CUDA 12.1, A100 80GB
