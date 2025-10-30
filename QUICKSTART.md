# Quick Start Guide (A100 GPU)

**Get up and running in 3 minutes!**

---

## Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone <your-repo-url> batch_invariant_ops
cd batch_invariant_ops

# Run automated setup script
./setup_gpu.sh
```

That's it! The script will:
- ✓ Detect your GPU
- ✓ Check Python version (3.10+)
- ✓ Create virtual environment
- ✓ Install PyTorch with CUDA
- ✓ Install Triton
- ✓ Install batch_invariant_ops
- ✓ Run validation tests

---

## Option 2: Manual Setup (3 commands)

```bash
# 1. Setup environment
python3.10 -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
pip install triton && \
pip install -e .

# 3. Run test
python test_batch_invariance.py
```

---

## Expected Output

### ✅ Success:
```
Standard PyTorch:
Batch Deterministic: False ...

Batch-Invariant Mode:
Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 ...
```

### ❌ Problem:
If you see errors, check `GPU_SETUP_GUIDE.md` for troubleshooting.

---

## What This Proves

The test demonstrates that:

1. **Standard PyTorch**: Different batch sizes produce different numerical results (non-deterministic)
2. **Batch-Invariant Mode**: All batch sizes produce identical results (deterministic)

This validates the batch invariance implementation works correctly on your GPU.

---

## Next Steps

- **Full Documentation**: See `GPU_SETUP_GUIDE.md`
- **vLLM Testing**: See `GPU_SETUP_GUIDE.md` Step 8
- **Code Details**: See `README.md`

---

## System Requirements

- **GPU**: NVIDIA GPU with CUDA support (A100 recommended)
- **Python**: 3.10 or higher
- **Memory**: 8-12 GB VRAM for basic test
- **OS**: Linux (Ubuntu 20.04/22.04)

---

## Troubleshooting One-Liners

```bash
# Check GPU
nvidia-smi

# Check Python version
python3 --version  # Must be 3.10+

# Check CUDA in PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall if needed
pip install -e . --force-reinstall
```

---

**Ready to verify batch invariance? Run `./setup_gpu.sh` now!**
