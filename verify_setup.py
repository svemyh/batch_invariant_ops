#!/usr/bin/env python3
"""
Quick verification script to check if batch_invariant_ops is properly set up.
Run this after installation to verify everything is working correctly.
"""

import sys


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_check(name, status, details=""):
    """Print a check result."""
    symbol = "✓" if status else "✗"
    status_text = "PASS" if status else "FAIL"
    print(f"{symbol} {name}: {status_text}")
    if details:
        print(f"  → {details}")


def main():
    all_checks_passed = True

    print_header("Batch Invariant Ops - Setup Verification")

    # Check 1: Python version
    print("\n1. Python Version Check")
    python_version = sys.version_info
    python_ok = python_version.major == 3 and python_version.minor >= 10
    print_check(
        "Python Version",
        python_ok,
        f"Python {python_version.major}.{python_version.minor}.{python_version.micro}",
    )
    all_checks_passed &= python_ok

    # Check 2: PyTorch
    print("\n2. PyTorch Check")
    try:
        import torch

        torch_ok = True
        torch_version = torch.__version__
        print_check("PyTorch Import", torch_ok, f"Version {torch_version}")
    except ImportError as e:
        torch_ok = False
        print_check("PyTorch Import", torch_ok, str(e))
        all_checks_passed = False
        print("\n⚠ PyTorch not installed. Cannot continue further checks.")
        sys.exit(1)

    # Check 3: CUDA
    print("\n3. CUDA Check")
    cuda_available = torch.cuda.is_available()
    print_check("CUDA Available", cuda_available)
    if cuda_available:
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  → CUDA Version: {cuda_version}")
        print(f"  → GPU: {gpu_name}")
        print(f"  → Memory: {gpu_memory:.1f} GB")
    else:
        print("  → No CUDA GPU detected")
    all_checks_passed &= cuda_available

    # Check 4: Triton
    print("\n4. Triton Check")
    try:
        import triton

        triton_ok = True
        triton_version = triton.__version__
        print_check("Triton Import", triton_ok, f"Version {triton_version}")
    except ImportError as e:
        triton_ok = False
        print_check("Triton Import", triton_ok, str(e))
        all_checks_passed = False

    # Check 5: batch_invariant_ops
    print("\n5. Package Check")
    try:
        from batch_invariant_ops import (
            set_batch_invariant_mode,
            is_batch_invariant_mode_enabled,
            matmul_persistent,
        )

        package_ok = True
        print_check("batch_invariant_ops Import", package_ok)
    except ImportError as e:
        package_ok = False
        print_check("batch_invariant_ops Import", package_ok, str(e))
        all_checks_passed = False

    # Check 6: Quick functional test
    if all_checks_passed:
        print("\n6. Functional Test")
        try:
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Create small test tensors
            a = torch.randn(16, 32, device=device, dtype=torch.float32)
            b = torch.randn(32, 16, device=device, dtype=torch.float32)

            # Test batch invariant mode
            with set_batch_invariant_mode(True):
                result = torch.mm(a, b)

            functional_ok = result.shape == (16, 16)
            print_check("Batch Invariant Operation", functional_ok, f"Output shape: {result.shape}")

            # Test if mode enables/disables correctly
            mode_before = is_batch_invariant_mode_enabled()
            with set_batch_invariant_mode(True):
                mode_during = is_batch_invariant_mode_enabled()
            mode_after = is_batch_invariant_mode_enabled()

            mode_ok = not mode_before and mode_during and not mode_after
            print_check("Mode Toggle", mode_ok, "Context manager working correctly")

            all_checks_passed &= functional_ok and mode_ok

        except Exception as e:
            print_check("Functional Test", False, str(e))
            all_checks_passed = False

    # Final summary
    print_header("Summary")
    if all_checks_passed:
        print("\n✓ All checks passed! Setup is complete and working.")
        print("\nYou can now run:")
        print("  python test_batch_invariance.py")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the errors above.")
        print("\nCommon fixes:")
        print("  • Install PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("  • Install Triton: pip install triton")
        print("  • Install package: pip install -e .")
        print("\nSee GPU_SETUP_GUIDE.md for detailed troubleshooting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
