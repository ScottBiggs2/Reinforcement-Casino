#!/bin/bash
# Installation script for lm-eval with Python 3.11 compatibility fixes

set -e

echo "Installing lm-eval with Python 3.11 compatibility fixes..."

# Step 1: Upgrade pip and setuptools and clear cache
echo "Step 1: Upgrading pip and clearing cache..."
pip cache purge || true
pip install --upgrade pip setuptools wheel

# Step 2: Install rouge-score separately (often causes build issues)
echo "Step 2: Installing rouge-score separately..."
pip install rouge-score || {
    echo "Warning: rouge-score installation failed, trying alternative..."
    pip install rouge-score --no-build-isolation || {
        echo "Trying to install from source..."
        pip install git+https://github.com/google-research/google-research.git#subdirectory=rouge || true
    }
}

# Step 3: Install other dependencies and resolve conflicts
echo "Step 3: Forcefully removing incompatible versions and fixing conflicts..."

# Force remove versions that might have been partially installed/cached by previous attempts
# This is critical to prevent "vllm 0.17.0" and "torchvision 0.25.0" ghosts from crashing the env
echo "Cleaning up 'ghost' packages (vllm, torchvision)..."
pip uninstall -y vllm torchvision 2>/dev/null || true

# Re-install basic build dependencies
pip install pybind11 numexpr langdetect

# Step 4: Fix evaluation environment conflicts
echo "Step 4: Fixing version conflicts for evaluation environment..."
# 1. Torchvision must match torch 2.9.0
pip install torchvision==0.24.0 --no-deps 
# 2. Fix protobuf for vLLM 0.11.1 compatibility
pip install protobuf==5.29.6
# 3. Handle numpy/numba constraint for this environment
pip install "numpy<2.3"

# Step 5: Install Eval Harness Core
echo "Step 5: Installing evaluation harness core (moved from requirements.txt)..."
pip install evaluate rouge-score

# Step 6: Try installing vLLM for 5-10x speedup
echo "Step 6: Installing vLLM (pinned to 0.11.1 for torch 2.9.0)..."
pip install vllm==0.11.1 --no-build-isolation || echo "Warning: vLLM installation failed, using Transformers fallback."

# Step 7: Try installing lm-eval
echo "Step 7: Installing lm-eval..."
if ! pip install lm-eval; then
    echo "Standard installation failed, trying from source..."
    echo "Cloning lm-evaluation-harness repository..."
    cd /tmp
    rm -rf lm-evaluation-harness
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git
    cd lm-evaluation-harness
    pip install -e .
    cd -
    echo "✓ Installed from source"
else
    echo "✓ Installed from PyPI"
fi

echo ""
echo "Verifying installation..."
python -c "import lm_eval; print(f'lm-eval version: {lm_eval.__version__}')" || {
    echo "ERROR: Installation verification failed"
    exit 1
}

echo ""
echo "✓ lm-eval installation complete!"
