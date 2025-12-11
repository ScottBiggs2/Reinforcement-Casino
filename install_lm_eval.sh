#!/bin/bash
# Installation script for lm-eval with Python 3.11 compatibility fixes

set -e

echo "Installing lm-eval with Python 3.11 compatibility fixes..."

# Step 1: Upgrade pip and setuptools
echo "Step 1: Upgrading pip and setuptools..."
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

# Step 3: Install other dependencies that might be needed
echo "Step 3: Installing build dependencies..."
pip install pybind11 numexpr

# Step 4: Try installing lm-eval
echo "Step 4: Installing lm-eval..."
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
