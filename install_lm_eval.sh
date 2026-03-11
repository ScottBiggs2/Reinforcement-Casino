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

# Step 3: Install all dependencies from the dedicated eval_requirements.txt
echo "Step 3: Installing evaluation dependencies from eval_requirements.txt..."
pip install -r eval_requirements.txt

# Post-install fixes if needed
echo "Step 4: Verifying installation..."
python -c "import lm_eval; print(f'lm-eval version: {lm_eval.__version__}')" || {
    echo "ERROR: Installation verification failed"
    exit 1
}

echo ""
echo "✓ lm-eval installation complete!"
