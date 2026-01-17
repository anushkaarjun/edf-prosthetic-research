#!/bin/bash
# Script to organize project files into folders

echo "Organizing project files..."

# Create directories
mkdir -p scripts models docs notebooks results

# Move training scripts
echo "Moving training scripts..."
mv train_*.py scripts/ 2>/dev/null || true
mv run_*.py scripts/ 2>/dev/null || true
mv eeg_api_server.py scripts/ 2>/dev/null || true
mv load_*.py scripts/ 2>/dev/null || true
mv load_*.sh scripts/ 2>/dev/null || true
mv compare_*.py scripts/ 2>/dev/null || true
mv evaluate_*.py scripts/ 2>/dev/null || true

# Move model files
echo "Moving model files..."
mv *.pth models/ 2>/dev/null || true
mv *.pkl models/ 2>/dev/null || true

# Move documentation
echo "Moving documentation..."
mv *.md docs/ 2>/dev/null || true
# Keep README.md in root
mv docs/README.md . 2>/dev/null || true

# Move notebooks
echo "Moving notebooks..."
mv *.ipynb notebooks/ 2>/dev/null || true

# Move results
echo "Moving results..."
mv *.txt results/ 2>/dev/null || true

echo "File organization complete!"
echo ""
echo "Current structure:"
echo "  scripts/ - Training and API scripts"
echo "  models/ - Trained model files (.pth, .pkl)"
echo "  docs/ - Documentation files (.md)"
echo "  notebooks/ - Jupyter notebooks"
echo "  results/ - Training results and outputs"
