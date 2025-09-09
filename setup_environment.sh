#!/bin/bash
# Setup script for Metformin OpenMM Simulation Environment

echo "🧪 Setting up Metformin OpenMM Simulation Environment"
echo "=================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment: metformin_sim"
conda create -n metformin_sim python=3.9 -y

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate metformin_sim

# Install core dependencies via conda (for better compatibility)
echo "📥 Installing core dependencies..."
conda install -c conda-forge openmm rdkit mdtraj numpy scipy matplotlib -y

# Install AmberTools (optional, for GAFF parameters)
echo "🔬 Installing AmberTools for GAFF parameter generation..."
conda install -c conda-forge ambertools -y

# Install Python packages via pip
echo "🐍 Installing Python packages..."
pip install openff-toolkit pubchempy requests

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "📋 To use the environment:"
echo "   conda activate metformin_sim"
echo "   python metformin_openmm_sim.py"
echo ""
echo "🔍 To test the force field generator:"
echo "   python molecular_ff_generator.py"
echo ""
echo "📚 Dependencies installed:"
echo "   - OpenMM (molecular dynamics)"
echo "   - RDKit (molecular manipulation)"
echo "   - OpenFF Toolkit (force field generation)"
echo "   - AmberTools (GAFF parameters)"
echo "   - MDTraj (trajectory analysis)"
echo ""
