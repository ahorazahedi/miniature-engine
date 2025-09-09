# GPU Configuration Guide for Metformin OpenMM Simulation

## 🚀 Quick Start

### Test Your GPU Setup
```bash
# Test all GPUs
python test_gpu_setup.py

# Test specific GPU through main script
python metformin_openmm_sim.py --test-gpu --gpu 1
```

### Run Simulation on Your Second GPU
```bash
# Use GPU 1 (second GPU) - default behavior
python metformin_openmm_sim.py

# Explicitly specify GPU 1
python metformin_openmm_sim.py --gpu 1

# Use GPU 0 (first GPU) instead
python metformin_openmm_sim.py --gpu 0
```

## 📋 Command Line Options

```bash
python metformin_openmm_sim.py [OPTIONS]

Options:
  --gpu INT              GPU device index (default: 1 for second GPU)
  --formulation STR      Formulation name identifier
  --no-ff-gen           Disable automatic force field generation
  --test-gpu            Test GPU configuration and exit
  --help                Show help message
```

## 🔧 GPU Configuration Details

### What's Been Modified

1. **GPU Selection**: The simulation now defaults to GPU 1 (your second GPU)
2. **CUDA Optimization**: Configured with mixed precision and performance optimizations
3. **Automatic Fallback**: Falls back to GPU 0, OpenCL, or CPU if GPU 1 is unavailable
4. **GPU Validation**: Tests GPU availability before starting simulation

### CUDA Properties Used

```python
properties = {
    'CudaDeviceIndex': '1',                    # Use second GPU
    'CudaPrecision': 'mixed',                  # Mixed precision for speed
    'CudaCompiler': '/usr/local/cuda/bin/nvcc', # CUDA compiler
    'CudaTempDir': '/tmp',                     # Temp directory
    'CudaUseBlockingSync': 'false',            # Non-blocking for performance
}
```

## 🧪 Testing Examples

### Check All GPUs
```bash
# Run comprehensive GPU test
python test_gpu_setup.py
```

### Test Specific GPU
```bash
# Test GPU 1 (your second GPU)
python metformin_openmm_sim.py --test-gpu --gpu 1

# Test GPU 0 (first GPU)
python metformin_openmm_sim.py --test-gpu --gpu 0
```

### Monitor GPU Usage During Simulation
```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi

# Or get detailed info
nvidia-smi -l 1
```

## 🎯 Performance Tips

### 1. GPU Memory Management
- The simulation will automatically use mixed precision to save memory
- Monitor GPU memory usage with `nvidia-smi`
- If you get out-of-memory errors, try GPU 0 or reduce system size

### 2. Concurrent GPU Usage
- GPU 0: Can be used for other tasks (display, other simulations)
- GPU 1: Dedicated to your metformin simulation
- This setup allows multitasking without interference

### 3. Optimization Settings
```bash
# For maximum performance (if GPU 1 has more memory)
python metformin_openmm_sim.py --gpu 1

# For stability (if GPU 1 is less powerful)
python metformin_openmm_sim.py --gpu 0
```

## 🔍 Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check OpenMM CUDA support
python -c "from openmm import Platform; print(Platform.getPlatformByName('CUDA'))"
```

### Simulation Falls Back to CPU
1. Check GPU availability: `python test_gpu_setup.py`
2. Verify CUDA installation
3. Check OpenMM installation: `conda list openmm`
4. Try different GPU: `--gpu 0`

### Out of Memory Errors
1. Try the other GPU: `--gpu 0`
2. Check GPU memory: `nvidia-smi`
3. Close other GPU applications
4. Reduce simulation system size (modify composition in code)

## 📊 Expected Output

When running successfully on GPU 1, you should see:
```
🔍 GPU System Information:
  GPU 0: [Your first GPU name]
  GPU 1: [Your second GPU name] <- SELECTED
  ✅ OpenMM CUDA platform available

🎯 Configuration:
   GPU Index: 1
   Formulation: metformin_500mg_test
   Auto Force Field Generation: True

✅ Successfully configured CUDA on GPU 1
Using platform: CUDA
Using GPU device: 1
```

## 🚨 Important Notes

1. **Default GPU**: The simulation now defaults to GPU 1 (second GPU)
2. **Automatic Fallback**: If GPU 1 fails, it will try GPU 0, then OpenCL, then CPU
3. **Memory Requirements**: Molecular dynamics simulations can use significant GPU memory
4. **Performance**: GPU 1 should provide substantial speedup over CPU (10-100x faster)

## 📞 Quick Commands Reference

```bash
# Basic run (uses GPU 1)
python metformin_openmm_sim.py

# Test setup
python test_gpu_setup.py

# Use different GPU
python metformin_openmm_sim.py --gpu 0

# Test specific GPU
python metformin_openmm_sim.py --test-gpu --gpu 1

# Monitor GPU usage
nvidia-smi -l 1
```
