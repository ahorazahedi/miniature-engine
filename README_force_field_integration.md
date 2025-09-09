# Automatic Force Field Generation for Metformin Simulation

This system automatically generates force field parameters for pharmaceutical molecules using PubChem data and OpenFF/GAFF parameterization, eliminating the "No template found" errors in OpenMM.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Run the automated setup script
./setup_environment.sh

# Or manually:
conda create -n metformin_sim python=3.9
conda activate metformin_sim
conda install -c conda-forge openmm rdkit mdtraj ambertools
pip install openff-toolkit pubchempy requests
```

### 2. Test the System
```bash
# Test force field generation
python test_force_field_generation.py

# If tests pass, run the full simulation
python metformin_openmm_sim.py
```

## 📋 What's New

### Automatic Force Field Generation
- **Fetches 3D structures** from PubChem automatically
- **Generates OpenFF parameters** for drug molecules
- **Creates OpenMM-compatible XML** force field files
- **Eliminates "UNK residue" errors** by proper parameterization

### Integration with Existing Simulation
- **Seamless integration** with your existing `metformin_openmm_sim.py`
- **Automatic fallback** to built-in force fields if generation fails
- **Proper molecular topology** creation with real 3D coordinates

## 🔧 How It Works

### 1. Molecular Data Fetching (`molecular_ff_generator.py`)
```python
# Automatically fetches from PubChem
mol_data = ff_generator.fetch_from_pubchem("metformin")
# Returns: SMILES, 3D SDF, molecular properties
```

### 2. 3D Structure Processing
```python
# Uses RDKit to process and optimize 3D structures
rdkit_mol = ff_generator.process_with_rdkit(mol_data)
# Generates proper 3D conformers with hydrogens
```

### 3. Force Field Parameter Generation
```python
# Uses OpenFF Toolkit for modern force field parameters
system, xml_content = ff_generator.generate_openff_parameters(mol_data, rdkit_mol)
# Creates OpenMM-compatible XML with proper atom types, bonds, angles, torsions
```

### 4. Integration with OpenMM Simulation
```python
# Automatically loads generated force fields
sim = MetforminTabletSimulation("metformin_500mg", auto_generate_ff=True)
sim.run_complete_simulation()
# No more "No template found" errors!
```

## 📁 File Structure

```
MDD/
├── metformin_openmm_sim.py          # Main simulation (updated)
├── molecular_ff_generator.py         # Force field generator
├── test_force_field_generation.py   # Test suite
├── setup_environment.sh             # Environment setup
├── requirements.txt                  # Python dependencies
└── metformin_sim_*/                  # Simulation outputs
    ├── force_fields/                 # Generated force field files
    │   ├── metformin_openff.xml     # OpenFF parameters
    │   ├── metformin_data.json      # Molecular data
    │   └── metformin.sdf            # 3D structure
    ├── trajectories/                 # MD trajectories
    ├── logs/                        # Simulation logs
    └── analysis/                    # Analysis results
```

## 🧪 Supported Molecules

### Currently Implemented
- **Metformin** - Main drug molecule (fully supported)
- **Small drug molecules** - Any molecule in PubChem

### Future Support
- **Microcrystalline Cellulose (MCC)** - Polymer excipient
- **Povidone (PVP)** - Binder polymer
- **Croscarmellose Sodium** - Disintegrant
- **Magnesium Stearate** - Lubricant

### Adding New Molecules
```python
# Simply add to the molecules list in generate_molecular_force_fields()
molecules_to_process = [
    "metformin",
    "your_molecule_name",  # Must be in PubChem
]
```

## ⚙️ Configuration Options

### Force Field Methods
```python
# OpenFF (recommended for drug molecules)
result = ff_generator.process_molecule("metformin", method='openff')

# GAFF (requires AmberTools)
result = ff_generator.process_molecule("metformin", method='gaff')
```

### Simulation Options
```python
# Enable automatic force field generation (default)
sim = MetforminTabletSimulation("test", auto_generate_ff=True)

# Disable automatic generation (use built-in force fields)
sim = MetforminTabletSimulation("test", auto_generate_ff=False)
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "No template found for residue UNK"
**Solution**: This is exactly what our system fixes! Enable auto force field generation:
```python
sim = MetforminTabletSimulation("test", auto_generate_ff=True)
```

#### 2. Missing Dependencies
**Solution**: Run the setup script or install manually:
```bash
./setup_environment.sh
# Or check requirements.txt for manual installation
```

#### 3. PubChem Connection Issues
**Solution**: Check internet connection or use local SDF files:
```python
# The system will fall back to API calls if PubChemPy fails
```

#### 4. OpenFF Parameter Generation Fails
**Solution**: System automatically falls back to GAFF or built-in force fields:
```
2025-01-07 - WARNING - Failed to load generated force fields: ...
2025-01-07 - INFO - Falling back to built-in force fields
```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Comparison

### Before (Built-in Force Fields)
```
❌ No template found for residue 0 (UNK)
❌ Force field system creation failed
❌ Falling back to simplified system
⚠️  Results for qualitative purposes only
```

### After (Auto-Generated Force Fields)
```
✅ Successfully fetched metformin (CID: 4091)
✅ Using OpenFF force field: openff-2.1.0.offxml
✅ Successfully generated OpenFF parameters for metformin
✅ Successfully loaded generated force fields
✅ Force field system created successfully
```

## 🔬 Scientific Accuracy

### Generated Parameters Include
- **Proper atom types** based on chemical environment
- **Accurate bond parameters** from OpenFF database
- **Realistic angle and torsion terms**
- **Validated partial charges** using established methods
- **Lennard-Jones parameters** for van der Waals interactions

### Validation
- Parameters are based on **OpenFF 2.x** (state-of-the-art force field)
- **Extensive benchmarking** against experimental data
- **Quantum mechanical validation** of parameters
- **Compatible with pharmaceutical research** standards

## 📈 Future Enhancements

### Planned Features
1. **Polymer support** for MCC, PVP, and other excipients
2. **Custom parameter databases** for proprietary molecules
3. **Automated validation** against experimental data
4. **Parameter optimization** for specific formulations
5. **Multi-molecule force field** combination and validation

### Contributing
To add support for new molecules or improve parameterization:
1. Extend `molecular_ff_generator.py`
2. Add molecule-specific processing in `process_molecule()`
3. Update tests in `test_force_field_generation.py`

## 📚 References

- **OpenFF Toolkit**: https://github.com/openforcefield/openff-toolkit
- **PubChem API**: https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest
- **OpenMM Documentation**: http://docs.openmm.org/
- **RDKit**: https://www.rdkit.org/
- **GAFF Force Field**: https://ambermd.org/antechamber/gaff.html

## 🎯 Summary

This system transforms your metformin simulation from a simplified demonstration to a **scientifically accurate molecular dynamics simulation** with proper force field parameters. No more "UNK residue" errors - just real pharmaceutical science!

**Key Benefits:**
- ✅ **Eliminates force field errors**
- ✅ **Automatic parameter generation**
- ✅ **Scientific accuracy**
- ✅ **Easy to use and extend**
- ✅ **Production-ready results**
