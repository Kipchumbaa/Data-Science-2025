# Module 14: Machine Learning Classification - Setup Guide

## Instructor Information
**Lead Instructor**: Dennis Omboga Mongare
**Role**: Lead Data Science Instructor
**Contact**: [Your contact information]
**Course**: Data Science B - Scalable Computing Series

## Environment Setup with UV

This module uses `uv` for dependency management and virtual environment isolation. Follow these steps to set up your environment.

### Prerequisites
- Python 3.13 or higher
- `uv` package manager (install with: `pip install uv` or follow [uv installation guide](https://github.com/astral-sh/uv))
- At least 8GB free disk space (cleaned temporary caches if needed)
- CPU-only setup (GPU acceleration optional for advanced users)

### Setup Commands

#### Linux (Cinnamon Manjaro)
```bash
# 1. Navigate to the module directory
cd "Data-Science-B/14_module_machine_learning_classification"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
```

#### Windows (PowerShell/Command Prompt)
```powershell
# 1. Navigate to the module directory
cd "Data-Science-B\14_module_machine_learning_classification"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
.venv\Scripts\activate

# 4. Verify installation
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
```

#### macOS (Terminal)
```bash
# 1. Navigate to the module directory
cd "Data-Science-B/14_module_machine_learning_classification"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
```

### Memory Optimization (Important for Cinnamon Manjaro)

#### System Memory Management
```bash
# Check current memory usage
free -h

# Clear system cache
sudo sync; sudo echo 3 > /proc/sys/vm/drop_caches

# Monitor disk usage
df -h
```

#### Python Memory Optimization
```python
# In your Python code, add memory management:
import gc
import sys

# Force garbage collection
gc.collect()

# Monitor memory usage
def memory_usage():
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory usage: {memory_usage():.2f} MB")
```

#### Clean Temporary Files (if needed)
```bash
# Remove pip cache
rm -rf ~/.cache/pip

# Remove uv cache
rm -rf ~/.cache/uv

# Clean package manager cache
sudo pacman -Scc  # Careful - removes all cached packages
```

### Running the Scripts

#### Python Scripts
```bash
# Activate environment (if not already activated)
source .venv/bin/activate

# Run lecture demo
python "lecture_demo.py"

# Run student lab
python "student_lab.py"
```

#### Jupyter Notebooks
```bash
# Activate environment (if not already activated)
source .venv/bin/activate

# Run Jupyter notebook server
jupyter notebook

# Or run specific notebooks
jupyter notebook "lecture_demo.ipynb"
jupyter notebook "student.ipynb"
```

#### FINAL WORKING COMMANDS FOR MODULES 14 & 15

Now that you have disk space, the environments should install successfully:

**Module 14 (Classification):**
```bash
cd "Data-Science-B/14_module_machine_learning_classification"
uv sync
source .venv/bin/activate
jupyter notebook lecture_demo.ipynb
# OR
jupyter notebook student.ipynb
```

**Module 15 (Clustering):**
```bash
cd "Data-Science-B/15_module_machine_learning_clustering_and_dimensionality_Reduction"
uv sync
source .venv/bin/activate
jupyter notebook lecture_demo.ipynb
# OR
jupyter notebook student.ipynb
```

### Dependencies Included

This environment includes:
- **scikit-learn 1.3.0+**: Core machine learning algorithms
- **pandas 2.0.0+**: Data manipulation
- **numpy 1.24.0+**: Numerical computing
- **matplotlib 3.6.0+**: Data visualization
- **seaborn 0.12.0+**: Statistical visualization
- **jupyter 1.0.0+**: Interactive notebooks
- **ipykernel 6.0.0+**: Jupyter kernel

**Note**: GPU dependencies removed for memory optimization. CPU-only implementations provide excellent performance for learning and most practical applications.

### Memory Management Best Practices

#### Data Processing Memory Tips
```python
# Use memory-efficient data types
df['Survived'] = df['Survived'].astype('int8')  # Instead of int64
df['Pclass'] = df['Pclass'].astype('category')  # Categorical for repeated values

# Process data in chunks for large datasets
chunk_size = 1000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    pass

# Delete unused variables
del large_dataframe
gc.collect()
```

#### Memory Optimization Tips
- Process data in batches for large datasets
- Use appropriate data types (int8, float32 instead of float64)
- Clear memory between operations with `gc.collect()`
- Monitor memory usage during processing
- Use chunked reading for large files

### Troubleshooting

#### Environment Issues
```bash
# Re-sync environment
uv sync --reinstall

# Clear cache and re-sync
uv cache clean
uv sync
```

#### Memory Issues
- **Out of memory**: Reduce batch sizes, use chunked processing
- **Slow performance**: Use memory-efficient data types
- **Large datasets**: Process in chunks, use appropriate data types
- **System hangs**: Monitor memory usage, clear caches regularly

#### Disk Space Issues
- **No space left**: Clean package caches, remove temporary files
- **Installation fails**: Free up disk space, use CPU-only dependencies
- **Slow downloads**: Clear download caches, check network connection

#### Deactivate Environment
```bash
# When done working
deactivate
```

### Dataset Requirements

The scripts expect sample datasets. Download if missing:
```bash
# Example datasets (modify as needed)
curl -o titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
curl -o iris.csv https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv
```

### Support

If you encounter issues:
1. Ensure you're in the correct directory
2. Verify the virtual environment is activated
3. Check that all dependencies are installed with `uv sync`
4. For GPU issues, verify CUDA installation and GPU compatibility
5. Monitor memory usage when working with large datasets

### Key ML Concepts Covered

- **Supervised Learning**: Training, validation, test sets
- **Classification Algorithms**: Linear models, trees, ensembles
- **Model Evaluation**: Confusion matrix, ROC-AUC, cross-validation
- **GPU Acceleration**: RAPIDS cuML for performance
- **Hyperparameter Tuning**: Grid search, cross-validation
- **Scalable ML**: Integration with Dask and distributed computing

### Performance Expectations

#### CPU-Only Setup (Current Configuration)
- Suitable for learning and small to medium datasets
- All algorithms work with excellent performance for educational purposes
- Good for understanding concepts and practical applications
- Memory-efficient for systems with limited resources
- Perfect for development and prototyping

#### Memory-Optimized Performance
- Reduced memory footprint compared to GPU setups
- Faster installation and startup times
- Stable performance on resource-constrained systems
- Suitable for production use on CPU-only infrastructure

### Next Steps

After setup completion:
1. Run the lecture demo to understand key concepts
2. Complete the student lab exercises
3. Experiment with different algorithms and parameters
4. Monitor memory usage and optimize as needed
5. Apply techniques to real datasets

### System Health Check

Before starting, verify your system is ready:

```bash
# Check available disk space
df -h /

# Check available memory
free -h

# Verify uv installation
uv --version

# Check Python version
python --version
```

---

**Instructor**: Dennis Omboga Mongare
**Last Updated**: October 2025
**Version**: 2.0 (Memory Optimized)
