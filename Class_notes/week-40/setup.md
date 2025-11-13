# Module 15: Machine Learning Clustering and Dimensionality Reduction - Setup Guide

## Instructor Information
**Lead Instructor**: Dennis Omboga Mongare
**Role**: Lead Data Science Instructor
**Contact**: [Your contact information]
**Course**: Data Science B - Unsupervised Learning Series

## Environment Setup with UV

This module uses `uv` for dependency management and virtual environment isolation. Follow these steps to set up your environment.

### Prerequisites
- Python 3.13 or higher
- `uv` package manager (install with: `pip install uv` or follow [uv installation guide](https://github.com/astral-sh/uv))
- GPU support recommended for RAPIDS acceleration

### Setup Commands

#### Linux (Cinnamon Manjaro)
```bash
# 1. Navigate to the module directory
cd "Data-Science-B/15_module_machine_learning_clustering_and_dimensionality_Reduction"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
python -c "import cuml; print(f'RAPIDS cuML version: {cuml.__version__}')"
python -c "import cudf; print('RAPIDS cuDF available')"
```

#### Windows (PowerShell/Command Prompt)
```powershell
# 1. Navigate to the module directory
cd "Data-Science-B\15_module_machine_learning_clustering_and_dimensionality_Reduction"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
.venv\Scripts\activate

# 4. Verify installation
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
python -c "import cuml; print(f'RAPIDS cuML version: {cuml.__version__}')"
python -c "import cudf; print('RAPIDS cuDF available')"
```

#### macOS (Terminal)
```bash
# 1. Navigate to the module directory
cd "Data-Science-B/15_module_machine_learning_clustering_and_dimensionality_Reduction"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
python -c "import cuml; print(f'RAPIDS cuML version: {cuml.__version__}')"
python -c "import cudf; print('RAPIDS cuDF available')"
```

### Running the Scripts

#### Python Scripts
```bash
# Activate environment (if not already activated)
source .venv/bin/activate

# Run clustering lecture demo
python "lecture_demo.py"

# Run clustering student lab
python "student.ipynb"  # Convert to .py if needed

# Run specific analysis scripts
python "clustering_analysis.py"
python "dimensionality_reduction.py"
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
- **scikit-learn 1.3.0+**: Traditional ML algorithms
- **RAPIDS cuML 23.0+**: GPU-accelerated ML
- **RAPIDS cuDF 23.0+**: GPU DataFrames
- **pandas 2.0.0+**: Data manipulation
- **numpy 1.24.0+**: Numerical computing
- **matplotlib 3.6.0+**: Data visualization
- **seaborn 0.12.0+**: Statistical visualization
- **jupyter 1.0.0+**: Interactive notebooks
- **ipykernel 6.0.0+**: Jupyter kernel
- **umap-learn 0.5.0+**: CPU UMAP implementation
- **hdbscan 0.8.0+**: Hierarchical DBSCAN

### GPU Setup (Recommended)

#### RAPIDS Installation Notes
RAPIDS requires CUDA-compatible GPU:
- **CUDA 11.8+** or **CUDA 12.0+**
- **NVIDIA GPU** with compute capability 6.0+
- **GPU Memory**: 4GB+ recommended

#### GPU Verification
```bash
# Check GPU availability
python -c "import cuml; print(f'CUDA available: {cuml.is_cuda_available()}')"

# Check GPU memory
python -c "
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f'GPU Memory: {info.total/1024**3:.1f}GB total, {info.free/1024**3:.1f}GB free')
"
```

### Troubleshooting

#### Environment Issues
```bash
# Re-sync environment
uv sync --reinstall

# Clear cache and re-sync
uv cache clean
uv sync
```

#### GPU Memory Issues
```bash
# Monitor GPU memory usage
nvidia-smi

# Check GPU processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

#### RAPIDS Installation Issues
If RAPIDS fails to install:
```bash
# Check CUDA version
nvcc --version

# Install specific RAPIDS version for your CUDA version
# CUDA 11.8: pip install cuml-cu118
# CUDA 12.0: pip install cuml-cu120
```

#### Deactivate Environment
```bash
# When done working
deactivate
```

### Dataset Requirements

The scripts expect various datasets. Download if missing:
```bash
# Example datasets
curl -o iris.csv https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv
curl -o wine.csv https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
curl -o digits.csv https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/digits.csv
```

### Support

If you encounter issues:
1. Ensure you're in the correct directory
2. Verify the virtual environment is activated
3. Check that all dependencies are installed with `uv sync`
4. For GPU issues, verify CUDA installation and GPU compatibility
5. Monitor memory usage when working with large datasets

### Key Libraries Covered

#### Clustering Algorithms
- **K-means**: Partitioning clustering
- **Hierarchical**: Agglomerative clustering
- **DBSCAN**: Density-based clustering
- **HDBSCAN**: Hierarchical density clustering

#### Dimensionality Reduction
- **PCA**: Principal Component Analysis
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding
- **UMAP**: Uniform Manifold Approximation and Projection

#### GPU Acceleration
- **cuML K-means**: GPU-accelerated K-means
- **cuML PCA**: GPU-accelerated PCA
- **cuML UMAP**: GPU-accelerated UMAP
- **cuML DBSCAN**: GPU-accelerated DBSCAN

### Performance Expectations

#### CPU vs GPU Performance
- **Small datasets (< 10K samples)**: 2-5x GPU speedup
- **Medium datasets (10K-100K samples)**: 5-20x GPU speedup
- **Large datasets (> 100K samples)**: 20-100x GPU speedup
- **High-dimensional data**: Even larger speedups

#### Memory Requirements
- **CPU**: Limited by system RAM
- **GPU**: Limited by GPU memory (typically 8-24GB)
- **Large datasets**: May require chunking or distributed processing

### Next Steps

After setting up the environment:
1. Run the lecture demo to understand basic concepts
2. Complete the student lab exercises
3. Experiment with different algorithms and parameters
4. Try GPU acceleration on larger datasets
5. Explore advanced techniques and custom implementations

---

**Instructor**: Dennis Omboga Mongare
**Last Updated**: October 2025
**Version**: 1.0