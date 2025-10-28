# Module 14 Assignment: GPU-Accelerated Classification with RAPIDS

## Overview
In this assignment, you will implement and compare GPU-accelerated machine learning classification algorithms using RAPIDS cuML. You'll work with large-scale datasets that demonstrate the performance benefits of GPU acceleration over traditional CPU-based implementations.

## Learning Objectives
- Implement GPU-accelerated classification algorithms
- Compare CPU vs GPU performance on large datasets
- Handle memory management for GPU computing
- Optimize workflows for accelerated computing
- Work with cloud-based GPU resources

## Dataset Options

### Option 1: Large-Scale Tabular Dataset
**California Housing Dataset** (206,040 samples, 8 features)
- Predict housing prices using regression (can be adapted for classification)
- Large enough to show GPU acceleration benefits
- Real-world geospatial data

### Option 2: High-Dimensional Classification Dataset
**Fashion MNIST** (70,000 samples, 784 features)
- Image classification task
- High-dimensional feature space
- Good for demonstrating dimensionality challenges

### Option 3: Custom Large Dataset
**NYC Taxi Dataset** (subset of 10M+ records)
- Time-series classification
- Real-world big data scenario
- Complex feature engineering required

## Assignment Tasks

### Part 1: Environment Setup and Data Preparation (20 points)

1. **Cloud GPU Environment Setup**
   - Set up Google Colab with GPU runtime
   - Configure AWS SageMaker or similar
   - Install RAPIDS cuML and cuDF
   - Verify GPU availability and memory

2. **Data Acquisition and Preprocessing**
   - Download and load large dataset
   - Handle data types and memory optimization
   - Implement efficient data loading with cuDF
   - Perform initial data exploration

### Part 2: GPU-Accelerated Model Implementation (40 points)

1. **Implement Multiple Algorithms**
   - Random Forest Classifier (cuML)
   - Logistic Regression (cuML)
   - K-Nearest Neighbors (cuML)
   - Support Vector Machine (cuML)
   - Compare with CPU scikit-learn versions

2. **Performance Optimization**
   - Memory management for large datasets
   - Batch processing for out-of-memory scenarios
   - Hyperparameter tuning with GPU acceleration
   - Cross-validation on GPU

3. **Scalability Testing**
   - Test with different dataset sizes
   - Measure memory usage and processing time
   - Identify performance bottlenecks

### Part 3: Performance Analysis and Comparison (30 points)

1. **Benchmarking**
   - Compare CPU vs GPU performance
   - Measure training time, prediction time, memory usage
   - Analyze speedup factors for different algorithms
   - Create comprehensive performance reports

2. **Scalability Analysis**
   - Test with increasing dataset sizes
   - Identify breaking points for CPU vs GPU
   - Analyze memory vs performance trade-offs

3. **Cost-Benefit Analysis**
   - Calculate computational costs
   - Analyze when GPU acceleration is beneficial
   - Provide recommendations for different use cases

## Technical Requirements

### Cloud Environment Options

#### Option A: Google Colab (Free)
```python
# Check GPU availability
!nvidia-smi

# Install RAPIDS
!pip install cuml-cu11 cudf-cu11 --extra-index-url=https://pypi.ngc.nvidia.com
```

#### Option B: Kaggle Notebooks (Free)
```python
# Enable GPU acceleration
# Use RAPIDS environment
```

#### Option C: AWS SageMaker (Paid)
```bash
# Configure GPU instance
# Install RAPIDS
pip install cuml-cu11 cudf-cu11
```

### Required Libraries
- cuml (RAPIDS ML)
- cudf (RAPIDS DataFrames)
- scikit-learn (CPU comparison)
- matplotlib/seaborn (visualization)
- pandas/numpy (data manipulation)

### Performance Metrics to Track
- Training time
- Prediction time
- Memory usage (GPU and CPU)
- Model accuracy
- Speedup ratio (CPU vs GPU)

## Evaluation Criteria

### Technical Implementation (40%)
- Correct RAPIDS/cuML implementation
- Proper GPU memory management
- Efficient data processing pipelines
- Error handling and debugging

### Performance Analysis (30%)
- Comprehensive benchmarking
- Accurate performance measurements
- Clear visualization of results
- Meaningful interpretation of speedup factors

### Scalability and Optimization (20%)
- Effective handling of large datasets
- Memory optimization techniques
- Performance tuning strategies
- Resource utilization efficiency

### Report and Documentation (10%)
- Clear methodology documentation
- Comprehensive results analysis
- Professional presentation
- Actionable recommendations

## Submission Requirements

1. **Jupyter Notebook** containing:
   - Complete implementation with GPU acceleration
   - Performance benchmarking code
   - Visualization of results
   - Comparative analysis

2. **Performance Report** including:
   - Setup and environment details
   - Benchmarking methodology
   - Results analysis and interpretation
   - Recommendations and conclusions

3. **GitHub Repository** with:
   - Well-documented code
   - Requirements.txt or environment.yml
   - README with setup instructions
   - Performance metrics and results

## Advanced Challenges (Bonus Points)

1. **Multi-GPU Implementation**
   - Distribute workload across multiple GPUs
   - Implement data parallelism

2. **Custom CUDA Kernels**
   - Optimize specific operations with custom CUDA code
   - Performance comparison with cuML implementations

3. **Real-time Inference**
   - Implement model serving with GPU acceleration
   - Optimize for low-latency predictions

4. **Energy Efficiency Analysis**
   - Compare power consumption CPU vs GPU
   - Analyze performance per watt metrics

## Free Resources and Datasets

### Cloud GPU Platforms
- **Google Colab**: Free GPU (Tesla T4/K80)
- **Kaggle**: Free GPU (Tesla P100)
- **Paperspace**: Free tier with GPU
- **AWS**: Free tier + credits for students

### Large Datasets
- **California Housing**: sklearn.datasets.fetch_california_housing()
- **Fashion MNIST**: tensorflow.keras.datasets.fashion_mnist
- **NYC Taxi**: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Census Income**: https://archive.ics.uci.edu/dataset/20/census+income

### Learning Resources
- [RAPIDS Documentation](https://docs.rapids.ai/)
- [cuML API Reference](https://docs.rapids.ai/api/cuml/stable/)
- [GPU Programming Guide](https://developer.nvidia.com/cuda-toolkit)

## Important Notes

- **GPU Availability**: Ensure your cloud environment has GPU access
- **Memory Management**: Monitor GPU memory usage carefully
- **Cost Awareness**: Track usage on paid cloud platforms
- **Reproducibility**: Document exact environment and versions

## Deadline
[Insert deadline here]

---

**Instructor**: Dennis Omboga Mongare
**Course**: Data Science B - GPU-Accelerated Machine Learning
**Assignment Weight**: 25% of Module 14 Grade