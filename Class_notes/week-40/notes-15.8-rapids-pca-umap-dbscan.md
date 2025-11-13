# Lecture 15.8: RAPIDS Acceleration for PCA, UMAP, and DBSCAN

## Key Learning Objectives
- Understand GPU acceleration for dimensionality reduction algorithms
- Learn RAPIDS cuML implementations of PCA, UMAP, and DBSCAN
- Compare CPU vs GPU performance for unsupervised learning
- Implement scalable unsupervised pipelines with distributed computing

## Core Concepts

### Why GPU for Dimensionality Reduction and Clustering?

#### Computational Characteristics
- **Matrix operations**: PCA involves eigendecomposition and matrix multiplication
- **Distance calculations**: UMAP and DBSCAN compute pairwise distances
- **Iterative optimization**: Gradient descent in UMAP requires many iterations
- **Large datasets**: GPU memory and parallelism handle massive data

#### RAPIDS Advantages
- **Unified ecosystem**: cuML provides consistent GPU API
- **Memory efficiency**: Optimized data structures for GPU memory
- **Scalability**: Handle datasets larger than CPU memory
- **Performance gains**: 10-100x speedup depending on algorithm and data

## PCA with RAPIDS cuML

### GPU-Accelerated PCA
```python
import cudf
import cupy as cp
from cuml.decomposition import PCA as CumlPCA
from cuml.metrics import explained_variance_score

# Load data into GPU memory
X_train_gpu = cudf.DataFrame.from_pandas(X_train)
X_test_gpu = cudf.DataFrame.from_pandas(X_test)

# Create GPU PCA
gpu_pca = CumlPCA(
    n_components=2,        # Number of components
    copy=True,             # Copy input data
    whiten=False,          # Don't whiten (normalize components)
    svd_solver='auto',     # SVD solver: 'auto', 'full', 'jacobi'
    tol=0.0,              # Tolerance for 'jacobi' solver
    iterated_power=15,    # Iterations for 'randomized' solver
    random_state=42
)

# Fit and transform
X_train_pca = gpu_pca.fit_transform(X_train_gpu)
X_test_pca = gpu_pca.transform(X_test_gpu)

# Get explained variance
explained_variance = gpu_pca.explained_variance_
explained_variance_ratio = gpu_pca.explained_variance_ratio_

print(f"Explained variance ratio: {explained_variance_ratio}")
print(f"Cumulative explained variance: {explained_variance_ratio.cumsum()}")

# Visualize PCA results
plt.figure(figsize=(12, 5))

# Plot explained variance
plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance_ratio) + 1), 
         explained_variance_ratio, 'bo-', alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)

# Plot cumulative explained variance
plt.subplot(1, 2, 2)
plt.plot(range(1, len(explained_variance_ratio) + 1), 
         explained_variance_ratio.cumsum(), 'ro-', alpha=0.7)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### PCA Performance Comparison
```python
import time
from sklearn.decomposition import PCA as SklearnPCA

def benchmark_pca(X_train, n_components_list):
    """
    Compare CPU and GPU PCA performance
    """
    results = []
    
    for n_comp in n_components_list:
        print(f"Testing PCA with {n_comp} components...")
        
        # CPU PCA
        cpu_start = time.time()
        cpu_pca = SklearnPCA(n_components=n_comp, random_state=42)
        X_cpu_pca = cpu_pca.fit_transform(X_train)
        cpu_time = time.time() - cpu_start
        
        # GPU PCA
        gpu_start = time.time()
        gpu_pca = CumlPCA(n_components=n_comp, random_state=42)
        X_gpu_pca = gpu_pca.fit_transform(X_train_gpu)
        gpu_time = time.time() - gpu_start
        
        # Calculate explained variance
        cpu_var = cpu_pca.explained_variance_ratio_.sum()
        gpu_var = float(gpu_pca.explained_variance_ratio_.sum())
        
        results.append({
            'n_components': n_comp,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': cpu_time / gpu_time,
            'cpu_variance': cpu_var,
            'gpu_variance': gpu_var
        })
    
    return pd.DataFrame(results)

# Run benchmark
n_components_list = [2, 5, 10, 20, 50]
pca_benchmark = benchmark_pca(X_train, n_components_list)

print(pca_benchmark)

# Visualize benchmark results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(pca_benchmark['n_components'], pca_benchmark['cpu_time'], 'b-o', label='CPU')
plt.plot(pca_benchmark['n_components'], pca_benchmark['gpu_time'], 'r-o', label='GPU')
plt.xlabel('Number of Components')
plt.ylabel('Time (seconds)')
plt.title('PCA Training Time')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(pca_benchmark['n_components'], pca_benchmark['speedup'], 'g-o')
plt.xlabel('Number of Components')
plt.ylabel('Speedup (CPU/GPU)')
plt.title('GPU Speedup')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(pca_benchmark['n_components'], pca_benchmark['cpu_variance'], 'b-o', label='CPU')
plt.plot(pca_benchmark['n_components'], pca_benchmark['gpu_variance'], 'r-o', label='GPU')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## UMAP with RAPIDS cuML

### GPU-Accelerated UMAP
```python
from cuml.manifold import UMAP as CumlUMAP

# Create GPU UMAP
gpu_umap = CumlUMAP(
    n_neighbors=15,        # Local neighborhood size
    n_components=2,        # Dimensions of embedding
    n_epochs=500,          # Number of training epochs
    learning_rate=1.0,     # Learning rate
    min_dist=0.1,          # Minimum distance in embedding
    spread=1.0,            # Scale parameter for min_dist
    set_op_mix_ratio=1.0,  # Balance fuzzy union and intersection
    local_connectivity=1,   # Number of nearest neighbors in local approximation
    repulsion_strength=1.0, # Weighting of negative samples
    negative_sample_rate=5, # Number of negative samples per positive sample
    transform_queue_size=4.0, # Queue size for transformation
    random_state=42,
    verbose=True           # Print progress
)

# Fit and transform
X_train_umap = gpu_umap.fit_transform(X_train_gpu)
X_test_umap = gpu_umap.transform(X_test_gpu)

print(f"UMAP embedding shape: {X_train_umap.shape}")

# Visualize UMAP results
plt.figure(figsize=(10, 8))
plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], alpha=0.6, s=50)
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('GPU UMAP Visualization')
plt.grid(True)
plt.show()
```

### UMAP Performance Comparison
```python
import time

def benchmark_umap(X_train, n_neighbors_list):
    """
    Compare CPU and GPU UMAP performance
    """
    results = []
    
    for n_n in n_neighbors_list:
        print(f"Testing UMAP with {n_n} neighbors...")
        
        # CPU UMAP (if available)
        try:
            import umap as cpu_umap
            cpu_start = time.time()
            cpu_reducer = cpu_umap.UMAP(n_neighbors=n_n, random_state=42, verbose=False)
            X_cpu_umap = cpu_reducer.fit_transform(X_train)
            cpu_time = time.time() - cpu_start
        except ImportError:
            cpu_time = None
            print("CPU UMAP not available")
        
        # GPU UMAP
        gpu_start = time.time()
        gpu_umap = CumlUMAP(n_neighbors=n_n, random_state=42, verbose=False)
        X_gpu_umap = gpu_umap.fit_transform(X_train_gpu)
        gpu_time = time.time() - gpu_start
        
        results.append({
            'n_neighbors': n_n,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': cpu_time / gpu_time if cpu_time else None
        })
    
    return pd.DataFrame(results)

# Run benchmark
n_neighbors_list = [5, 15, 30, 50]
umap_benchmark = benchmark_umap(X_train, n_neighbors_list)

print(umap_benchmark)

# Visualize benchmark
if umap_benchmark['cpu_time'].notna().any():
    plt.figure(figsize=(10, 6))
    plt.plot(umap_benchmark['n_neighbors'], umap_benchmark['cpu_time'], 'b-o', label='CPU')
    plt.plot(umap_benchmark['n_neighbors'], umap_benchmark['gpu_time'], 'r-o', label='GPU')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Time (seconds)')
    plt.title('UMAP Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
```

## DBSCAN with RAPIDS cuML

### GPU-Accelerated DBSCAN
```python
from cuml.cluster import DBSCAN as CumlDBSCAN

# Create GPU DBSCAN
gpu_dbscan = CumlDBSCAN(
    eps=0.5,              # Maximum distance between points
    min_samples=5,        # Minimum samples in neighborhood
    metric='euclidean',   # Distance metric
    algorithm='brute',    # Algorithm: 'brute' or 'auto'
    verbose=True,         # Print progress
    max_mbytes_per_batch=None  # Memory per batch
)

# Fit and predict
gpu_cluster_labels = gpu_dbscan.fit_predict(X_scaled_gpu)

# Analyze results
gpu_n_clusters = len(cp.unique(gpu_cluster_labels)) - (1 if -1 in gpu_cluster_labels.values_host else 0)
gpu_n_noise = cp.sum(gpu_cluster_labels == -1).item()

print(f"GPU DBSCAN - Clusters: {gpu_n_clusters}, Noise points: {gpu_n_noise}")

# Evaluate clustering quality
if gpu_n_clusters > 1:
    mask = gpu_cluster_labels != -1
    if cp.sum(mask) > gpu_n_clusters:
        from cuml.metrics.cluster import silhouette_score
        sil_score = silhouette_score(X_scaled_gpu[mask], gpu_cluster_labels[mask])
        print(f"Silhouette score: {sil_score:.3f}")
```

### DBSCAN Performance Comparison
```python
import time
from sklearn.cluster import DBSCAN as SklearnDBSCAN

def benchmark_dbscan(X_scaled, eps_list, min_samples=5):
    """
    Compare CPU and GPU DBSCAN performance
    """
    results = []
    
    for eps in eps_list:
        print(f"Testing DBSCAN with eps={eps}...")
        
        # CPU DBSCAN
        cpu_start = time.time()
        cpu_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        cpu_labels = cpu_dbscan.fit_predict(X_scaled)
        cpu_time = time.time() - cpu_start
        
        cpu_n_clusters = len(set(cpu_labels)) - (1 if -1 in cpu_labels else 0)
        cpu_n_noise = list(cpu_labels).count(-1)
        
        # GPU DBSCAN
        gpu_start = time.time()
        gpu_dbscan = CumlDBSCAN(eps=eps, min_samples=min_samples, verbose=False)
        gpu_labels = gpu_dbscan.fit_predict(X_scaled_gpu)
        gpu_time = time.time() - gpu_start
        
        gpu_n_clusters = len(cp.unique(gpu_labels)) - (1 if -1 in gpu_labels.values_host else 0)
        gpu_n_noise = cp.sum(gpu_labels == -1).item()
        
        results.append({
            'eps': eps,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': cpu_time / gpu_time,
            'cpu_clusters': cpu_n_clusters,
            'gpu_clusters': gpu_n_clusters,
            'cpu_noise': cpu_n_noise,
            'gpu_noise': gpu_n_noise
        })
    
    return pd.DataFrame(results)

# Run benchmark
eps_list = [0.3, 0.5, 0.7, 1.0]
dbscan_benchmark = benchmark_dbscan(X_scaled, eps_list)

print(dbscan_benchmark)

# Visualize benchmark
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(dbscan_benchmark['eps'], dbscan_benchmark['cpu_time'], 'b-o', label='CPU')
plt.plot(dbscan_benchmark['eps'], dbscan_benchmark['gpu_time'], 'r-o', label='GPU')
plt.xlabel('eps')
plt.ylabel('Time (seconds)')
plt.title('DBSCAN Time Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(dbscan_benchmark['eps'], dbscan_benchmark['speedup'], 'g-o')
plt.xlabel('eps')
plt.ylabel('Speedup (CPU/GPU)')
plt.title('GPU Speedup')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(dbscan_benchmark['eps'], dbscan_benchmark['cpu_clusters'], 'b-o', label='CPU Clusters')
plt.plot(dbscan_benchmark['eps'], dbscan_benchmark['gpu_clusters'], 'r-o', label='GPU Clusters')
plt.plot(dbscan_benchmark['eps'], dbscan_benchmark['cpu_noise'], 'b--', label='CPU Noise')
plt.plot(dbscan_benchmark['eps'], dbscan_benchmark['gpu_noise'], 'r--', label='GPU Noise')
plt.xlabel('eps')
plt.ylabel('Count')
plt.title('Clusters and Noise Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Integrated GPU Unsupervised Pipeline

### Complete GPU Pipeline
```python
import pandas as pd
import numpy as np
import cudf
import cupy as cp
from cuml.decomposition import PCA
from cuml.manifold import UMAP
from cuml.cluster import KMeans, DBSCAN
from cuml.metrics.cluster import silhouette_score
from cuml.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# Load and prepare data
print("Loading data...")
df = pd.read_csv('large_unsupervised_dataset.csv')

# Handle potential ID columns
id_columns = [col for col in df.columns if 'id' in col.lower() or 'index' in col.lower()]
if id_columns:
    X = df.drop(id_columns, axis=1)
else:
    X = df.copy()

print(f"Dataset shape: {X.shape}")

# GPU preprocessing
print("GPU preprocessing...")
scaler = StandardScaler()
X_scaled_gpu = scaler.fit_transform(cudf.DataFrame.from_pandas(X))

# Step 1: GPU PCA for dimensionality reduction
print("Step 1: GPU PCA...")
pca_start = time.time()

gpu_pca = PCA(n_components=min(50, X.shape[1]), random_state=42)
X_pca_gpu = gpu_pca.fit_transform(X_scaled_gpu)

pca_time = time.time() - pca_start
explained_var = gpu_pca.explained_variance_ratio_
cumulative_var = cp.cumsum(explained_var)

print(f"PCA completed in {pca_time:.2f} seconds")
print(f"Explained variance by first 10 components: {explained_var[:10]}")

# Determine optimal number of components (95% variance)
n_components_95 = cp.where(cumulative_var >= 0.95)[0][0] + 1
print(f"Components needed for 95% variance: {n_components_95}")

# Use optimal components
X_pca_optimal = X_pca_gpu[:, :n_components_95]

# Step 2: GPU UMAP for visualization
print("Step 2: GPU UMAP...")
umap_start = time.time()

gpu_umap = UMAP(
    n_neighbors=15,
    n_components=2,
    n_epochs=500,
    learning_rate=1.0,
    min_dist=0.1,
    random_state=42,
    verbose=False
)

X_umap_gpu = gpu_umap.fit_transform(X_pca_optimal)
umap_time = time.time() - umap_start

print(f"UMAP completed in {umap_time:.2f} seconds")

# Step 3: GPU clustering comparison
print("Step 3: GPU clustering...")

# K-means clustering
kmeans_start = time.time()
gpu_kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = gpu_kmeans.fit_predict(X_pca_optimal)
kmeans_time = time.time() - kmeans_start

# DBSCAN clustering
dbscan_start = time.time()
gpu_dbscan = DBSCAN(eps=0.5, min_samples=5, verbose=False)
dbscan_labels = gpu_dbscan.fit_predict(X_pca_optimal)
dbscan_time = time.time() - dbscan_start

print(f"K-means completed in {kmeans_time:.2f} seconds")
print(f"DBSCAN completed in {dbscan_time:.2f} seconds")

# Analyze clustering results
kmeans_n_clusters = len(cp.unique(kmeans_labels))
kmeans_n_noise = 0

dbscan_n_clusters = len(cp.unique(dbscan_labels)) - (1 if -1 in dbscan_labels.values_host else 0)
dbscan_n_noise = cp.sum(dbscan_labels == -1).item()

print(f"K-means: {kmeans_n_clusters} clusters")
print(f"DBSCAN: {dbscan_n_clusters} clusters, {dbscan_n_noise} noise points")

# Evaluate clustering quality
if kmeans_n_clusters > 1:
    kmeans_silhouette = silhouette_score(X_pca_optimal, kmeans_labels)
    print(f"K-means silhouette: {kmeans_silhouette:.3f}")

if dbscan_n_clusters > 1:
    mask = dbscan_labels != -1
    if cp.sum(mask) > dbscan_n_clusters:
        dbscan_silhouette = silhouette_score(X_pca_optimal[mask], dbscan_labels[mask])
        print(f"DBSCAN silhouette: {dbscan_silhouette:.3f}")

# Step 4: Visualization and analysis
print("Step 4: Creating visualizations...")

# Convert to numpy for plotting
X_umap_cpu = X_umap_gpu.values
kmeans_labels_cpu = kmeans_labels.values
dbscan_labels_cpu = dbscan_labels.values

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# PCA explained variance
ax = axes[0, 0]
explained_var_cpu = explained_var.values
ax.plot(range(1, len(explained_var_cpu) + 1), explained_var_cpu, 'bo-', alpha=0.7)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title('PCA Scree Plot')
ax.grid(True)

# UMAP embedding
ax = axes[0, 1]
scatter = ax.scatter(X_umap_cpu[:, 0], X_umap_cpu[:, 1], alpha=0.6, s=30, c='blue')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('UMAP Embedding')
ax.grid(True)

# K-means clusters on UMAP
ax = axes[1, 0]
unique_kmeans = np.unique(kmeans_labels_cpu)
colors_kmeans = plt.cm.tab10(np.linspace(0, 1, len(unique_kmeans)))

for i, label in enumerate(unique_kmeans):
    mask = kmeans_labels_cpu == label
    ax.scatter(X_umap_cpu[mask, 0], X_umap_cpu[mask, 1], 
              c=[colors_kmeans[i]], alpha=0.7, s=50, label=f'Cluster {label}')

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title(f'K-means Clusters (k={kmeans_n_clusters})')
ax.legend()
ax.grid(True)

# DBSCAN clusters on UMAP
ax = axes[1, 1]
unique_dbscan = np.unique(dbscan_labels_cpu)
colors_dbscan = plt.cm.tab10(np.linspace(0, 1, len(unique_dbscan)))

for i, label in enumerate(unique_dbscan):
    mask = dbscan_labels_cpu == label
    if label == -1:
        ax.scatter(X_umap_cpu[mask, 0], X_umap_cpu[mask, 1], 
                  c='black', alpha=0.5, s=30, label='Noise')
    else:
        ax.scatter(X_umap_cpu[mask, 0], X_umap_cpu[mask, 1], 
                  c=[colors_dbscan[i]], alpha=0.7, s=50, label=f'Cluster {label}')

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title(f'DBSCAN Clusters ({dbscan_n_clusters} clusters, {dbscan_n_noise} noise)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

# Step 5: Performance summary
print("
=== GPU Unsupervised Learning Performance Summary ===")
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"PCA components: {n_components_95} (95% variance)")
print(f"PCA time: {pca_time:.2f} seconds")
print(f"UMAP time: {umap_time:.2f} seconds")
print(f"K-means time: {kmeans_time:.2f} seconds")
print(f"DBSCAN time: {dbscan_time:.2f} seconds")
print(f"Total GPU time: {pca_time + umap_time + kmeans_time + dbscan_time:.2f} seconds")

# Step 6: Save results
print("Saving results...")
results_df = pd.DataFrame({
    'umap_1': X_umap_cpu[:, 0],
    'umap_2': X_umap_cpu[:, 1],
    'kmeans_cluster': kmeans_labels_cpu,
    'dbscan_cluster': dbscan_labels_cpu
})

# Add PCA components if not too many
if n_components_95 <= 10:
    for i in range(n_components_95):
        results_df[f'pca_{i+1}'] = X_pca_gpu[:, i].values

results_df.to_csv('gpu_unsupervised_analysis.csv', index=False)
print("Results saved to 'gpu_unsupervised_analysis.csv'")

print("\nGPU unsupervised learning pipeline completed!")
```

## Distributed Computing with Dask

### Dask cuML for Massive Datasets
```python
import dask
import dask_cudf
from dask.distributed import Client, LocalCluster
from cuml.dask.decomposition import PCA as DaskPCA
from cuml.dask.manifold import UMAP as DaskUMAP
from cuml.dask.cluster import KMeans as DaskKMeans

# Start Dask cluster
cluster = LocalCluster(n_workers=2, threads_per_worker=1)
client = Client(cluster)

# Load data with dask-cudf
X_dask = dask_cudf.read_csv('massive_dataset.csv')
X_dask = X_dask.drop(['id'], axis=1, errors='ignore')

# Distributed GPU PCA
dask_pca = DaskPCA(n_components=50, random_state=42)
X_pca_dask = dask_pca.fit_transform(X_dask)

# Distributed GPU UMAP
dask_umap = DaskUMAP(n_neighbors=15, n_components=2, random_state=42)
X_umap_dask = dask_umap.fit_transform(X_pca_dask)

# Distributed GPU K-means
dask_kmeans = DaskKMeans(n_clusters=10, random_state=42)
cluster_labels_dask = dask_kmeans.fit_predict(X_pca_dask)

# Compute results
final_umap = X_umap_dask.compute()
final_labels = cluster_labels_dask.compute()

print(f"Distributed processing completed on {len(X_dask)} samples")
```

## Best Practices

### GPU Memory Management
1. **Monitor usage**: Track GPU memory throughout processing
2. **Batch processing**: Process large datasets in chunks
3. **Data types**: Use appropriate precision (float32 vs float64)
4. **Cleanup**: Explicitly free GPU memory when done

### Performance Optimization
1. **Algorithm selection**: Choose GPU-optimized algorithms
2. **Parameter tuning**: Optimize for GPU architecture
3. **Data preprocessing**: Prepare data efficiently for GPU
4. **Batch sizes**: Tune batch sizes for optimal performance

### Production Deployment
1. **Model persistence**: Save trained GPU models
2. **Inference optimization**: Optimize for real-time prediction
3. **Resource management**: Monitor GPU resources in production
4. **Fallback strategies**: Have CPU implementations for edge cases

## Troubleshooting

### Common GPU Issues
- **Out of memory**: Reduce batch sizes, use data chunking, or reduce model complexity
- **Slow data transfer**: Minimize CPU-GPU transfers, keep data on GPU
- **Convergence issues**: Adjust algorithm parameters for GPU-specific behavior
- **Installation problems**: Ensure proper CUDA and RAPIDS versions

### Performance Debugging
```python
# Monitor GPU usage during processing
import subprocess
import time

def monitor_gpu_during_processing(duration=60):
    """Monitor GPU usage during long-running processes"""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        print(f"Time: {time.time() - start_time:.1f}s - {result.stdout.strip()}")
        time.sleep(5)

# Time individual operations
operations = [
    ("PCA", lambda: gpu_pca.fit_transform(X_train_gpu)),
    ("UMAP", lambda: gpu_umap.fit_transform(X_pca_gpu)),
    ("K-means", lambda: gpu_kmeans.fit_predict(X_pca_gpu))
]

for name, operation in operations:
    start = time.time()
    result = operation()
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.2f} seconds")
```

## Connection to Scalable Computing

### Integration with Previous Modules
- **Dask**: Distributed GPU processing for massive datasets
- **Spark**: GPU acceleration within Spark ML pipelines
- **HBase**: Fast feature retrieval for GPU models
- **Module 14**: Feature engineering from supervised learning

### Enterprise Applications
- **Real-time dimensionality reduction**: GPU-accelerated preprocessing
- **Large-scale clustering**: Distributed clustering for massive datasets
- **Interactive visualization**: GPU-powered exploration of high-dimensional data
- **Automated feature extraction**: GPU-accelerated unsupervised feature learning

## Summary

This lecture completes Module 15 by demonstrating how RAPIDS cuML enables GPU acceleration for the complete unsupervised learning pipeline:

1. **PCA**: GPU-accelerated eigendecomposition and dimensionality reduction
2. **UMAP**: GPU-accelerated non-linear dimensionality reduction
3. **DBSCAN**: GPU-accelerated density-based clustering
4. **Integration**: Complete pipelines combining all techniques
5. **Distributed computing**: Dask integration for massive scale

The module successfully bridges the gap between traditional unsupervised learning and modern GPU-accelerated computing, preparing students for real-world applications with large, complex datasets.

**Key Achievements**:
- Understanding of unsupervised learning fundamentals
- Practical implementation of clustering and dimensionality reduction
- GPU acceleration for scalable unsupervised learning
- Integration with distributed computing frameworks
- Production-ready unsupervised learning pipelines