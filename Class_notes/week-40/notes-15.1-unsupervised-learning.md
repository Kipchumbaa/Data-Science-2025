# Lecture 15.1: Introduction to Unsupervised Learning

## Key Learning Objectives
- Understand the difference between supervised and unsupervised learning
- Learn the goals and applications of unsupervised learning
- Master data preprocessing for unsupervised methods
- Recognize when to use unsupervised vs supervised approaches

## Core Concepts

### Supervised vs Unsupervised Learning

#### Supervised Learning (Review)
- **Labeled data**: Input-output pairs (X, y)
- **Goal**: Learn mapping from inputs to outputs
- **Examples**: Classification, regression
- **Evaluation**: Compare predictions to true labels

#### Unsupervised Learning
- **Unlabeled data**: Only input features X (no y)
- **Goal**: Discover hidden patterns and structures
- **Examples**: Clustering, dimensionality reduction
- **Evaluation**: Internal metrics, domain knowledge

### Types of Unsupervised Learning

#### Clustering
- **Goal**: Group similar data points together
- **Output**: Cluster assignments for each point
- **Applications**: Customer segmentation, anomaly detection
- **Algorithms**: K-means, hierarchical, DBSCAN

#### Dimensionality Reduction
- **Goal**: Reduce number of features while preserving information
- **Output**: Lower-dimensional representation
- **Applications**: Visualization, noise reduction, feature extraction
- **Algorithms**: PCA, t-SNE, UMAP

#### Other Types
- **Association rule mining**: Find relationships between variables
- **Anomaly detection**: Identify unusual patterns
- **Generative modeling**: Learn data distribution

## Mathematical Foundations

### Distance Metrics

#### Euclidean Distance
```
d(p,q) = √∑(p_i - q_i)²
```
- **Properties**: Always positive, symmetric
- **Use cases**: Continuous numerical data
- **Sensitivity**: Affected by scale differences

#### Manhattan Distance
```
d(p,q) = ∑|p_i - q_i|
```
- **Properties**: Sum of absolute differences
- **Use cases**: Grid-based data, categorical features
- **Robustness**: Less sensitive to outliers than Euclidean

#### Cosine Similarity
```
cosine(p,q) = (p·q) / (||p|| × ||q||)
```
- **Properties**: Measures angle between vectors
- **Use cases**: Text data, high-dimensional sparse data
- **Range**: [-1, 1] (1 = identical, -1 = opposite)

### Data Preprocessing for Unsupervised Learning

#### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (zero mean, unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max scaling (0 to 1 range)
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)
```

#### Handling Categorical Variables
```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# One-hot encoding for nominal variables
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(X_categorical)

# Label encoding for ordinal variables
le = LabelEncoder()
X_ordinal = le.fit_transform(X_ordinal)
```

#### Missing Value Handling
```python
from sklearn.impute import SimpleImputer

# Mean imputation for numerical
num_imputer = SimpleImputer(strategy='mean')
X_num_imputed = num_imputer.fit_transform(X_numerical)

# Most frequent for categorical
cat_imputer = SimpleImputer(strategy='most_frequent')
X_cat_imputed = cat_imputer.fit_transform(X_categorical)
```

## Evaluation Challenges

### Internal Evaluation Metrics

#### Clustering Quality Metrics
- **Within-cluster sum of squares (WCSS)**: Compactness measure
- **Between-cluster sum of squares (BCSS)**: Separation measure
- **Silhouette coefficient**: Individual point cluster quality
- **Calinski-Harabasz index**: Ratio of between/within cluster variance

#### Dimensionality Reduction Metrics
- **Explained variance ratio**: Information preserved (PCA)
- **Reconstruction error**: How well data can be reconstructed
- **Trustworthiness/Continuity**: Neighborhood preservation
- **Visualization quality**: Human interpretability

### External Evaluation (When Labels Available)
```python
from sklearn.metrics import adjusted_rand_score, homogeneity_score

# Adjusted Rand Index (ARI)
ari = adjusted_rand_score(true_labels, predicted_labels)

# Homogeneity score
homogeneity = homogeneity_score(true_labels, predicted_labels)
```

## Practical Applications

### Customer Analytics
- **Segmentation**: Group customers by behavior/purchase patterns
- **Personalization**: Tailored marketing strategies
- **Churn prediction**: Identify at-risk customers

### Computer Vision
- **Image segmentation**: Group pixels into meaningful regions
- **Feature learning**: Extract meaningful features from images
- **Anomaly detection**: Identify unusual images or patterns

### Natural Language Processing
- **Topic modeling**: Discover themes in text collections
- **Document clustering**: Group similar documents
- **Word embeddings**: Learn semantic relationships

### Bioinformatics
- **Gene expression analysis**: Cluster genes with similar expression
- **Protein classification**: Group proteins by function/structure
- **Disease subtype discovery**: Identify patient subgroups

## Algorithm Selection Guidelines

### When to Use Clustering
- **Data exploration**: Understand data structure
- **Customer segmentation**: Marketing applications
- **Anomaly detection**: Quality control, fraud detection
- **Preprocessing**: Feature engineering for supervised learning

### When to Use Dimensionality Reduction
- **Visualization**: High-dimensional data exploration
- **Noise reduction**: Remove irrelevant features
- **Computational efficiency**: Speed up other algorithms
- **Feature extraction**: Create new meaningful features

### Choosing Between Methods
- **Linear relationships**: Use PCA
- **Non-linear manifolds**: Use t-SNE, UMAP
- **Preserve global structure**: Use PCA, MDS
- **Preserve local structure**: Use t-SNE, UMAP

## Implementation Considerations

### Computational Complexity
- **K-means**: O(n × k × d × i) - scalable for large n
- **Hierarchical**: O(n²) - expensive for large datasets
- **DBSCAN**: O(n × log n) - efficient with good indexing
- **PCA**: O(n × d²) - efficient for high dimensions
- **t-SNE**: O(n²) - expensive, use approximations

### Scalability Solutions
- **Mini-batch processing**: Process data in chunks
- **Approximate algorithms**: Faster but less accurate
- **GPU acceleration**: RAPIDS for massive speedup
- **Distributed computing**: Dask for cluster-scale processing

## Common Challenges

### Data Quality Issues
- **Missing values**: Appropriate imputation strategies
- **Outliers**: Robust algorithms or preprocessing
- **Scale differences**: Feature normalization
- **Mixed data types**: Proper encoding strategies

### Algorithm Selection
- **Unknown cluster number**: Try multiple k values
- **Arbitrary cluster shapes**: DBSCAN for irregular shapes
- **High dimensionality**: Dimensionality reduction first
- **Large datasets**: GPU acceleration or sampling

### Evaluation Difficulties
- **No ground truth**: Internal metrics only
- **Subjective interpretation**: Domain expertise required
- **Multiple valid solutions**: Different algorithms may find different patterns

## Connection to Supervised Learning

### Unsupervised as Preprocessing
```python
# Use clustering for feature engineering
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

# Create cluster features
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# One-hot encode cluster membership
ohe = OneHotEncoder()
cluster_features = ohe.fit_transform(cluster_labels.reshape(-1, 1))

# Add to original features
X_with_clusters = np.concatenate([X_scaled, cluster_features.toarray()], axis=1)
```

### Semi-Supervised Learning
- **Pseudo-labeling**: Use clustering to create labels
- **Self-training**: Use confident predictions as labels
- **Co-training**: Train on different feature views

## Best Practices

### Data Preparation
1. **Scale features**: Essential for distance-based methods
2. **Handle outliers**: Robust preprocessing
3. **Choose appropriate distance**: Based on data characteristics
4. **Consider dimensionality**: Reduce if too high

### Algorithm Selection
1. **Understand data structure**: Linear vs non-linear relationships
2. **Consider computational constraints**: Time and memory limits
3. **Evaluate multiple approaches**: Compare different algorithms
4. **Use domain knowledge**: Incorporate expert insights

### Validation Strategy
1. **Internal metrics**: Silhouette, Calinski-Harabasz
2. **Stability assessment**: Consistency across random seeds
3. **Visual inspection**: Plot clusters and reduced dimensions
4. **Business validation**: Check if results make sense

## Next Steps

This lecture establishes the foundation for unsupervised learning. The next lecture (15.2) will dive into K-means and hierarchical clustering, the most fundamental partitioning and hierarchical clustering methods.