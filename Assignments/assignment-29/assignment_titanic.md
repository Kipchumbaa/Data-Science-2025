# Module 15 Assignment: Titanic Passenger Clustering and Survival Analysis

## Overview
In this assignment, you will apply unsupervised machine learning techniques (clustering and dimensionality reduction) to analyze patterns in the Titanic dataset. You'll discover hidden structures in passenger data and explore how these patterns relate to survival outcomes.

## Learning Objectives
- Apply unsupervised learning algorithms (K-means, hierarchical clustering, DBSCAN)
- Implement dimensionality reduction techniques (PCA, t-SNE, UMAP)
- Interpret clustering results and identify meaningful patterns
- Visualize high-dimensional data in lower dimensions
- Connect unsupervised learning insights with supervised learning outcomes

## Dataset
The Titanic dataset contains information about passengers who were on board the RMS Titanic. You'll analyze passenger characteristics to identify natural groupings and survival patterns.

**Dataset Source**: [Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

## Assignment Tasks

### Part 1: Data Preparation and Exploration (25 points)

1. **Data Loading and Preprocessing**
   - Load the Titanic dataset
   - Handle missing values appropriately
   - Encode categorical variables for clustering
   - Scale/normalize features for distance-based algorithms

2. **Feature Engineering for Clustering**
   - Create meaningful features for passenger segmentation
   - Handle mixed data types (numerical + categorical)
   - Consider domain knowledge for feature creation

3. **Exploratory Data Analysis**
   - Visualize distributions and correlations
   - Identify potential clusters through visualization
   - Analyze feature relationships

### Part 2: Clustering Analysis (40 points)

1. **K-means Clustering**
   - Implement K-means with different k values
   - Use elbow method and silhouette analysis to find optimal k
   - Analyze cluster characteristics and interpret results

2. **Hierarchical Clustering**
   - Implement agglomerative clustering
   - Create dendrograms for visualization
   - Compare different linkage methods

3. **Density-Based Clustering (DBSCAN)**
   - Apply DBSCAN for density-based clustering
   - Tune epsilon and min_samples parameters
   - Handle noise points appropriately

4. **Cluster Validation**
   - Use internal validation metrics (silhouette score, Calinski-Harabasz index)
   - Compare clustering algorithms
   - Assess cluster stability

### Part 3: Dimensionality Reduction (25 points)

1. **Principal Component Analysis (PCA)**
   - Apply PCA for dimensionality reduction
   - Analyze explained variance ratios
   - Visualize principal components

2. **Advanced Dimensionality Reduction**
   - Implement t-SNE for visualization
   - Apply UMAP for manifold learning
   - Compare different techniques

3. **Integration with Clustering**
   - Use dimensionality reduction for cluster visualization
   - Analyze clusters in reduced dimensional space
   - Interpret results in the context of original features

### Part 4: Survival Pattern Analysis (10 points)

1. **Connect Unsupervised to Supervised Learning**
   - Analyze survival rates within clusters
   - Identify high-risk and low-risk passenger groups
   - Compare clustering-based insights with direct classification

2. **Business Insights**
   - Provide actionable insights from clustering results
   - Suggest targeted interventions based on cluster characteristics

## Technical Requirements

### Environment Setup
```bash
cd "Data-Science-B/15_module_machine_learning_clustering_and_dimensionality_Reduction"
uv sync
source .venv/bin/activate
jupyter notebook assignment_titanic.ipynb
```

### Required Libraries
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- umap-learn (optional, for UMAP)
- hdbscan (optional, for advanced clustering)

### Evaluation Metrics
- **Silhouette Score**: Measures cluster cohesion and separation
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
- **Davies-Bouldin Index**: Average similarity of each cluster with its most similar cluster

## Code Structure Requirements

Your solution should include:

```python
# Data preprocessing
def preprocess_data(df):
    # Handle missing values, encoding, scaling
    pass

# Clustering functions
def perform_kmeans_clustering(X, k_range):
    # Implement K-means with evaluation
    pass

def perform_hierarchical_clustering(X, linkage_methods):
    # Implement hierarchical clustering
    pass

def perform_dbscan_clustering(X, eps_range, min_samples_range):
    # Implement DBSCAN with parameter tuning
    pass

# Dimensionality reduction
def apply_pca(X, n_components_range):
    # PCA implementation with analysis
    pass

def apply_tsne_umap(X):
    # Advanced dimensionality reduction
    pass

# Visualization functions
def plot_clusters(X, labels, method_name):
    # Cluster visualization
    pass

def plot_dimensionality_reduction(X, labels, method_name):
    # Low-dimensional visualization
    pass
```

## Evaluation Criteria

### Technical Implementation (40%)
- Correct implementation of clustering algorithms
- Proper preprocessing and feature engineering
- Appropriate parameter tuning and validation
- Efficient code execution

### Analysis Quality (35%)
- Thorough exploration of different algorithms
- Meaningful interpretation of results
- Clear visualization of clusters and patterns
- Connection between unsupervised and supervised insights

### Documentation and Presentation (25%)
- Well-documented code with comments
- Clear explanations of methodology and results
- Professional visualizations
- Actionable insights and recommendations

## Submission Requirements

1. **Jupyter Notebook** (`assignment_titanic.ipynb`) containing:
   - Complete analysis workflow
   - All visualizations and results
   - Interpretation of findings
   - Code documentation

2. **Analysis Report** (PDF/Markdown) including:
   - Methodology overview
   - Results for each clustering algorithm
   - Dimensionality reduction analysis
   - Survival pattern insights
   - Recommendations

## Advanced Challenges (Optional - Bonus Points)

1. **Ensemble Clustering**
   - Combine multiple clustering algorithms
   - Implement consensus clustering

2. **Deep Clustering**
   - Use autoencoders for feature learning
   - Implement deep embedded clustering

3. **Temporal Clustering**
   - Analyze passenger boarding patterns
   - Time-based cluster analysis

4. **Interactive Visualization**
   - Create interactive cluster exploration
   - Web-based visualization dashboard

## Resources

- [Scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Clustering Evaluation Metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)

## Dataset Features for Clustering

Consider these features for passenger segmentation:
- **Demographic**: Age, Sex, Pclass
- **Family**: SibSp, Parch, Family_Size
- **Economic**: Fare, Fare_Category
- **Travel**: Embarked, Cabin availability
- **Social**: Title, Is_Alone

## Expected Outcomes

By the end of this assignment, you should be able to:
- Identify natural passenger groupings
- Understand survival patterns within clusters
- Visualize complex data relationships
- Apply unsupervised learning to real-world problems

## Deadline
[Insert deadline here]

---

**Instructor**: Dennis Omboga Mongare
**Course**: Data Science B - Unsupervised Learning
**Assignment Weight**: 25% of Module 15 Grade