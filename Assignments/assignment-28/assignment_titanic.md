# Module 14 Assignment: Titanic Survival Classification

## Overview
In this assignment, you will apply machine learning classification techniques to predict passenger survival on the Titanic using the famous Titanic dataset. This assignment builds on the concepts covered in Module 14: Machine Learning Classification.

## Learning Objectives
- Apply supervised learning classification algorithms
- Implement data preprocessing and feature engineering
- Evaluate model performance using appropriate metrics
- Compare different classification algorithms
- Handle class imbalance and model validation

## Dataset
The Titanic dataset contains information about passengers who were on board the RMS Titanic when it sank in 1912. Your task is to predict whether a passenger survived based on various features.

**Dataset Source**: [Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

## Assignment Tasks

### Part 1: Data Exploration and Preprocessing (25 points)

1. **Load and explore the data**
   - Load the Titanic dataset
   - Examine the data structure, missing values, and basic statistics
   - Visualize distributions of key features

2. **Data preprocessing**
   - Handle missing values appropriately
   - Convert categorical variables to numerical
   - Create new features (feature engineering)
   - Handle outliers if any

3. **Data visualization**
   - Create visualizations showing relationships between features and survival
   - Analyze correlations between variables

### Part 2: Classification Model Implementation (50 points)

1. **Implement multiple classification algorithms**
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)

2. **Model training and validation**
   - Split data into training and testing sets (80/20 split)
   - Use cross-validation for robust evaluation
   - Implement proper scaling where needed

3. **Hyperparameter tuning**
   - Use GridSearchCV or RandomizedSearchCV for at least 2 algorithms
   - Optimize key hyperparameters for each model

### Part 3: Model Evaluation and Comparison (25 points)

1. **Performance metrics**
   - Calculate accuracy, precision, recall, F1-score for each model
   - Generate confusion matrices
   - Plot ROC curves and calculate AUC scores

2. **Model comparison**
   - Compare all models using a comprehensive metrics table
   - Identify the best performing model
   - Analyze trade-offs between different metrics

3. **Feature importance analysis**
   - Analyze which features are most important for prediction
   - Provide insights about what factors contributed most to survival

## Technical Requirements

### Environment Setup
```bash
cd "Data-Science-B/14_module_machine_learning_classification"
uv sync
source .venv/bin/activate
jupyter notebook assignment_titanic.ipynb
```

### Required Libraries
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- jupyter

### Code Structure
Your solution should include:
- Data loading and exploration
- Data preprocessing pipeline
- Model training and evaluation functions
- Visualization of results
- Clear documentation and comments

## Evaluation Criteria

### Code Quality (30%)
- Clean, well-documented code
- Proper error handling
- Efficient implementation
- Good coding practices

### Data Preprocessing (25%)
- Appropriate handling of missing values
- Effective feature engineering
- Proper data transformations
- Outlier handling

### Model Implementation (25%)
- Correct implementation of algorithms
- Proper train/test splits
- Cross-validation usage
- Hyperparameter tuning

### Analysis and Insights (20%)
- Comprehensive model evaluation
- Meaningful visualizations
- Clear interpretation of results
- Actionable insights

## Submission Requirements

1. **Jupyter Notebook** (`assignment_titanic.ipynb`) containing:
   - All code with proper documentation
   - Visualizations and analysis
   - Conclusions and insights

2. **Report** (PDF or Markdown) including:
   - Methodology explanation
   - Results summary
   - Model comparison analysis
   - Key findings and insights

## Advanced Challenges (Optional - Bonus Points)

1. **Handle class imbalance** using SMOTE or other techniques
2. **Implement ensemble methods** combining multiple models
3. **Feature selection** using statistical tests or model-based methods
4. **Model interpretation** using SHAP or LIME
5. **Pipeline implementation** for production-ready code

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Titanic Dataset Description](https://www.kaggle.com/c/titanic/data)
- [Machine Learning Classification Guide](https://developers.google.com/machine-learning/crash-course/classification)

## Deadline
[Insert deadline here]

## Academic Integrity
This is an individual assignment. All work must be your own. Plagiarism will result in a zero grade and potential academic penalties.

---

**Instructor**: Dennis Omboga Mongare
**Course**: Data Science B - Machine Learning Classification
**Assignment Weight**: 25% of Module 14 Grade