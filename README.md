# Student Placement Prediction using Logistic Regression

This project predicts student placement outcomes based on CGPA and IQ scores using machine learning techniques, specifically logistic regression.

## Overview

The project analyzes student data to determine the relationship between academic performance (CGPA), intelligence quotient (IQ), and placement success. It uses logistic regression to create a binary classification model that can predict whether a student will be placed or not.

## Dataset

The dataset (`placement.csv`) contains student information with the following features:
- **CGPA**: Cumulative Grade Point Average
- **IQ**: Intelligence Quotient score
- **Placement**: Binary target variable (0 = Not Placed, 1 = Placed)

## Key Findings

Based on initial data exploration:
- Students with CGPA < 6: Generally not placed
- Students with CGPA > 6: Generally placed
- The model uses both CGPA and IQ to make more accurate predictions

## Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
```

## Installation

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn mlxtend
```

## Code Structure

### 1. Data Loading and Preprocessing
```python
df = pd.read_csv("/content/placement.csv")
df = df.drop(columns=['Unnamed: 0'])
```

### 2. Exploratory Data Analysis
- Scatter plots showing relationship between CGPA and placement
- Colored scatter plot showing CGPA vs IQ with placement outcomes
- 
<img width="567" height="438" alt="image" src="https://github.com/user-attachments/assets/1d4a9f06-d68d-498c-be9a-dadf2c8b1efb" />

### 3. Data Preparation
- Feature selection (CGPA and IQ as independent variables)
- Target variable selection (Placement as dependent variable)
- Train-test split (90% training, 10% testing)

### 4. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```

### 5. Model Training
```python
model = LogisticRegression()
model.fit(x_train, y_train)
```

### 6. Model Evaluation
- Predictions on test set
- Accuracy score calculation
- Decision boundary visualization

## Visualizations

The project includes several visualizations:

1. **CGPA vs Placement Scatter Plot**: Shows the basic relationship between CGPA and placement outcomes
2. **CGPA vs IQ Colored by Placement**: 2D visualization showing how both features relate to placement
3. **Decision Boundary Plot**: Shows the logistic regression model's decision boundary in the standardized feature space

## Model Performance

The model's performance is evaluated using accuracy score on the test set. The decision boundary visualization helps understand how the model separates the two classes (placed vs not placed).

## Usage

1. Ensure your dataset is in the correct path (`/content/placement.csv`)
2. Run the script to:
   - Load and explore the data
   - Train the logistic regression model
   - Evaluate model performance
   - Visualize results and decision boundary

## Key Machine Learning Concepts Demonstrated

- **Binary Classification**: Predicting placement (yes/no)
- **Feature Scaling**: Standardizing features for better model performance
- **Train-Test Split**: Proper evaluation methodology
- **Logistic Regression**: Linear model for binary classification
- **Decision Boundary Visualization**: Understanding model behavior

## Future Improvements

1. **Cross-Validation**: Use k-fold cross-validation for more robust evaluation
2. **Feature Engineering**: Create additional features or polynomial terms
3. **Model Comparison**: Try other algorithms (SVM, Random Forest, etc.)
4. **Hyperparameter Tuning**: Optimize model parameters
5. **More Metrics**: Add precision, recall, F1-score, and confusion matrix
   <img width="689" height="547" alt="image" src="https://github.com/user-attachments/assets/fd4c1b1d-b01f-4fab-b4f6-4f3c16be1a71" />


## File Structure

```
├── placement.csv          # Dataset
├── placement_analysis.py  # Main analysis script
└── README.md             # This file
```

## Notes

- The random state is not set in train_test_split, so results may vary between runs
- Standard scaling is applied to improve logistic regression performance
- The model assumes a linear decision boundary between classes

## Contributing

Feel free to contribute by:
- Adding more evaluation metrics
- Implementing additional visualization techniques
- Trying different machine learning algorithms
- Improving data preprocessing steps
