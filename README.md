# PROJECT 2: DATA CLASSIFICATION USING AI

## 1. PROJECT OVERVIEW

IrisClassifier is a supervised machine learning project developed in Python using Scikit-learn. It demonstrates the complete workflow of teaching a machine to recognize patterns and classify data using the famous Iris Benchmark Dataset.

The model combines **StandardScaler**, **K-Nearest Neighbors (KNN)**, and **GridSearchCV** to create an optimized classification pipeline capable of predicting flower species based on their physical measurements.

This project satisfies the requirements of **Project 2: Data Classification Using AI**, demonstrating the implementation of a validated supervised learning system.

---

## 2. FEATURES

### Dataset Loading

* Uses Scikit-learn's built-in Iris dataset.
* Contains:

  * 150 samples
  * 4 numerical features
  * 3 target classes

### Data Splitting

* Uses `train_test_split()`.
* 80% training data and 20% testing data.
* Applies `stratify=y` to maintain class balance.

### Feature Scaling

* Uses `StandardScaler()`.
* Ensures all features contribute equally during classification.
* Prevents dimensional bias.

### Classification Algorithm

* Uses `KNeighborsClassifier()`.
* Classifies samples based on similarity to neighboring data points.

### Hyperparameter Tuning

* Uses `GridSearchCV()`.
* Automatically selects the optimal value of K from 1 to 10.
* Reduces manual parameter selection.

### Model Validation

* Uses:

  * Accuracy Score
  * F1 Score
  * Classification Report
  * 5-Fold Cross Validation

---

## 3. HOW TO RUN

### Prerequisites

* Python 3.9 or higher installed

### Installation

Open terminal and run:

```bash
pip install -r requirements.txt
```

### Execute the Project

```bash
python main.py
```

### Output

The program displays:

* Best K value selected by GridSearchCV
* Cross-validation score
* Test accuracy
* F1 score
* Detailed classification report

---

## 4. PROJECT STRUCTURE

```text
data_classification_ai/
│
├── main.py            # Main machine learning pipeline
├── requirements.txt   # Required dependencies
└── README.md          # Project documentation
```

---

## 5. WORKING PRINCIPLE (IPO MODEL)

### Input

The Iris dataset is loaded using:

```python
load_iris()
```

The dataset contains flower measurements including:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

### Process

1. Split dataset into training and testing sets.
2. Apply StandardScaler for feature normalization.
3. Use GridSearchCV to find the optimal K value.
4. Train the KNN model using:

```python
model.fit()
```

### Output

Predictions are generated using:

```python
model.predict()
```

Performance is evaluated using:

```python
accuracy_score()
f1_score()
classification_report()
```

---

## 6. MACHINE LEARNING PIPELINE

### Step 1: Load Dataset

Import the Iris dataset from Scikit-learn.

### Step 2: Split Data

Separate data into training and testing subsets.

### Step 3: Scale Features

Normalize feature values using StandardScaler.

### Step 4: Hyperparameter Optimization

Use GridSearchCV to determine the best K value.

### Step 5: Train Model

Fit the KNN classifier on training data.

### Step 6: Make Predictions

Predict flower species on unseen test data.

### Step 7: Evaluate Performance

Measure accuracy and classification effectiveness.

---

## 7. KEY CONCEPTS DEMONSTRATED

### 1. Supervised Learning

The model learns from labeled examples and predicts target classes.

### 2. Feature Scaling

StandardScaler standardizes features to improve model performance.

### 3. Classification

KNN classifies data points based on nearest neighbors.

### 4. Hyperparameter Tuning

GridSearchCV automatically finds the optimal model configuration.

### 5. Cross Validation

5-Fold Cross Validation improves reliability and reduces overfitting risk.

### 6. Model Evaluation

Accuracy, F1 Score, and Classification Report provide performance insights.

---

## 8. EXPECTED OUTPUT

Example output may include:

```text
Best K Value: 5

Cross Validation Score: 0.96

Accuracy: 0.97

F1 Score: 0.97

Classification Report:
--------------------------------
Precision
Recall
F1-Score
Support
--------------------------------
```

Actual results may vary slightly depending on the train-test split.

---

## 9. LEARNING OUTCOMES

This project demonstrates:

* Data preprocessing
* Feature scaling
* Supervised machine learning
* Classification algorithms
* Hyperparameter optimization
* Cross-validation techniques
* Performance evaluation metrics
* End-to-end machine learning workflow

---

## 10. TECHNOLOGIES USED

* Python
* Scikit-learn
* NumPy
* StandardScaler
* K-Nearest Neighbors (KNN)
* GridSearchCV

---
