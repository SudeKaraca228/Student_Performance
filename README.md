# Student Final Grade Prediction (G3) - Machine Learning Project

In this project, we aimed to predict students' final grades (G3) based on several features such as previous grades, study time, and family background. The dataset was obtained from the UCI Machine Learning Repository.

## Dataset Used
- [UCI Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/student+performance)
- File: `student-mat.csv`

## Methods Used
- Random Forest Regressor
- Linear Regression

## Steps
1. Reading and preprocessing the data
2. Converting categorical features to numerical
3. Correlation analysis and selecting important features
4. Model training and evaluation on test data
5. Comparison of methods

## Features Used
- G1, G2, studytime, failures, Medu, Fedu, Dalc, Walc

## Results

| Method               | R2 Score | Mean Squared Error (MSE) |
|----------------------|----------|---------------------------|
| Random Forest         | 0.85     | 2.10                      |
| Linear Regression     | 0.72     | 3.45                      |

## Requirements

```bash
pip install -r requirements.txt
