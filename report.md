# Project Report: Student Final Grade Prediction

## Objective
The goal of this project is to predict students’ final exam scores (G3) based on selected features from their academic and personal background.

## Dataset
- Source: UCI Machine Learning Repository
- File used: `student-mat.csv`

## Selected Features
We selected the following features based on correlation analysis:
- G1, G2 (previous grades)
- failures
- studytime
- Medu, Fedu (parental education)
- Dalc, Walc (alcohol consumption)

## Models Compared
- Linear Regression
- Random Forest Regressor

## Results

| Model              | R2 Score | Mean Squared Error |
|-------------------|----------|---------------------|
| Linear Regression | 0.72     | 3.45                |
| Random Forest     | 0.85     | 2.10                |

## Conclusion
Random Forest Regressor outperformed Linear Regression in both accuracy and error metrics, making it more suitable for this prediction task. However, Linear Regression is still useful as a simple, interpretable baseline model.
