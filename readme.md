# Piecewise_Spline Regression Models Using NumPy - Predictive Sports Analytics

## Overview
This project focuses on forecasting the points earned by a sports team based on their training performance, including factors like yoga sessions, laps completed, water intake, and weightlifting sessions. By applying machine learning techniques, this project aims to assist stakeholders in optimizing the team's training performance.

In this project, we explore Piecewise Regression and Spline Regression techniques to improve upon our previous results. Piecewise Regression divides the independent variable into intervals and fits different linear functions to each interval. Spline Regression combines polynomial regression functions into a piecewise continuous function.

---

## Aim
The main goal of this project is to predict a sports team's points using Piecewise and Spline Regression.

---

## Data Description
The dataset used in this project contains information about the points scored by sports teams based on various attributes.
 - NBA_Dataset_csv.csv

---

## Tech Stack
- Language: `Python`
- Libraries: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `sklearn`, `statsmodels`, `piecewise_regression`, `csaps`, `py-earth`, `mlfoundry`

---

## Approach
The project follows a structured approach, including the following steps:
1. Data Reading
2. Data Preprocessing
   - Outlier Removal
   - One-Hot Encoding
   - Imputing Missing Values
3. Model Building
   - Linear Regression
   - Polynomial Regression
   - Step Functions
   - Piecewise Regression
   - Basis Functions
   - Spline Regression
     - Univariate Model
     - Bivariate Model
     - Multivariate Adaptive Regression Splines (MARS)
4. Model Evaluation and Comparison
5. Experiment Tracking with ML Foundry

---

## Modular Code Overview

1. `input`: Contains the NBA dataset used in the project.
2. `lib`: A reference folder containing the original Jupyter notebook from the lectures.
3. `ml_pipeline`: Contains functions divided into different Python files, which are appropriately named. The `Engine.py` script calls these functions to execute the project's steps, train the model, and display the results.
4. `requirements.txt`: Lists all the required libraries along with their versions. You can install these libraries using the command `pip install -r requirements.txt`.

---

## Execution Instructions:

- Install requirements with "pip install -r requirements.txt"
- Run Engine.py to train models and get the results

---

## Concepts Explored

1. Step functions and the Heaviside function
2. The limitations of Polynomial Regression
3. Piecewise Regression using Python
4. Understanding Log-Likelihood
5. Adaptive Subdivision and Sequential Search
6. Davies Test
7. Basis Functions, their importance and Implementing in Python
8. Splines and how they overcome the limitations of Piecewise Regression
9. Degrees of Freedom
10. B-splines and Cubic Splines
11. Understanding Bernstein Polynomials and Weierstrass Transforms
12. Implementing Spline Regression in Python
13. What Multivariate Adaptive Regression Splines (MARS) is
14. Implementing MARS in Python
15. Tracking model experiments using ML Foundry


---











