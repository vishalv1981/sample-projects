# Used Car Price Prediction

This project aims to identify the key drivers of used car prices using regression analysis. The goal is to provide meaningful insights to used car dealers, helping them fine-tune their inventory strategies.

## Table of Contents
- [Project Overview](#project-overview)
- [Steps and Libraries](#steps-and-libraries)
- [Installation](#installation)
- [Evaluation](#evaluation)
- [Deployment](#deployment)

## Project Overview

This project involves the following steps:
1. **Data Cleaning**: Handle missing data and prepare the dataset for analysis.
2. **Feature Engineering**: Encode categorical variables using target encoding to reduce memory usage.
3. **Regression Analysis**: Perform linear regression to identify the key drivers of used car prices.
4. **Model Evaluation**: Assess the performance of the model using metrics like Mean Absolute Error (MAE) and R-squared (R²).
5. **Deployment**: Summarize findings and prepare a report for used car dealers.

## Steps and Libraries

### 1. Data Cleaning
- Handle missing values in the dataset by dropping columns with a high percentage of missing data or imputing values.
- Libraries used: `pandas`

### 2. Feature Engineering
- Convert categorical variables to numerical values using target encoding to avoid memory issues.
- Libraries used: `pandas`, `category_encoders`

### 3. Regression Analysis
- Split the data into training and testing sets.
- Train a linear regression model and evaluate its performance.
- Libraries used: `scikit-learn`

### 4. Model Evaluation
- Evaluate the model using Mean Absolute Error (MAE) and R-squared (R²) metrics.
- Libraries used: `scikit-learn`

### 5. Deployment
- Compile findings into a report suitable for used car dealers, focusing on the key drivers of used car prices.

## Installation

To run this project, you need to install the necessary libraries. You can do so by running the following commands:

```bash
pip install pandas scikit-learn category_encoders

