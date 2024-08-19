
# Coupon Analysis Project

## Overview

This project involves analyzing a dataset containing information about various coupon types and the behavior of drivers in accepting these coupons. The goal is to identify patterns and correlations that influence whether a driver is likely to accept a coupon.

## Files in the Project

- **`prompt.ipynb`**: The main jupyter notebook containing the code for data analysis.
- **Datasets and Images**: Various datasets and images generated during the analysis (referenced in the code and outputs).

## Dependencies

To run this project, you will need the following Python libraries:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib`: For creating static, animated, and interactive visualizations.
- `seaborn`: For statistical data visualization.
- `plotly`: For creating interactive plots.

Install the dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn plotly
```

## How to Run the Code

1. Ensure you have all the required libraries installed.
2. Place the `prompt.ipynb` script and the datasets in the same directory.
3. Run the `prompt.ipynb` jupyter notebook

4. The notebook will execute various analyses and generate plots that provide insights into coupon acceptance patterns among drivers.

## Analysis Overview

### 1. Data Cleaning and Preparation

- **Handling Missing Values**: Missing values in columns like `CoffeeHouse`, `Restaurant20To50`, `CarryAway`, `RestaurantLessThan20`, and `Bar` are filled using the mode (most frequent value) of each column.
- **Dropping Duplicates**: Duplicate records in the dataset are removed to avoid skewing the analysis.

### 2. Exploratory Data Analysis (EDA)

- **Coupon Distribution**: The distribution of different coupon types is visualized using a bar plot.
- **Acceptance Rate Analysis**: The acceptance rate of each coupon type is calculated and visualized.
- **Categorical Analysis**: Analysis based on various categories like bar visit frequency, age, income, occupation, and presence of passengers.

### 3. Specific Hypotheses Testing

The script tests several hypotheses regarding the acceptance of bar coupons, including:

- The influence of bar visit frequency on coupon acceptance.
- The effect of age, passenger type, and occupation on acceptance rates.
- Complex criteria, combining multiple factors to analyze coupon acceptance behavior.

### 4. Visualization

- **Bar Charts**: Used extensively to compare acceptance rates across different driver categories.
- **Histogram**: To visualize the distribution of continuous variables like temperature.

## Results and Hypotheses

The analysis indicates that drivers who are younger, visit bars frequently, have non-rural occupations, or are in certain income brackets are more likely to accept bar coupons. These findings can be used to target specific demographics with promotional offers.

## Conclusion

This project provides insights into the factors influencing coupon acceptance among drivers, with a focus on bar-related promotions. The results can be utilized to tailor marketing strategies to the most receptive segments of the population.

