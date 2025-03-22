# Olympic Medal Forecasting Using Macroeconomic Indicators

## Overview
This project analyzes the relationship between a country's GDP, population, and Olympic medal count using various statistical models.

## Dataset
- Data from 71 countries (medal winners in the last 3 Olympics).
- Includes population, GDP, and medal counts.

## Tasks Performed
1. **Linear Regression:** Predict medals using GDP and population.
2. **Log-Transformation:** Evaluate impact of log-transformed outputs.
3. **Custom Regression Model:** Developed Least Median of Squares (LMS) model.
4. **Model Selection with AIC:** Identified the best model.
5. **Probability Estimation:** Predicted UK’s probability of winning a medal.

## Key Findings
- GDP is the strongest predictor of Olympic medal count.
- Population has minimal influence.
- LMS regression performed best (Adjusted R²: 0.9985).
- UK has a **100% probability** of winning a medal in 2024.

## Requirements
To run the scripts, install dependencies in R:
```r
install.packages(c("ggplot2", "MASS", "dplyr"))
