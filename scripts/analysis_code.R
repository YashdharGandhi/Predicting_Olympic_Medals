## ----setup, include=FALSE-------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)



## ----load_data------------------------------------------------------------------------------
library(ggplot2)
#Load the dataset
olympic_data <- read.csv("C:/Users/yashd/Downloads/medal_pop_gdp_data_statlearn.csv")
str(olympic_data)
head(olympic_data)

# Check for missing values
missing_values <- sum(is.na(olympic_data))
print(paste("Number of missing values in the dataset:", missing_values))



## ----log_transform_GDP----------------------------------------------------------------------
# Log-transform the GDP variable (transform the GDP variable to address skewness.)
olympic_data$GDP_log <- log(olympic_data$GDP)
head(olympic_data)

ggplot(data = olympic_data, aes(x = GDP_log)) +
  geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Log-Transformed GDP",
       x = "Log(GPD)",
       y = "Frequency") +
  theme_minimal()



## ----population_distribution, fig.width=8, fig.height=6-------------------------------------
library(ggplot2)

ggplot(data = olympic_data, aes(x = Population)) +
  geom_histogram(binwidth = 1e6, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Population",
       x = "Population (in millions)",
       y = "Frequency") +
  theme_minimal()


## ----log_transform_population---------------------------------------------------------------
library(ggplot2)
# Log-transform the population variable
olympic_data$Population_log <- log(olympic_data$Population)

ggplot(data = olympic_data, aes(x = Population_log)) +
  geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Log-Transformed Population",
       x = "Log(Population)",
       y = "Frequency") +
  theme_minimal()


## ----boxplot_population, echo=FALSE---------------------------------------------------------
ggplot(data = olympic_data, aes(y = Population_log)) +
  geom_boxplot(fill = "skyblue", color = "black") +
  labs(title = "Box Plot of Log-Transformed Population",
       y = "Log(Population)") +
  theme_minimal()


## ----boxplot_gdp, echo=FALSE----------------------------------------------------------------
ggplot(data = olympic_data, aes(y = GDP_log)) +
  geom_boxplot(fill = "skyblue", color = "black") +
  labs(title = "Box Plot of Log-Transformed GDP",
       y = "Log(GDP)") +
  theme_minimal()


## ----scatterplot_pOPOLATION_vs_GDP, echo=FALSE----------------------------------------------
ggplot(data = olympic_data, aes(x = GDP_log, y = Population_log)) +
  geom_point(color = "skyblue") +
  labs(title = "Scatter Plot of Log-Transformed Population vs. Log-Transformed GDP",
       x = "Log(GDP)",
       y = "Log(Population)") +
  theme_minimal()


## ----transformation_medal_counts, echo=FALSE------------------------------------------------
# Apply logarithmic transformation to medal counts
olympic_data$Medal2012_log <- log(olympic_data$Medal2012)
olympic_data$Medal2008_log <- log(olympic_data$Medal2008)
olympic_data$Medal2016_log <- log(olympic_data$Medal2016)

ggplot(data = olympic_data, aes(x = Medal2008_log)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 20) +
  labs(title = "Histogram of Transformed Medal Counts (2008 Olympics)",
       x = "Log(Medal Counts)",
       y = "Frequency") +
  theme_minimal()


ggplot(data = olympic_data, aes(x = Medal2012_log)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 20) +
  labs(title = "Histogram of Transformed Medal Counts (2012 Olympics)",
       x = "Log(Medal Counts)",
       y = "Frequency") +
  theme_minimal()

ggplot(data = olympic_data, aes(x = Medal2016_log)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 20) +
  labs(title = "Histogram of Transformed Medal Counts (2016 Olympics)",
       x = "Log(Medal Counts)",
       y = "Frequency") +
  theme_minimal()


## ----transformation_medal_counts_sqrt, echo=FALSE-------------------------------------------
# Apply square root transformation to medal counts
olympic_data$Medal2008_sqrt <- sqrt(olympic_data$Medal2008)
olympic_data$Medal2012_sqrt <- sqrt(olympic_data$Medal2012)
olympic_data$Medal2016_sqrt <- sqrt(olympic_data$Medal2016)

ggplot(data = olympic_data, aes(x = Medal2008_sqrt)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 20) +
  labs(title = "Histogram of Transformed Medal Counts (2008 Olympics)",
       x = "Square Root of Medal Counts",
       y = "Frequency") +
  theme_minimal()


ggplot(data = olympic_data, aes(x = Medal2012_sqrt)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 20) +
  labs(title = "Histogram of Transformed Medal Counts (2012 Olympics)",
       x = "Square Root of Medal Counts",
       y = "Frequency") +
  theme_minimal()

ggplot(data = olympic_data, aes(x = Medal2016_sqrt)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 20) +
  labs(title = "Histogram of Transformed Medal Counts (2016 Olympics)",
       x = "Square Root of Medal Counts",
       y = "Frequency") +
  theme_minimal()



## ----data_analysis_task1, echo=FALSE--------------------------------------------------------
ggplot(data = olympic_data, aes(x = Population, y = Medal2008)) +
  geom_point(color = "skyblue") +
  labs(title = "Scatter Plot of Population vs. Medal Counts (2008 Olympics)",
       x = "Population",
       y = "Medal Counts") +
  theme_minimal()

ggplot(data = olympic_data, aes(x = GDP, y = Medal2008)) +
  geom_point(color = "skyblue") +
  labs(title = "Scatter Plot of GDP vs. Medal Counts (2008 Olympics)",
       x = "GDP",
       y = "Medal Counts") +
  theme_minimal()

ggplot(data = olympic_data, aes(x = Population, y = Medal2012)) +
  geom_point(color = "skyblue") +
  labs(title = "Scatter Plot of Population vs. Medal Counts (2012 Olympics)",
       x = "Population",
       y = "Medal Counts") +
  theme_minimal()

ggplot(data = olympic_data, aes(x = GDP, y = Medal2012)) +
  geom_point(color = "skyblue") +
  labs(title = "Scatter Plot of GDP vs. Medal Counts (2012 Olympics)",
       x = "GDP",
       y = "Medal Counts") +
  theme_minimal()

ggplot(data = olympic_data, aes(x = Population, y = Medal2016)) +
  geom_point(color = "skyblue") +
  labs(title = "Scatter Plot of Population vs. Medal Counts (2016 Olympics)",
       x = "Population",
       y = "Medal Counts") +
  theme_minimal()

ggplot(data = olympic_data, aes(x = GDP, y = Medal2016)) +
  geom_point(color = "skyblue") +
  labs(title = "Scatter Plot of GDP vs. Medal Counts (2016 Olympics)",
       x = "GDP",
       y = "Medal Counts") +
  theme_minimal()


## ----model_building_task1, echo=FALSE-------------------------------------------------------
# Fit linear regression model for 2012 Olympics
model_2012 <- lm(Medal2012 ~ Population + GDP, data = olympic_data)
summary(model_2012)


## ----data_analysis_task2, echo=FALSE--------------------------------------------------------
ggplot(data = olympic_data, aes(x = Population_log, y = Medal2008_log)) +
  geom_point(color = "skyblue") +
  labs(title = "Scatter Plot of Population vs. Medal Counts (2008 Olympics)",
       x = "Population",
       y = "Medal Counts") +
  theme_minimal()

ggplot(data = olympic_data, aes(x = GDP_log, y = Medal2008_log)) +
  geom_point(color = "skyblue") +
  labs(title = "Scatter Plot of GDP vs. Medal Counts (2008 Olympics)",
       x = "GDP",
       y = "Medal Counts") +
  theme_minimal()

ggplot(data = olympic_data, aes(x = Population_log, y = Medal2012_log)) +
  geom_point(color = "skyblue") +
  labs(title = "Scatter Plot of Population vs. Medal Counts (2012 Olympics)",
       x = "Population",
       y = "Medal Counts") +
  theme_minimal()

ggplot(data = olympic_data, aes(x = GDP_log, y = Medal2012_log)) +
  geom_point(color = "skyblue") +
  labs(title = "Scatter Plot of GDP vs. Medal Counts (2012 Olympics)",
       x = "GDP",
       y = "Medal Counts") +
  theme_minimal()

ggplot(data = olympic_data, aes(x = Population_log, y = Medal2016_log)) +
  geom_point(color = "skyblue") +
  labs(title = "Scatter Plot of Population vs. Medal Counts (2016 Olympics)",
       x = "Population",
       y = "Medal Counts") +
  theme_minimal()

ggplot(data = olympic_data, aes(x = GDP_log, y = Medal2016_log)) +
  geom_point(color = "skyblue") +
  labs(title = "Scatter Plot of GDP vs. Medal Counts (2016 Olympics)",
       x = "GDP",
       y = "Medal Counts") +
  theme_minimal()


## ----model_building_task2, echo=FALSE-------------------------------------------------------
# Fit linear regression model for log-transformed medal counts in 2012 Olympics
model_2012_log <- lm(Medal2012_log ~ Population + GDP, data = olympic_data)
summary(model_2012_log)


## ----best_model_building_task3, echo=FALSE--------------------------------------------------
# Function to fit LMS regression
fit_lms_regression <- function(X, y, num_iterations = 100, subset_size = 10) {
  best_model <- NULL
  best_residuals <- Inf
  
  for (i in 1:num_iterations) {
    indices <- sample(1:nrow(X), size = subset_size, replace = FALSE)
    X_subset <- X[indices, ]
    y_subset <- y[indices]
    
    model <- lm(y_subset ~ X_subset)
    
    residuals <- residuals(model)
    
    median_squared_residuals <- median(residuals^2)
    
    if (median_squared_residuals < best_residuals) {
      best_model <- model
      best_residuals <- median_squared_residuals
    }
  }
  
  return(best_model)
}


X <- model.matrix(~ Population + GDP + I(Population^2)+ I(GDP^2), data = olympic_data)
y <- olympic_data$Medal2012

best_model <- fit_lms_regression(X, y)
summary(best_model)

residuals <- residuals(best_model)
par(mfrow=c(1,3))
qqnorm(residuals)
qqline(residuals)
hist(residuals, breaks = 20, main = "Histogram of Residuals", xlab = "Residuals")
plot(density(residuals), main = "Density Plot of Residuals", xlab = "Residuals")



## ----AIC_CALCULATION, echo=FALSE------------------------------------------------------------
model_task1 <- lm(Medal2012 ~ Population, data = olympic_data)
model_task2 <- lm(Medal2012_log ~ Population + GDP, data = olympic_data)
X_model3 <- model.matrix(~ Population + GDP + I(Population^2) + I(GDP^2), data = olympic_data)
y_model3 <- olympic_data$Medal2012
model_task3 <- fit_lms_regression(X_model3, y_model3)

AIC_task1 <- AIC(model_task1)
AIC_task2 <- AIC(model_task2)
AIC_task3 <- AIC(model_task3)

summary_task1 <- summary(model_task1)
summary_task2 <- summary(model_task2)
summary_task3 <- summary(model_task3)

print("Summary for Model 1:")
summary_task1
print("Summary for Model 2:")
summary_task2
print("Summary for Model 3:")
summary_task3

best_model <- NULL
lowest_AIC <- Inf
for (model in list(model_task1, model_task2, model_task3)) {
  AIC_value <- AIC(model)
  if (AIC_value < lowest_AIC) {
    lowest_AIC <- AIC_value
    best_model <- model
  }
}

print("Summary for Best Model:")
summary(best_model)
AIC_task1 <- AIC(model_task1)
AIC_task2 <- AIC(model_task2)
AIC_task3 <- AIC(model_task3)

print("AIC for Model 1:")
AIC_task1
print("AIC for Model 2:")
AIC_task2
print("AIC for Model 3:")
AIC_task3




## ----Probablity_Estimation------------------------------------------------------------------
library(MASS) 
intercept <- 4.612e+00
population_coef <- -3.603e-08
gdp_coef <- 1.379e-02
population_squared_coef <- 2.355e-17
gdp_squared_coef <- -4.408e-07

uk_population <- 67910685  # UK population as of April 9, 2024
uk_gdp <- uk_gdp_2024 <- 2342865  # in million GBP  

generate_predicted_medal_counts <- function(num_simulations, intercept, population_coef, gdp_coef, population_squared_coef, gdp_squared_coef, uk_population, uk_gdp) {
  predicted_medal_counts <- numeric(num_simulations)
  
  samples <- mvrnorm(num_simulations, mu = c(intercept, population_coef * uk_population, gdp_coef * uk_gdp, population_squared_coef * (uk_population^2), gdp_squared_coef * (uk_gdp^2)), Sigma = diag(5))
  
  for (i in 1:num_simulations) {
    predicted_medal_counts[i] <- samples[i, 1]
  }
  
  return(predicted_medal_counts)
}

num_simulations <- 10000

predicted_medal_counts <- generate_predicted_medal_counts(num_simulations, intercept, population_coef, gdp_coef, population_squared_coef, gdp_squared_coef, uk_population, uk_gdp_2024)

probability_at_least_one_medal <- sum(predicted_medal_counts > 0) / num_simulations

print(paste("Probability of winning at least one medal in 2024:", probability_at_least_one_medal))

