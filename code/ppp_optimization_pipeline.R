# Script Name: ppp_optimization_pipeline.R
# Description: Vectorized Elastic Net PPP Optimization & Rolling-Window Engine

library(dplyr)

# 1. Automatic Feature Identification (Excluding ID and Target)
feature_cols <- setdiff(names(df), c("permno", "yyyymm", "RET"))
cat(sprintf("Successfully identified %d factor features for screening.\n", length(feature_cols)))

# 2. Intelligent Cross-Sectional Cleaning Function 
# (Handles missing values via median imputation, Winsorization, and Z-scoring)
process_feature <- function(x) {
  # 2.1 Median imputation
  m <- median(x, na.rm = TRUE)
  if(is.na(m)) m <- 0
  x[is.na(x)] <- m
  
  # 2.2 Winsorization (1% and 99% percentiles)
  q <- quantile(x, probs = c(0.01, 0.99), na.rm = TRUE)
  x[x < q[1]] <- q[1]
  x[x > q[2]] <- q[2]
  
  # 2.3 Z-score standardization
  s <- sd(x, na.rm = TRUE)
  if(is.na(s) || s == 0) {
    return(rep(0, length(x)))
  } else {
    return((x - mean(x, na.rm = TRUE)) / s)
  }
}

# 3. Vectorized Elastic Net Utility Function (Optimized for performance)
calculate_utility_matrix_fast <- function(theta, X, r, inv_N, time_id, gamma = 5, lambda = 0.0005, alpha = 0.5) {
  # Compute portfolio weights via matrix multiplication
  w <- inv_N * (1 + X %*% theta)
  asset_contrib <- w * r
  
  # Aggregate monthly portfolio returns using vectorized tapply
  Rp <- as.numeric(tapply(asset_contrib, time_id, sum))
  
  # Bankruptcy protection constraint
  if (any(Rp <= -1)) return(1e10)
  
  # Compute negative log utility
  u <- mean(log(1 + Rp))
  neg_utility <- -u
  
  # Elastic Net dual penalty term
  total_penalty <- lambda * (alpha * sum(abs(theta)) + (1 - alpha) * sum(theta^2))
  
  return(neg_utility + total_penalty)
}

# ==============================================================================
# Data Pipeline: Cleaning and Standardization
# ==============================================================================
cat("Running cross-sectional cleaning and feature standardization...\n")

df_clean <- df %>%
  filter(!is.na(RET), !is.na(yyyymm)) %>%
  arrange(yyyymm) %>%
  group_by(yyyymm) %>%
  mutate(
    time_id = cur_group_id(),
    inv_N = 1 / n()
  ) %>%
  mutate(across(all_of(feature_cols), process_feature, .names = "x_{.col}")) %>%
  ungroup()

standardized_features <- paste0("x_", feature_cols)

# ==============================================================================
# Rolling-Window Backtesting Engine
# ==============================================================================
total_months <- max(df_clean$time_id)
window_size <- 60  
num_loops <- total_months - window_size

oos_results <- data.frame(
  time_id = integer(),
  OOS_Return = numeric(),
  Benchmark_Return = numeric()
)

theta_history <- matrix(0, nrow = num_loops, ncol = length(feature_cols))
colnames(theta_history) <- feature_cols

cat(sprintf("Starting rolling-window backtest. Total iterations: %d...\n", num_loops))

for (i in 1:num_loops) {
  
  if(i %% 20 == 0) cat(sprintf("Progress: Completed %d / %d iterations...\n", i, num_loops))
  
  # 1. Training set extraction & matrix conversion
  train_data    <- df_clean %>% filter(time_id >= i & time_id < (i + window_size))
  X_train       <- as.matrix(train_data[, standardized_features])
  r_train       <- train_data$RET
  inv_N_train   <- train_data$inv_N
  time_id_train <- train_data$time_id
  
  # 2. Optimization for optimal Theta vector via BFGS
  opt_res <- optim(
    par = rep(0, length(feature_cols)), 
    fn = calculate_utility_matrix_fast, 
    X = X_train, r = r_train, inv_N = inv_N_train, time_id = time_id_train,
    gamma = 5,  
    lambda = 0.0005,  
    alpha = 0.5,       
    method = "BFGS"
  )
  
  theta_history[i, ] <- opt_res$par
  
  # 3. Out-of-Sample (OOS) testing (Month t + 60)
  test_data <- df_clean %>% filter(time_id == (i + window_size))
  X_test    <- as.matrix(test_data[, standardized_features])
  R_test    <- test_data$RET
  N_test    <- nrow(test_data)
  
  weights_oos <- (1/N_test) + (1/N_test) * (X_test %*% opt_res$par)
  portfolio_return_oos <- sum(weights_oos * R_test)
  benchmark_return     <- mean(R_test)
  
  oos_results <- rbind(oos_results, data.frame(
    time_id          = i + window_size,
    OOS_Return       = portfolio_return_oos,
    Benchmark_Return = benchmark_return
  ))
}

cat("Backtest and feature evaluation completed\n\n")

# ==============================================================================
# Performance Reporting & Feature Importance Extraction
# ==============================================================================
ann_return_oos   <- mean(oos_results$OOS_Return) * 12
ann_return_bench <- mean(oos_results$Benchmark_Return) * 12
ann_vol_oos      <- sd(oos_results$OOS_Return) * sqrt(12)
ann_vol_bench    <- sd(oos_results$Benchmark_Return) * sqrt(12)
sharpe_oos       <- ann_return_oos / ann_vol_oos
sharpe_bench     <- ann_return_bench / ann_vol_bench

cat("=== Performance Summary Report ===\n")
cat(sprintf("PPP Strategy      - Ann. Return: %.2f%%, Sharpe Ratio: %.3f\n", ann_return_oos*100, sharpe_oos))
cat(sprintf("Equal-Weight Bench - Ann. Return: %.2f%%, Sharpe Ratio: %.3f\n\n", ann_return_bench*100, sharpe_bench))

feature_importance <- data.frame(
  Feature = colnames(theta_history),
  Mean_Absolute_Theta = colMeans(abs(theta_history))
) %>%
  arrange(desc(Mean_Absolute_Theta))

cat("Top 10 Feature Importance Ranking (Mean Absolute Theta) \n")
print(head(feature_importance, 10))
