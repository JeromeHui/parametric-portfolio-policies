# Script Name: ppp_model_comparison_loop.R
# Description: Comparative Rolling-Window Evaluation Across Penalty Specifications (Ridge alpha=0.0, Elastic Net alpha=0.5, Lasso alpha=1.0, Unpenalized)

library(dplyr)

# ------------------------------------------------------------------------------
# 1. Utility Functions & Preprocessing
# ------------------------------------------------------------------------------

# Winsorization helper (1% and 99% tails)
winsorize <- function(x) {
  q <- quantile(x, probs = c(0.01, 0.99), na.rm = TRUE)
  x[x < q[1]] <- q[1]
  x[x > q[2]] <- q[2]
  return(x)
}

# Elastic Net BSV Utility Objective Function
calculate_utility_elastic_net <- function(theta, data, gamma = 5, lambda = 0.0001, alpha = 0.5) {
  
  # A. Portfolio return calculation across cross-sections
  monthly_returns <- data %>%
    group_by(year, month) %>%
    summarise(Rp = {
      X <- as.matrix(cbind(x_mom, x_gp, x_vol, x_skew))
      r <- RET
      N <- n()
      w <- (1/N) + (1/N) * (X %*% theta)
      sum(w * r)
    }, .groups = "drop")
  
  # Log utility calculation
  u <- mean(log(1 + monthly_returns$Rp))
  neg_utility <- -u
  
  # B. Regularization penalty term
  l1_penalty <- sum(abs(theta))
  l2_penalty <- sum(theta^2)
  total_penalty <- lambda * (alpha * l1_penalty + (1 - alpha) * l2_penalty)
  
  return(neg_utility + total_penalty)
}

# ------------------------------------------------------------------------------
# 2. Data Cleaning & Standardization Pipeline
# ------------------------------------------------------------------------------
df_clean <- df %>%
  filter(!is.na(RET), !is.na(Mom1m_p), !is.na(GP_p), !is.na(OptionVolume1_p), !is.na(skew_p)) %>%
  mutate(across(c(Mom1m_p, GP_p, OptionVolume1_p, skew_p, RET), winsorize)) %>%
  group_by(year, month) %>% 
  mutate(
    x_mom  = (Mom1m_p - mean(Mom1m_p)) / sd(Mom1m_p),
    x_gp   = (GP_p - mean(GP_p)) / sd(GP_p),
    x_vol  = (OptionVolume1_p - mean(OptionVolume1_p)) / sd(OptionVolume1_p),
    x_skew = (skew_p - mean(skew_p)) / sd(skew_p)
  ) %>%
  ungroup() %>%
  na.omit() %>%
  arrange(year, month) %>%
  group_by(year, month) %>%
  mutate(time_id = cur_group_id()) %>%
  ungroup()

# ------------------------------------------------------------------------------
# 3. Parameterized Backtest Wrapper Function
# ------------------------------------------------------------------------------
run_ppp_model <- function(alpha_val, lambda_val, model_label, df_data, window_size = 60) {
  
  total_months <- max(df_data$time_id)
  num_loops <- total_months - window_size
  
  oos_results <- data.frame(
    time_id = integer(),
    year = integer(),
    month = integer(),
    OOS_Return = numeric(),
    Benchmark_Return = numeric()
  )
  
  cat(sprintf("\n--- Executing Backtest for: %s (alpha = %.1f, lambda = %.4f) ---\n", 
              model_label, alpha_val, lambda_val))
  
  for (i in 1:num_loops) {
    
    # Training window extraction (60 months)
    train_data <- df_data %>% filter(time_id >= i & time_id < (i + window_size))
    
    # BFGS Optimization
    opt_res <- optim(
      par = rep(0, 4), 
      fn = calculate_utility_elastic_net, 
      data = train_data, 
      gamma = 5, 
      lambda = lambda_val,
      alpha = alpha_val,
      method = "BFGS"
    )
    theta_train <- opt_res$par 
    
    # Testing set extraction (Month 61)
    test_data <- df_data %>% filter(time_id == (i + window_size))
    X_test <- as.matrix(test_data[, c("x_mom", "x_gp", "x_vol", "x_skew")])
    R_test <- test_data$RET
    N_test <- nrow(test_data)
    
    # OOS Return calculation
    weights_oos <- (1/N_test) + (1/N_test) * (X_test %*% theta_train)
    portfolio_return_oos <- sum(weights_oos * R_test)
    benchmark_return <- mean(R_test)
    
    oos_results <- rbind(oos_results, data.frame(
      time_id = i + window_size,
      year = test_data$year[1],
      month = test_data$month[1],
      OOS_Return = portfolio_return_oos,
      Benchmark_Return = benchmark_return
    ))
  }
  
  # Calculate summary metrics
  ann_return_oos <- mean(oos_results$OOS_Return) * 12
  ann_vol_oos    <- sd(oos_results$OOS_Return) * sqrt(12)
  sharpe_oos     <- ann_return_oos / ann_vol_oos
  
  return(data.frame(
    Model = model_label,
    Alpha = alpha_val,
    Lambda = lambda_val,
    Annualized_Return = ann_return_oos,
    Annualized_Vol = ann_vol_oos,
    Sharpe_Ratio = sharpe_oos
  ))
}

# ------------------------------------------------------------------------------
# 4. Comparative Model Execution Loop
# ------------------------------------------------------------------------------

models_to_run <- list(
  list(alpha = 0.0, lambda = 0.0001, label = "Ridge (alpha = 0.0)"),
  list(alpha = 0.5, lambda = 0.0001, label = "Elastic Net (alpha = 0.5)"),
  list(alpha = 1.0, lambda = 0.0001, label = "Lasso (alpha = 1.0)"),
  list(alpha = 0.0, lambda = 0.0000, label = "Original (Unpenalized)")
)

comparison_summary <- data.frame()

for (m in models_to_run) {
  res <- run_ppp_model(m$alpha, m$lambda, m$label, df_clean)
  comparison_summary <- rbind(comparison_summary, res)
}

# ------------------------------------------------------------------------------
# 5. Output Summary Results
# ------------------------------------------------------------------------------
cat("\n=========================================================\n")
cat("Out-of-Sample Performance Comparison Report\n")
cat("=========================================================\n")
print(comparison_summary)
