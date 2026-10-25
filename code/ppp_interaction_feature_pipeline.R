# Script Name: ppp_interaction_feature_pipeline.R
# Description: Automated Interaction Feature Engineering & Rolling Elastic Net Portfolio Optimization with Dynamic Sharpe Cross-Validation

library(dplyr)
library(glmnet)

cat("=== 1. Starting Generation of Smile Slope and Interaction Features... ===\n")

# 1. Identify original feature columns (excluding identifiers and target variable)
orig_feature_cols <- setdiff(names(data_sorted), c("permno", "yyyymm", "RET"))

# 2. Automatically match Smile Slope related columns
smile_cols <- grep("smile|slope", orig_feature_cols, value = TRUE, ignore.case = TRUE)

if(length(smile_cols) == 0) {
  cat("No 'smile' or 'slope' keywords automatically matched. Defaulting to the first feature.\n")
  smile_cols <- orig_feature_cols[1]
} else {
  cat("Successfully identified Smile Slope features:", paste(smile_cols, collapse = ", "), "\n")
}

# 3. Other feature columns to interact with
other_cols <- setdiff(orig_feature_cols, smile_cols)

# 4. Batch generate interaction terms
interaction_list <- list()
for (s_col in smile_cols) {
  for (f_col in orig_feature_cols) {
    new_col_name <- paste0(s_col, "_x_", f_col)
    interaction_list[[new_col_name]] <- data_sorted[[s_col]] * data_sorted[[f_col]]
  }
}

interaction_df <- as.data.frame(interaction_list)

# 5. Bind new interaction features to the dataset
data_extended <- cbind(data_sorted, interaction_df)

# Update total feature list
all_feature_cols <- setdiff(names(data_extended), c("permno", "yyyymm", "RET"))

cat(sprintf("Feature expansion completed! Original features: %d, New interaction features: %d, Total features: %d.\n\n", 
            length(orig_feature_cols), ncol(interaction_df), length(all_feature_cols)))

# ==============================================================================
# Part 2: Cross-Sectional Standardization and Winsorization Cleaning
# ==============================================================================
cat("=== 2. Performing High-Dimensional Cross-Sectional Standardization & Winsorization... ===\n")

process_feature <- function(x) {
  m <- median(x, na.rm = TRUE)
  if(is.na(m)) m <- 0; x[is.na(x)] <- m
  q <- quantile(x, probs = c(0.01, 0.99), na.rm = TRUE)
  x[x < q[1]] <- q[1]; x[x > q[2]] <- q[2]
  s <- sd(x, na.rm = TRUE)
  if(is.na(s) || s == 0) return(rep(0, length(x)))
  return((x - mean(x, na.rm = TRUE)) / s)
}

df_clean <- data_extended %>%
  filter(!is.na(RET), !is.na(yyyymm)) %>%
  arrange(yyyymm) %>%
  group_by(yyyymm) %>%
  mutate(time_id = cur_group_id(), inv_N = 1 / n()) %>%
  mutate(across(all_of(all_feature_cols), process_feature, .names = "x_{.col}")) %>%
  ungroup()

standardized_features <- paste0("x_", all_feature_cols)

# ==============================================================================
# Part 3: Rolling Window & Dynamic Sharpe-CV Calculation
# ==============================================================================
total_months <- max(df_clean$time_id)
window_size  <- 60 
num_loops    <- total_months - window_size

oos_results <- data.frame(
  time_id = integer(),
  OOS_Return = numeric(),
  Benchmark_Return = numeric(),
  Selected_Lambda = numeric()
)

theta_history <- matrix(0, nrow = num_loops, ncol = length(all_feature_cols))
colnames(theta_history) <- all_feature_cols

cat("=== 3. Starting Rolling-Window Backtest (Dynamic Sharpe-CV for Lambda Selection)... ===\n")

for (i in 1:num_loops) {
  
  if(i %% 20 == 0) cat(sprintf("Backtest Progress: Completed %d / %d rolling windows\n", i, num_loops))
  
  # 1. Extract current 60-month training block
  window_data <- df_clean %>% filter(time_id >= i & time_id < (i + window_size))
  
  # 2. Inner validation split: first 48 months train, last 12 months validate Sharpe ratio
  inner_train <- window_data %>% filter(time_id < (i + window_size - 12))
  inner_val   <- window_data %>% filter(time_id >= (i + window_size - 12))
  
  X_inner_tr <- as.matrix(inner_train[, standardized_features])
  r_inner_tr <- inner_train$RET
  
  # 3. Generate 100 candidate lambda paths via glmnet
  glmnet_path <- glmnet(X_inner_tr, r_inner_tr, alpha = 0.5, nlambda = 100)
  lambda_candidates <- glmnet_path$lambda
  beta_matrix <- as.matrix(glmnet_path$beta) 
  
  # 4. Compute return matrix for 100 lambdas across the 12 validation months
  val_months <- unique(inner_val$time_id)
  val_returns_matrix <- matrix(0, nrow = length(val_months), ncol = length(lambda_candidates))
  
  for (m_idx in seq_along(val_months)) {
    month_data <- inner_val %>% filter(time_id == val_months[m_idx])
    X_val_m <- as.matrix(month_data[, standardized_features])
    r_val_m <- month_data$RET
    N_val_m <- nrow(month_data)
    
    weights_matrix <- (1 / N_val_m) + (1 / N_val_m) * (X_val_m %*% beta_matrix)
    val_returns_matrix[m_idx, ] <- colSums(weights_matrix * r_val_m)
  }
  
  # 5. Select optimal lambda maximizing validation Sharpe ratio
  val_means <- colMeans(val_returns_matrix)
  val_sds   <- apply(val_returns_matrix, 2, sd)
  sharpe_by_path <- ifelse(val_sds > 0, val_means / val_sds, -999)
  
  best_idx    <- which.max(sharpe_by_path)
  best_lambda <- lambda_candidates[best_idx]
  
  # 6. Full-window refitting to record coefficient vector theta
  X_train <- as.matrix(window_data[, standardized_features])
  r_train <- window_data$RET
  
  final_fit <- glmnet(X_train, r_train, alpha = 0.5, lambda = best_lambda)
  theta_opt <- as.numeric(final_fit$beta)
  theta_history[i, ] <- theta_opt
  
  # 7. Out-of-Sample (OOS) test for month t + 60
  test_data <- df_clean %>% filter(time_id == (i + window_size))
  X_test    <- as.matrix(test_data[, standardized_features])
  R_test    <- test_data$RET
  N_test    <- nrow(test_data)
  
  weights_oos          <- (1 / N_test) + (1 / N_test) * (X_test %*% theta_opt)
  portfolio_return_oos <- sum(weights_oos * R_test)
  
  oos_results <- rbind(oos_results, data.frame(
    time_id          = i + window_size,
    OOS_Return       = portfolio_return_oos,
    Benchmark_Return = mean(R_test),
    Selected_Lambda  = best_lambda
  ))
}

cat("\n=== All computations completed! ===\n\n")

# ==============================================================================
# Part 4: Feature Importance Evaluation & Performance Summary
# ==============================================================================

# 1. Compute mean absolute theta and selection rate across historical windows
feature_importance <- data.frame(
  Feature = colnames(theta_history),
  Mean_Abs_Theta = colMeans(abs(theta_history)),
  Selection_Rate = colMeans(theta_history != 0)
) %>% 
  mutate(Is_Interaction = grepl("_x_", Feature)) %>% 
  arrange(desc(Mean_Abs_Theta))

cat("=========================================================\n")
cat("Top 15 Feature Importance Ranking\n")
cat("=========================================================\n")
print(head(feature_importance, 15))
cat("\n")

# 2. Proportion of interaction terms in top 50 features
top_50 <- head(feature_importance, 50)
cat(sprintf("Among top 50 core features:\n- Smile Slope interaction terms: %d\n- Original characteristics: %d\n\n",
            sum(top_50$Is_Interaction), sum(!top_50$Is_Interaction)))

# 3. Strategy performance metrics
ann_return_oos   <- mean(oos_results$OOS_Return) * 12
ann_return_bench <- mean(oos_results$Benchmark_Return) * 12
ann_vol_oos      <- sd(oos_results$OOS_Return) * sqrt(12)
ann_vol_bench    <- sd(oos_results$Benchmark_Return) * sqrt(12)

sharpe_oos       <- ann_return_oos / ann_vol_oos
sharpe_bench     <- ann_return_bench / ann_vol_bench

cat("=========================================================\n")
cat("Strategy Performance & Sharpe Ratios\n")
cat("=========================================================\n")
cat(sprintf("Expanded Feature Strategy - Ann. Return: %.2f%% | Ann. Vol: %.2f%% | Sharpe Ratio: %.3f\n", 
            ann_return_oos * 100, ann_vol_oos * 100, sharpe_oos))
cat(sprintf("Equal-Weight Benchmark    - Ann. Return: %.2f%% | Ann. Vol: %.2f%% | Sharpe Ratio: %.3f\n", 
            ann_return_bench * 100, ann_vol_bench * 100, sharpe_bench))
cat("=========================================================\n")
