# ==============================================================================
# 第一部分：自动拓展数据（ Smile Slope 交叉项构建 ）
# ==============================================================================
library(dplyr)
library(glmnet)

cat("=== 1. 开始生成 Smile Slope 交叉项特征... ===\n")

# 1. 识别原始特征列
orig_feature_cols <- setdiff(names(data_sorted), c("permno", "yyyymm", "RET"))

# 2. 自动匹配 Smile Slope 相关的列（根据列名中包含 'smile' 或 'slope'）
smile_cols <- grep("smile|slope", orig_feature_cols, value = TRUE, ignore.case = TRUE)

# 防护提示：如果列名没有 smile/slope 关键字，请手动在下面定义列名
if(length(smile_cols) == 0) {
  cat("⚠️ 警告：未在列名中自动匹配到 'smile' 或 'slope'，默认使用第一个特征作为 Smile Slope 代表。\n")
  smile_cols <- orig_feature_cols[1]
} else {
  cat("成功识别出 Smile Slope 相关特征:", paste(smile_cols, collapse = ", "), "\n")
}

# 3. 其它需要与其相乘的特征列
other_cols <- setdiff(orig_feature_cols, smile_cols)

# 4. 批量生成相乘的新列 ( Interaction Terms )
interaction_list <- list()
for (s_col in smile_cols) {
  for (f_col in orig_feature_cols) {
    # 如果避免自身相乘，可以加 if(s_col != f_col)；这里默认生成所有组合
    new_col_name <- paste0(s_col, "_x_", f_col)
    interaction_list[[new_col_name]] <- data_sorted[[s_col]] * data_sorted[[f_col]]
  }
}

interaction_df <- as.data.frame(interaction_list)

# 5. 将新的交叉特征列紧跟在旧特征数据之后
data_extended <- cbind(data_sorted, interaction_df)

# 更新拓展后的总特征列表
all_feature_cols <- setdiff(names(data_extended), c("permno", "yyyymm", "RET"))

cat(sprintf("特征拓展完成！原始特征 %d 个，新增交叉特征 %d 个，当前总特征 %d 个。\n\n", 
            length(orig_feature_cols), ncol(interaction_df), length(all_feature_cols)))

# ==============================================================================
# 第二部分：拓展特征的截面标准化清洗
# ==============================================================================
cat("=== 2. 进行高维截面标准化与异常值 Winsorize 清洗... ===\n")

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
# 第三部分：滚动窗口（Rolling Window）与动态 Sharpe-CV 计算
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

# 用于存储所有特征（含原始特征和新交叉项）的系数历史
theta_history <- matrix(0, nrow = num_loops, ncol = length(all_feature_cols))
colnames(theta_history) <- all_feature_cols

cat("=== 3. 开启 M3 矩阵化 Rolling 回测 (基于动态 Sharpe-CV 选 Lambda)... ===\n")

for (i in 1:num_loops) {
  
  if(i %% 20 == 0) cat(sprintf("回测进度: 已完成 %d / %d 个滚动窗口\n", i, num_loops))
  
  # 1. 提取当前 60 个月的大组数据
  window_data <- df_clean %>% filter(time_id >= i & time_id < (i + window_size))
  
  # 2. 划分 Inner 验证组：前 48 个月训练，后 12 个月（第 49-60 月）验证夏普比率
  inner_train <- window_data %>% filter(time_id < (i + window_size - 12))
  inner_val   <- window_data %>% filter(time_id >= (i + window_size - 12))
  
  X_inner_tr <- as.matrix(inner_train[, standardized_features])
  r_inner_tr <- inner_train$RET
  
  # 3. 瞬间生成该数据密度下的 100 个候选 Lambda 及其系数矩阵
  glmnet_path <- glmnet(X_inner_tr, r_inner_tr, alpha = 0.5, nlambda = 100)
  lambda_candidates <- glmnet_path$lambda
  beta_matrix <- as.matrix(glmnet_path$beta) # 维度: P x 100
  
  # 4. 计算 100 个 Lambda 在后 12 个验证月里的收益率矩阵
  val_months <- unique(inner_val$time_id)
  val_returns_matrix <- matrix(0, nrow = length(val_months), ncol = length(lambda_candidates))
  
  for (m_idx in seq_along(val_months)) {
    month_data <- inner_val %>% filter(time_id == val_months[m_idx])
    X_val_m <- as.matrix(month_data[, standardized_features])
    r_val_m <- month_data$RET
    N_val_m <- nrow(month_data)
    
    # 矩阵乘法计算当前月 100 个 Lambda 的持仓权重及组合收益
    weights_matrix <- (1 / N_val_m) + (1 / N_val_m) * (X_val_m %*% beta_matrix)
    val_returns_matrix[m_idx, ] <- colSums(weights_matrix * r_val_m)
  }
  
  # 5. 选择验证集上夏普比率最高的最优 Lambda
  val_means <- colMeans(val_returns_matrix)
  val_sds   <- apply(val_returns_matrix, 2, sd)
  sharpe_by_path <- ifelse(val_sds > 0, val_means / val_sds, -999)
  
  best_idx    <- which.max(sharpe_by_path)
  best_lambda <- lambda_candidates[best_idx]
  
  # 6. 全量训练并记录该期特征系数 theta
  X_train <- as.matrix(window_data[, standardized_features])
  r_train <- window_data$RET
  
  final_fit <- glmnet(X_train, r_train, alpha = 0.5, lambda = best_lambda)
  theta_opt <- as.numeric(final_fit$beta)
  theta_history[i, ] <- theta_opt
  
  # 7. 第 61 个月进行样本外 (OOS) 实战测试
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

cat("\n=== 所有计算完成！ ===\n\n")

# ==============================================================================
# 第四部分：特征重要性评估（判断哪个 Feature 最好用）与 夏普比率结算
# ==============================================================================

# 1. 计算所有特征在整个历史窗口期的【平均系数绝对值】作为重要性指标
feature_importance <- data.frame(
  Feature = colnames(theta_history),
  Mean_Abs_Theta = colMeans(abs(theta_history)),
  Selection_Rate = colMeans(theta_history != 0) # 被选入模型的频率（非零占比）
) %>% 
  mutate(Is_Interaction = grepl("_x_", Feature)) %>% # 标记是否为新生成的交叉项
  arrange(desc(Mean_Abs_Theta))

cat("=========================================================\n")
cat("🏆【特征战力排行榜】 Top 15 最好用的 Feature\n")
cat("=========================================================\n")
print(head(feature_importance, 15))
cat("\n")

# 2. 统计原始特征 vs 新生成的交叉特征的贡献对比
top_50 <- head(feature_importance, 50)
cat(sprintf("在排名前 50 的核心特征中：\n- 新生成的 Smile Slope 交叉项占了: %d 个\n- 原始因子占了: %d 个\n\n",
            sum(top_50$Is_Interaction), sum(!top_50$Is_Interaction)))

# 3. 计算拓展数据后的最终策略 Performance
ann_return_oos   <- mean(oos_results$OOS_Return) * 12
ann_return_bench <- mean(oos_results$Benchmark_Return) * 12
ann_vol_oos      <- sd(oos_results$OOS_Return) * sqrt(12)
ann_vol_bench    <- sd(oos_results$Benchmark_Return) * sqrt(12)

sharpe_oos       <- ann_return_oos / ann_vol_oos
sharpe_bench     <- ann_return_bench / ann_vol_bench

cat("=========================================================\n")
cat("📊【拓展数据后的策略绩效与夏普比率 (Sharpe Ratio)】\n")
cat("=========================================================\n")
cat(sprintf("拓展特征策略 - 年化收益: %.2f%% | 年化波动: %.2f%% | 夏普比率: %.3f\n", 
            ann_return_oos * 100, ann_vol_oos * 100, sharpe_oos))
cat(sprintf("等权重基准策略 - 年化收益: %.2f%% | 年化波动: %.2f%% | 夏普比率: %.3f\n", 
            ann_return_bench * 100, ann_vol_bench * 100, sharpe_bench))
cat("=========================================================\n")