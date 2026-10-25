# Define a wrapper function to run the optimization for any alpha value
run_ppp_model <- function(alpha_val, model_name) {
  cat(sprintf("Running model: %s (alpha = %.1f)...\n", model_name, alpha_val))
}

results_ridge      <- run_ppp_model(alpha_val = 0.0, model_name = "Ridge")
results_elastic    <- run_ppp_model(alpha_val = 0.5, model_name = "Elastic Net")
results_lasso      <- run_ppp_model(alpha_val = 1.0, model_name = "Lasso")
