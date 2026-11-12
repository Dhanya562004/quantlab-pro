# Rubric: Task 01 - Leakage

## Procedural Test Checks
1. `test_public_api_signatures`: Confirms `compute_features` and `prepare_dataset` maintain expected argument names and default values.
2. `test_feature_columns_and_types`: Confirms columns produced are exactly `['returns', 'rolling_mean', 'rolling_std', 'z_score', 'volatility_ratio']`.
3. `test_future_perturbation_leakage`: Confirms that mutating prices from $t \ge 60$ has zero impact on feature values at $t \le 50$.
4. `test_no_backward_fill_leakage`: Confirms no `bfill` or forward-looking imputation was applied to warmup rows.
5. `test_prepare_dataset_contract_and_alignment`: Confirms output $(X, y)$ has identical length, no missing values, and binary integer target labels.

## Qualitative Checks (Non-executable)
- Code does not disable warnings to hide pandas alignment or future downcasting issues.
- Solution does not hardcode sample data shapes or specific indices.

