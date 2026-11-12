# Rubric: Task 04 - Nondeterminism

## Procedural Test Checks
1. `test_api_contract_and_feature_sorting`: Validates return dict keys, types, SHA-256 weight hash string length, and alphabetical feature sorting.
2. `test_identical_runs_produce_identical_weights`: Validates that training twice with seed 42 produces identical parameter byte hashes and identical epoch loss history.
3. `test_different_seeds_produce_different_weights`: Confirms model generates distinct parameter weights for different seeds.
4. `test_model_actively_learns`: Confirms optimization occurs and final loss is lower than initial loss.

