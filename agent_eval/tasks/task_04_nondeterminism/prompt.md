# Task Prompt: Fix Nondeterminism in PyTorch Training Pipeline

You are assigned to eliminate nondeterminism from `train_pipeline.py`.

## Problem Statement
In model training and validation, calling `train_model(seed=42)` repeatedly produces different model weights, fluctuating loss curves, and different weight hashes across runs. This destroys reproducibility for backtests and production audits.

## Requirements in `train_pipeline.py`

### 1. Function Signature & Return Contract
Modify `train_model(seed: int = 42, epochs: int = 15) -> dict[str, Any]` in place. It must return a dictionary with exact keys:
- `"model"`: The trained PyTorch `nn.Module`.
- `"loss_history"`: List of training loss floats for each epoch (`list[float]`).
- `"weight_hash"`: SHA-256 hex string (`str`) computed from concatenated model parameter bytes.
- `"final_loss"`: Float value of the final epoch loss.
- `"feature_names"`: List of feature names sorted alphabetically (`list[str]`).

### 2. Strict Reproducibility Guarantees
- **Bit-for-Bit Determinism**: Calling `train_model(seed=42)` twice under identical environments must produce:
  - Bit-for-bit identical `"weight_hash"` values.
  - Identical `"loss_history"` values across all epochs.
  - Identical sorted `"feature_names"`.
- **Seed Sensitivity (No Mocking or Hardcoding)**:
  - Calling `train_model(seed=42)` vs `train_model(seed=99)` must produce different `"weight_hash"` values. The model must genuinely train and not return dummy static constants.
  - The model must actively learn: final loss must be strictly lower than initial epoch loss (`final_loss < loss_history[0]`).

### 3. Required Fixes Across the Pipeline
To achieve complete determinism, your implementation must address all underlying causes:
1. **Multi-Generator Seeding**: Synchronize random seeds for Python `random.seed(seed)`, NumPy `np.random.seed(seed)`, and PyTorch (`torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`).
2. **Deterministic Data Ordering**: Sort input feature columns alphabetically before feeding tensors into the model (`sorted(...)`).
3. **PyTorch DataLoader Generator**: When creating the `DataLoader`, instantiate a seeded generator `g = torch.Generator().manual_seed(seed)` and pass it: `DataLoader(..., generator=g, shuffle=True)`.
4. **PyTorch Backend Flags**: Configure `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.

