# Task 04: Nondeterministic Model Training & Multi-Layer Seed Synchronization

## Overview
This task evaluates an agent's ability to diagnose and fix nondeterminism in a PyTorch model training pipeline. In machine learning research and quantitative backtesting, training nondeterminism destroys experiment reproducibility, invalidates ablation studies, and makes model auditing impossible.

## Sources of Difficulty (Why This Task Is Hard)

### 1. Multi-Library RNG De-synchronization
A common failure mode for coding agents is assuming that setting `torch.manual_seed(seed)` is sufficient for deep learning reproducibility. In practice, modern pipelines mix multiple random generators:
- **Python Standard Library `random`:** Used for sample ordering or data augmentation.
- **NumPy `np.random`:** Used for synthetic noise and array sampling.
- **PyTorch Core RNG (`torch.manual_seed`, `torch.cuda.manual_seed_all`):** Weights initialization.
- **PyTorch DataLoader Generator:** Standard PyTorch `DataLoader(..., shuffle=True)` draws from the global thread-local generator. Unless an explicit `torch.Generator().manual_seed(seed)` is passed to `DataLoader(..., generator=g)`, mini-batch ordering can drift between runs if any intermediate operations touch the global RNG.

### 2. Collection Ordering & Python Hash Seed Randomization
Even if all pseudo-random number generators are seeded, nondeterminism can leak through data structures:
- In the starter code, input feature names are held in a `set()`. In Python, `set` iteration order is randomized by `PYTHONHASHSEED`. Converting an unsorted `set` to a list assigns feature columns to arbitrary neural network input weights on different processes or invocations.
- To achieve true bit-for-bit weight hash equality (`res1["weight_hash"] == res2["weight_hash"]`), the agent must deterministically sort feature columns (`sorted(list(feature_set))`) and configure PyTorch backend determinism (`torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`, `torch.use_deterministic_algorithms(True)`).

