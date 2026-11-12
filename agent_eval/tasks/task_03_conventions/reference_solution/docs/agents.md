# Multi-Agent Pipeline Architecture

QuantLab Pro coordinates a deterministic sequential pipeline across 7 specialized research agents:

1. **DataQualityAgent**: Validates schema and dataset integrity.
2. **PlannerAgent**: Formulates research plan and hypothesis.
3. **FeatureEngineeringAgent**: Builds leakage-free feature representations.
4. **TrainingAgent**: Trains ML models under fixed random seeds.
5. **EvaluationAgent**: Computes held-out metrics and backtest performance.
6. **AuditorAgent**: Assembles reproducible manifest and hashes artifacts.
7. **RiskStressAgent**: Performs downside stress testing, margin haircuts, and tail-risk audits.

