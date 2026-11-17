"""
Training pipeline and evaluation metrics for Video Anomaly Detection.
"""

def __getattr__(name: str):
    if name in ("compute_metrics", "MetricsCalculator"):
        from . import metrics
        return getattr(metrics, name)
    if name == "TrainingPipeline":
        from . import trainer
        return getattr(trainer, name)
    if name == "HyperparameterTuner":
        from . import tuner
        return getattr(tuner, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["compute_metrics", "MetricsCalculator", "TrainingPipeline", "HyperparameterTuner"]
