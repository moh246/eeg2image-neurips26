"""Fitting and evaluation loops."""

from eeg2img.engine.evaluate import evaluate, predict
from eeg2img.engine.fit import fit_model
from eeg2img.engine.train import evaluate_source, predict_source, train_model

__all__ = ["evaluate", "evaluate_source", "fit_model", "predict", "predict_source", "train_model"]
