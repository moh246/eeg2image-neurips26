"""Fitting and evaluation loops."""

from eeg2img.engine.evaluate import evaluate, predict
from eeg2img.engine.fit import fit_model

__all__ = ["evaluate", "fit_model", "predict"]
