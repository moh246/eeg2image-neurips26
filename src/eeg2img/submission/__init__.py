"""Packaging trained models into the Codabench submission format."""

from eeg2img.submission.builder import render_submission_source, write_package
from eeg2img.submission.check import check_submission
from eeg2img.submission.runtime import PackagedModel, load_packaged_model

__all__ = [
    "PackagedModel",
    "check_submission",
    "load_packaged_model",
    "render_submission_source",
    "write_package",
]
