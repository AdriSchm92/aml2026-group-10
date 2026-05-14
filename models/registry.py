"""Model registry.

Convention: ``--model <name>`` loads ``models/<name>.py`` and calls
its ``build_model(num_classes)`` function. Adding a new model is a single
file with no edits elsewhere.

Example new model file (``models/my_model.py``)::

    import torch.nn as nn

    def build_model(num_classes: int) -> nn.Module:
        return MyModel(num_classes=num_classes)

Then run::

    python train.py --model my_model
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import torch.nn as nn


def available_models() -> list[str]:
    """List the Python files in ``models/`` that could be model entry points."""
    here = Path(__file__).resolve().parent
    names = []
    for info in pkgutil.iter_modules([str(here)]):
        if info.name in ("registry", "__init__"):
            continue
        names.append(info.name)
    return sorted(names)


def load_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    """Import ``models.<name>`` and return its ``build_model(num_classes, **kwargs)``.

    ``kwargs`` are forwarded to ``build_model`` as keyword arguments.  Existing
    models that only accept ``num_classes`` should declare ``**_ignored`` to
    stay compatible, or simply ignore the extras — Python ignores unknown kwargs
    when a function declares ``**kwargs``.  The ``cnn_transformer`` model uses
    this to expose architecture hyperparameters (``d_model``, ``n_layers``, …)
    to the HP search runner without modifying the training harness.
    """
    try:
        module = importlib.import_module(f"models.{name}")
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Unknown model {name!r}. Available: {available_models()}. "
            f"Add models/{name}.py with a build_model(num_classes) function."
        ) from e

    if not hasattr(module, "build_model"):
        raise AttributeError(
            f"models/{name}.py must define build_model(num_classes) -> nn.Module"
        )
    model = module.build_model(num_classes, **kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"models/{name}.build_model must return an nn.Module, got {type(model)}"
        )
    return model
