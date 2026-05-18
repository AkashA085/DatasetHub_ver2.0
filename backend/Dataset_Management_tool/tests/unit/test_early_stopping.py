import pytest
import torch
import torch.nn as nn
from pathlib import Path
from app.api.routes.training import EarlyStopping

def test_early_stopping_initialization():
    stopper = EarlyStopping(patience=5, min_delta=0.01, save_path="test_best.pt")
    assert stopper.patience == 5
    assert stopper.min_delta == 0.01
    assert stopper.save_path == "test_best.pt"
    assert stopper.best_metric is None
    assert stopper.no_improve_count == 0

def test_early_stopping_improvement(tmp_path):
    save_path = tmp_path / "best.pt"
    model = nn.Linear(10, 1)
    stopper = EarlyStopping(patience=3, min_delta=0.1, save_path=str(save_path))
    
    # First step
    stop = stopper.step(0.5, model)
    assert stop is False
    assert stopper.best_metric == 0.5
    assert save_path.exists()
    
    # Improved step
    stop = stopper.step(0.7, model)
    assert stop is False
    assert stopper.best_metric == 0.7
    assert stopper.no_improve_count == 0
    
    # Minor improvement (less than min_delta)
    stop = stopper.step(0.75, model) # delta is 0.05 < 0.1
    assert stop is False
    assert stopper.best_metric == 0.7
    assert stopper.no_improve_count == 1

def test_early_stopping_patience(tmp_path):
    save_path = tmp_path / "best.pt"
    model = nn.Linear(10, 1)
    stopper = EarlyStopping(patience=2, min_delta=0.1, save_path=str(save_path))
    
    stopper.step(0.5, model)
    
    # No improvement 1
    stop = stopper.step(0.5, model)
    assert stop is False
    assert stopper.no_improve_count == 1
    
    # No improvement 2 -> Stop
    stop = stopper.step(0.5, model)
    assert stop is True
    assert stopper.no_improve_count == 2
