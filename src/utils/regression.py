import numpy as np
import torch
from typing import Callable

class RegressionResults:
    def __init__(self, values:torch.Tensor, predictions:torch.Tensor,loss_func:Callable):
        self.values = values
        self.predictions = predictions
        self.total = values.size(0)
        self.loss_func = loss_func
        self.device = self.values.device

    def loss(self):
        return self.loss_func(self.values,self.predictions)