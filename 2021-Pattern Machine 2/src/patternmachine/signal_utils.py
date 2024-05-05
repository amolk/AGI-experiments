import torch


class SignalUtils:
    @staticmethod
    def compute_precision(variance):
        return torch.exp(-variance)
