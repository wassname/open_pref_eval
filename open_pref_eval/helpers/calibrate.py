from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration
from sklearn.linear_model import LogisticRegression
import torch
import numpy as np

def get_calibrator(prob_chosen):
    N = len(prob_chosen)
    X = prob_chosen
    y = np.ones(N)

    # # flip around half, for stable calibration
    # X[N//2:] = 1 - X[N//2:]
    # y[N//2:] = 1 - y[N//2:]

    # return IsotonicRegression(out_of_bounds="nan", increasing=True).fit(X, y)
    # return _SigmoidCalibration().fit(X, y)
    return LogisticRegression().fit(X.reshape(-1, 1), y)

class PTIsotonicRegression:
    """Use Sklearn's IsotonicRegression with PyTorch tensors to calibrate probabilities
    
    see: 
    - https://scikit-learn.org/stable/api/sklearn.calibration.html
    - https://github.com/scikit-learn/scikit-learn/blob/70fdc843a4b8182d97a3508c1a426acc5e87e980/sklearn/calibration.py#L620
    """

    def __init__(self, y_min=None, y_max=None, increasing=True, out_of_bounds='clip'):
        self.ir = IsotonicRegression(y_min=y_min, y_max=y_max, increasing=increasing, out_of_bounds=out_of_bounds)

    def fit(self, prob_chosen, prob_rejected):
        prob_chosen = prob_chosen.flatten()
        prob_rejected = prob_rejected.flatten()

        X = torch.concatenate([prob_chosen, prob_rejected])
        y = torch.concatenate([torch.ones_like(prob_chosen), torch.zeros_like(prob_rejected)])
        return self.ir.fit(X, y)

    def predict(self, prob):
        prob_calib = self.ir.predict(prob).reshape(prob.shape)
        return torch.from_numpy(prob_calib)
