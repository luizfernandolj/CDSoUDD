import pandas as pd
import numpy as np
from DataStream.base import Detector
from detectors.iks_code.IKSSW import IKSSW
from detectors.iks_code.IKS import IKS

class IKS(Detector):
    
    def __init__(self, ca:float=1.95) -> None:
        self.ca = ca
        self.ikssw = None
        
    def __call__(self, current_window:pd.DataFrame) -> None:
        instances = current_window.values.tolist()
        
        for instance in instances:
            self.ikssw.Increment(instance)
        
    
    def fit(self, X_ref_window: pd.DataFrame, y_ref_window: pd.DataFrame) -> None:
        if self.ikssw is None:
            self.ikssw = IKSSW(X_ref_window.values.tolist())
        else:
            self.ikssw.Update()
    
    def detect(self, current_window:pd.DataFrame=None) -> bool:
        return self.ikssw.Test(self.ca)