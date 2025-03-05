import pandas as pd
import numpy as np
from DataStream.base import Detector

class TOPLINE(Detector):
    
    def __init__(self) -> None:
        self.context = None
        self.detected = False
        
    def __call__(self, current_window:pd.DataFrame) -> None:
        pass
        
    
    def fit(self, X_ref_window: pd.DataFrame, y_ref_window: pd.DataFrame, context) -> None:
        if self.context is None:
            self.context = context
    
    def detect(self, current_window:pd.DataFrame=None) -> bool:
        if self.context == 2 and not self.detected:
            self.detected = True
            return True