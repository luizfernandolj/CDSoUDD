from mlquantify.utils import generate_artificial_indexes, generate_artificial_prevalences, get_real_prev
import time as t
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from copy import deepcopy

class DriftExperiments:
    def __init__(self, X_ref, y_ref, detector, classifier, dataset, window_size, n_batches, context, files_to_del):
        self.X_ref = X_ref
        self.y_ref = y_ref
        self.detector_art = deepcopy(detector)
        self.detector_normal = deepcopy(detector)
        self.classifier = classifier
        self.X = dataset.iloc[:, :-2]
        self.Y = dataset.iloc[:, -2]
        self.window_size = window_size
        self.n_batches = n_batches
        self.context = context
        self.files_to_del = files_to_del
        
    def split_batches(self, typpe):
        if typpe == "normal":
            for i in range(self.n_batches):
                indexes = np.random.choice(len(self.Y), self.window_size, replace=True)
                
                X_batch = self.X.iloc[indexes]
                Y_batch = self.Y.iloc[indexes]
                
                prev = get_real_prev(Y_batch)
                
                yield X_batch, Y_batch, prev[2]
        else:
            prevalences = []
            for _ in range(self.n_batches):
                val = np.random.uniform(0.1, 0.9)
                prevalences.append([val, 1 - val])
                
            for prevalence in prevalences:
                indexes = generate_artificial_indexes(self.Y, prevalence, self.window_size, classes=[1, 2])
                
                X_batch = self.X.iloc[indexes]
                Y_batch = self.Y.iloc[indexes]
                
                yield X_batch, Y_batch, prevalence[1]

    def run(self, type_exp):
        
        if type_exp == "normal":
            self.detector = self.detector_normal
        else:
            self.detector = self.detector_art
        
        if self.detector.__class__.__name__ == "TOPLINE":
            self.detector.fit(self.X_ref, self.y_ref, self.context)
        else:
            self.detector.fit(self.X_ref, self.y_ref)
            
        self.classifier.fit(self.X_ref, self.y_ref)
        
        d2 = None
        
        
        result = {"drifs_detected":np.zeros(self.n_batches), 
              "false_alarms":np.zeros(self.n_batches),
              "class_distribution":np.zeros(self.n_batches), 
              "time (s)":np.zeros(self.n_batches), 
              "classification":np.zeros(self.n_batches)}
        
        i = 0
        
        for X_batch, Y_batch, prevalence in self.split_batches(type_exp):
            print(f"       Batch {i+1}/{self.n_batches}", end="\r")
            start = t.time()    
            
            self.detector(X_batch)
            result["class_distribution"][i] = prevalence
            y_pred = self.classifier.predict(X_batch)
            result["classification"][i] = accuracy_score(Y_batch, y_pred)
            
            if self.detector.detect(X_batch):
                print(f"Drift")
                
                result["drifs_detected"][i] = 1
                
                if self.context == 1:
                    result["false_alarms"][i] = 1
                
                if self.context == 2:
                    if d2 is not None:
                        result["false_alarms"][i] = 1
                    else:
                        d2 = i
                    
                if self.detector.__class__.__name__ == "TOPLINE":
                    self.detector.fit(X_batch, Y_batch, self.context)
                else:
                    self.detector.fit(X_batch, Y_batch)
                self.classifier.fit(X_batch, Y_batch)
                
            end = t.time()
            
            result["time (s)"][i] = abs(end - start)
            
            i += 1
            
        if self.detector.__class__.__name__ == "IBDD":
            for file in self.files_to_del:
                try:
                    os.remove(file)
                except:
                    pass
        
        return pd.DataFrame(result)
            
        
        
        
    