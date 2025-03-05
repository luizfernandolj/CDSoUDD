import os
import shutil
import pandas as pd
import numpy as np
from mlquantify.utils import *
from detectors import *
from DataStream import *
from sklearn.ensemble import RandomForestClassifier
from DP import DriftExperiments
import argparse


def delete_files(files: list) -> None:
    for f in files:
        if os.path.isdir(f):
            os.remove(f)

def initialize_ibdd_folder(folder:str) -> None:
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

def run(dataset: str,
    WINDOW_SIZE: int,
    N_BATCHES: int,
    path_train: str,
    path_test: str,
    path_results: str,
    classifier: np.any,
    detector: Detector = None,
    files_to_del: list = None) -> None:
    
    print("Starting run() function.")
    
    ibdd_dir = f"{os.getcwd()}/detectors/for_ibdd/{dataset}"
    print(f"Initializing IBDD folder at: {ibdd_dir}")
    initialize_ibdd_folder(ibdd_dir)
    
    # IMPORTING DATASETS
    print("Reading training and testing data.")
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)


    # SPLITTING REFERENCE WINDOW
    print("Splitting reference window from training data.")
    X_REFERENCE = train.iloc[-WINDOW_SIZE:, :-2]
    Y_REFERENCE = train.iloc[-WINDOW_SIZE:, -2]
    
    # SPLITTING TEST CONTEXTS
    print("Splitting test data into context 1 and context 2.")
    context_1 = test[test["context"] == 1]
    context_2 = test[test["context"] == 2]
    
    print("Initializing drift experiments for context 1 and context 2.")
    context_1 = DriftExperiments(X_REFERENCE, Y_REFERENCE, detector, classifier, context_1, WINDOW_SIZE, N_BATCHES, 1, files_to_del)
    context_2 = DriftExperiments(X_REFERENCE, Y_REFERENCE, detector, classifier, context_2, WINDOW_SIZE, N_BATCHES, 2, files_to_del)
    
    # RUNNING CONTEXT 1
    print("Running drift experiments for context 1: artificial\n\n")
    result_artificial_1 = context_1.run("artificial")
    print("Running drift experiments for context 1: normal\n\n")
    result_normal_1 = context_1.run("normal")

    
    # RUNNING CONTEXT 2
    print("Running drift experiments for context 2: artificial\n\n")
    result_artificial_2 = context_2.run("artificial")
    print("Running drift experiments for context 2: normal\n\n")
    result_normal_2 = context_2.run("normal")
    
    result_artificial_1["context"] = 1
    result_normal_1["context"] = 1
    result_artificial_2["context"] = 2
    result_normal_2["context"] = 2
    
    print("Combining results for all runs.")
    result_artificial = pd.concat([result_artificial_1, result_artificial_2], ignore_index=True)
    result_normal = pd.concat([result_normal_1, result_normal_2], ignore_index=True)

        
    
    # SAVING RESULTS
    file_artificial = f"{path_results}/{dataset}_{detector.__class__.__name__}_artificial.csv"
    file_normal = f"{path_results}/{dataset}_{detector.__class__.__name__}_normal.csv"
    
    print(f"Saving artificial results to {file_artificial}.")
    result_artificial.to_csv(file_artificial, index=False)
    print(f"Saving normal results to {file_normal}.")
    result_normal.to_csv(file_normal, index=False)
    
    print("Run() function completed.")




if __name__ == '__main__':
    
    files2del = ['w1.jpeg', 'w2.jpeg', 'w1_cv.jpeg', 'w2_cv.jpeg']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('window_size', type=int)    
    parser.add_argument('detector', type=str)
    args = parser.parse_args()
    
    
    print(f"Starting -> {args.detector} on {args.dataset}")
    
    
    path_train = f"{os.getcwd()}/datasets/train/{args.dataset}.train.csv"
    path_test = f"{os.getcwd()}/datasets/test/{args.dataset}.test.csv"
    path_results = f"{os.getcwd()}/results"
    
    classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf_cdt = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    
    detector = None
    
    if args.detector == "CDT":
        p = 1.5
        train_size = 0.5
        detector = CDT(classifier=clf_cdt, p=p, train_split_size=train_size)
    if args.detector == "IKS":
        ca = 1.95
        detector = IKS(ca=ca)
    if args.detector == "IBDD":
        epsilon = 3
        detector = IBDD(consecutive_values=epsilon, n_runs=20, dataset=args.dataset, window_size=args.window_size)
    if args.detector == "WRS":
        threshold = 0.001
        detector = WRS(threshold=threshold, window_size=args.window_size)
    if args.detector == "BASELINE":
        detector = BASELINE()
    if args.detector == "TOPLINE":
        detector = TOPLINE()

    run(dataset=args.dataset,
        WINDOW_SIZE=args.window_size,
        N_BATCHES=100,
        path_train=path_train,
        path_test=path_test,
        path_results=path_results,
        classifier=classifier,
        detector=detector,
        files_to_del=files2del)
    
    print(f"\nEnd {args.dataset} with {args.detector}")
    
    
    
    