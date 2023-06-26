import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle

from project.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(TrainFeatures, TrainTarget, TestFeatures, TestTarget, models, params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(TrainFeatures, TrainTarget)
            
            model.set_params(**gs.best_params_)            
            model.fit(TrainFeatures, TrainTarget)
            
            y_train_pred = model.predict(TrainFeatures)
            y_test_pred = model.predict(TestFeatures)
            
            train_model_score = accuracy_score(TrainTarget, y_train_pred)
            test_model_score = accuracy_score(TestTarget, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    