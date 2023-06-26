import os
import sys
import numpy as np
import pandas as pd

from mushroom.exception import CustomException
from mushroom.logger import logging
from mushroom.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds
            
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 bruises: str,
                 gill_spacing: str,
                 gill_size: str,
                 gill_color: str,
                 stalk_root: str,
                 ring_type: str,
                 spore_print_color: str 
                 ):
        self.bruises = bruises
        self.gill_spacing = gill_spacing
        self.gill_size = gill_size
        self.gill_color = gill_color
        self.stalk_root = stalk_root
        self.ring_type = ring_type
        self.spore_print_color = spore_print_color
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "bruises": [self.bruises],
                "gill-spacing": [self.gill_spacing],
                "gill-size": [self.gill_size],
                "gill-color": [self.gill_color],
                "stalk-root": [self.stalk_root],
                "ring-type": [self.ring_type],
                "spore-print-color": [self.spore_print_color]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)