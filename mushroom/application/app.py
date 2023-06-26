import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from mushroom.pipeline.prediction_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)

app = application

# route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    
    else:
        data = CustomData(
            bruises = request.form.get("bruises"),
            gill_spacing = request.form.get("gill-spacing"),
            gill_size = request.form.get("gill-size"),
            gill_color = request.form.get("gill-color"),
            stalk_root = request.form.get("stalk-root"),
            ring_type = request.form.get("ring-type"),
            spore_print_color = request.form.get("spore-print-color")
        )
        
        pred_df = data.get_data_as_data_frame()
        
        predit_pipeline = PredictionPipeline()
        results = predit_pipeline.predict(pred_df)
        
        if results == 0.0:
            answer = "edible"
        else:
            answer = "poisonous"
            
        return render_template('index.html',results=answer)
    
if __name__ == "__main__":
    app.run(debug=True)  