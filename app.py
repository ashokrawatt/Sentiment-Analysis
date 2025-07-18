from flask import Flask , request, render_template
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application


@app.route('/')
def index():
    """
    Renders the landing page (index.html).
    """
    return render_template('index.html')

@app.route('/home')
def home():
    """
    Renders the main analysis page (home.html).
    """
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    custom_data_obj = CustomData(
        Translated_Review=request.form.get('review_text'),
        Category="EDUCATION",
        Rating=4.5,
        Reviews=50000,
        Installs="1,000,000+",
        Type="Free",
        Content_Rating="Everyone"
    )
    
    pred_df = custom_data_obj.get_data_as_data_frame()
    
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    prediction_text = sentiment_map.get(results[0], "Prediction Error")
    
    return render_template('home.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)