# In src/pipeline/prediction_pipeline.py

import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        """
        Initializes the prediction pipeline.
        """
        pass

    def predict(self, features):
        """
        Loads the saved model and preprocessor to make a prediction.
        Args:
            features (pd.DataFrame): A DataFrame containing the new data to predict on.
        Returns:
            The prediction result.
        """
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 Category: str,
                 Installs: str,
                 Type: str,
                 Content_Rating: str,
                 Rating: float,
                 Reviews: int,
                 Translated_Review: str):
        """
        This class is responsible for mapping all the inputs from an HTML form
        (or any other source) to the specific data types required.
        """
     
        self.Category = Category
        self.Installs = Installs
        self.Type = Type
        self.Content_Rating = Content_Rating
        self.Rating = Rating
        self.Reviews = Reviews
        self.Translated_Review = Translated_Review

    def get_data_as_data_frame(self):
        """
        Converts the custom data into a pandas DataFrame, which is the
        format required by the prediction pipeline.
        """
        try:
          
            custom_data_input_dict = {
                "Category": [self.Category],
                "Installs": [self.Installs],
                "Type": [self.Type],
                "Content Rating": [self.Content_Rating],
                "Rating": [self.Rating],
                "Reviews": [self.Reviews],
                "Translated_Review": [self.Translated_Review]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    print("Starting prediction test...")

    sample_data = CustomData(
        Category="GAME",
        Installs="1,000,000+",
        Type="Free",
        Content_Rating="Everyone",
        Rating=4.5,
        Reviews=12345,
        Translated_Review="This is a fantastic game, I play it every day!"
    )

    # Convert the sample data into a DataFrame
    pred_df = sample_data.get_data_as_data_frame()
    print("\nSample Input DataFrame:")
    print(pred_df)

    # prediction
    pipeline = PredictPipeline()
    results = pipeline.predict(pred_df)
    
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_sentiment = sentiment_map[results[0]]

    print(f"\nPrediction Result (Numerical): {results[0]}")
    print(f"Predicted Sentiment: {predicted_sentiment}")