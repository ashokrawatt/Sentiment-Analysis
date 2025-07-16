import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

def flatten_array(x):
    return x.flatten()

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns=['Rating', 'Reviews' ]
            categorical_columns=[ 'Category', 'Installs', 'Type', 'Content Rating' ]
            text_column = ['Translated_Review']

            num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
            cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),("scaler", StandardScaler(with_mean=False))])

            # Reduce the number of features created by TF-IDF
            text_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='')),
                ('flattener', FunctionTransformer(flatten_array)),
                ('tfidf', TfidfVectorizer(max_features=1000)) # Reduced from 5000 to 1000
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns),
                    ("text_pipeline", text_pipeline, text_column)
                ],
                remainder='drop'
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed.")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "Sentiment"
            
            train_df.dropna(subset=[target_column_name], inplace=True)
            test_df.dropna(subset=[target_column_name], inplace=True)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
            target_feature_train_df = target_feature_train_df.map(sentiment_mapping)
            target_feature_test_df = target_feature_test_df.map(sentiment_mapping)

            logging.info("Applying preprocessing object... This may take a moment.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr.toarray(), np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr.toarray(), np.array(target_feature_test_df)]
            
            train_arr = train_arr[~np.isnan(train_arr).any(axis=1)]
            test_arr = test_arr[~np.isnan(test_arr).any(axis=1)]

            train_arr = train_arr.astype(np.float32)
            test_arr = test_arr.astype(np.float32)
            logging.info("Converted final arrays to float32 to save memory.")

            logging.info("Saving preprocessing object.")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e, sys)