import os
import sys
import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    # ... (this function is unchanged)
    pass

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for model_name, model in models.items():
            para = param.get(model_name, {})

            # --- START: FIX 3 (Optional, if memory error persists) ---
            # Change n_jobs from -1 to 1 to stop parallel processing.
            # This will be slower but use much less RAM.
            gs = GridSearchCV(model, para, cv=3, verbose=2, n_jobs=1) 
            # --- END: FIX 3 ---

            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        
        return report

    except Exception as e:
        logging.error(f"Error in evaluate_models: {str(e)}")
        raise CustomException(e, sys)