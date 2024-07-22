import os
import sys
import tensorflow as tf
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_models(X_train, y_train, X_test, y_test, model):
    try:
        
        ### Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss', patience = 20,min_delta = 0.0001,
            verbose = 1,mode = 'auto', baseline = None,
            restore_best_weights = False )
        
        model.fit(X_train, y_train, validation_data = (X_test, y_test), 
                  epochs = 50, batch_size = 64, callbacks = early_stopping)
        
        logging.info("Model training completed.")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        y_train_pred =np.where(y_train_pred > 0.5, 1, 0)
        y_test_pred =np.where(y_test_pred > 0.5, 1, 0)
        
        train_model_score = accuracy_score(y_train, y_train_pred)
        test_model_score = accuracy_score(y_test, y_test_pred)
        
        report = classification_report(y_test, y_test_pred)
        
        return report, test_model_score
    
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)        
            