import os 
import sys
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Initializinf model trainer')
            logging.info('splitting training and test input data')
            
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            embedding_vector_features = 40  # Dimensions
            voc_size = 5000
            sent_length = 20
            model = Sequential()
            model.add(Embedding(voc_size, embedding_vector_features, input_length = sent_length))
            model.add(LSTM(100))
            model.add(Dense(1, activation = 'sigmoid'))
            model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
            logging.info(f"model summary {model.summary()}")
            
            model_report = evaluate_models(X_train=X_train, y_train=y_train,
                                           X_test=X_test, y_test=y_test,model=model)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            
            predicted = model.predict(X_test)
            predicted = np.where(predicted >0.5, 1, 0)
            
            acc_score = accuracy_score(y_test, predicted)
            return acc_score
        
        except Exception as e:
            raise CustomException(e, sys)
            
            
            
            
            
            
                    