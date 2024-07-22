import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding

@dataclass
class DatatransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DatatransformationConfig()

    def get_data_transformer_object(self, messages):
        """
        This function is responsible for data transformation
        """
        try:
            ps = PorterStemmer()
            corpus = []
            voc_size = 5000
            
            for message in messages:
                review = re.sub('[^a-zA-Z]', ' ', message)
                review = review.lower()
                review = review.split()
                
                # Stemming and removing stopwords
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
                review = ' '.join(review)
                corpus.append(review)
                
            # Performing one hot encoding
            onehot_repr = [one_hot(words, voc_size) for words in corpus]
                
            # Creating Embedding layer and padding to make length of all sentences same
            sent_length = 20
            embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
                
            return embedded_docs
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        
            logging.info('Reading training and testing data completed')
            logging.info('Obtaining preprocessing object')

            target_column_name = 'label'
            text_column_name = 'title'  # Assuming 'text' is the column name for text data
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing dataframe")

            # Apply transformation
            input_feature_train_arr = self.get_data_transformer_object(input_feature_train_df[text_column_name].values)
            input_feature_test_arr = self.get_data_transformer_object(input_feature_test_df[text_column_name].values)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Saving preprocessing object")
            
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=self.get_data_transformer_object)
            
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e, sys)
