import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline :
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")   # It is returning model path which we are storing in model_path
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")  # It is returning preprocessor path which we are storing in preprocessor_path
            print("Before Loading")
            model=load_object(file_path=model_path)# To open the model pickle file
            preprocessor=load_object(file_path=preprocessor_path) # To open the model pickle file
            print("After Loading")
            data_scaled=preprocessor.transform(features)   # To scaled the data(normalize the data) data is passed by the user we pass it through features
            preds=model.predict(data_scaled)  # It will predict the result
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education:str,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:int,
                 writing_score:int
                 ):                                 # To mention the datatype
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)