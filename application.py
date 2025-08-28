from flask import Flask,request,render_template
# import numpy as np
# import pandas as pd

# from sklearn.preprocessing import StandardScaler
from src.Pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__) # To activate the flask...or to create the local_host

app=application

# Route for a home page

@app.route('/') # means it will open localhost and the index function
def index():
    return render_template('index.html')
# render template is flash function used to display html files.helps to connect python code with html

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':               # Used to read only.....Just to view the page
        return render_template('home.html')
    else:                                       # It is Post used to get the data from user
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get("parental_level_of_education"),  # request.form.get.....to accept the data
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=request.form.get("reading_score"),
            writing_score=request.form.get("writing_score")
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()   # predict_pipeline is an object of the PredictPipeline
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)  # Here predict is a function of PredictPipeline class
        print("After Prediction")
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)

