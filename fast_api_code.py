from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from io import StringIO
import requests

app = FastAPI()

class IrisSpecies(BaseModel):
    sepal_length: float 
    sepal_width: float 
    petal_length: float 
    petal_width: float

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post('/predict')
async def predict_species(iris: IrisSpecies):
    data = iris.dict()
    loaded_model = pickle.load(open('LRClassifier.pkl', 'rb'))
    data_in = [[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]]
    prediction = loaded_model.predict(data_in)
    probability = loaded_model.predict_proba(data_in).max()
            
    return {
        'prediction': prediction[0],
        'probability': probability
    }

@app.post('/predict_mlflow')
async def predict_species(iris: IrisSpecies):
    data = iris.dict()

    data_in = [[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]]
    
    endpoint = "http://localhost:1234/invocations"
    inference_request = {
        "dataframe_records": data_in
    }
    print(inference_request)
    response = requests.post(endpoint, json=inference_request)
    print(response)
    return {
        'prediction': response.text,
    }

@app.post("/files_mlflow/")
async def create_file(file: bytes = File(...), token: str = Form(...)):
    s=str(file,'utf-8')
    data = StringIO(s)
    df=pd.read_csv(data)
    lst = df.values.tolist()
    inference_request = {
        "dataframe_records": lst
    }
    
    endpoint = "http://localhost:1234/invocations"
    response = requests.post(endpoint, json=inference_request)
    print(response)
    
    #return df
    return response.text


@app.post("/files/")
async def create_file(file: bytes = File(...), token: str = Form(...)):
    s=str(file,'utf-8')
    data = StringIO(s)
    df=pd.read_csv(data)
    print(df)
    #return df
    return {
        "file": df,
        "token": token,
    }



