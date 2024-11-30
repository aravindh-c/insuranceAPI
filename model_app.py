# importing libraires
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib

app = FastAPI()

class Input(BaseModel):
    Gender: object
    Age: int
    Driving_License: int
    Region_Code: float
    Previously_Insured: int
    Vehicle_Age: object
    Vehicle_Damage: object
    Annual_Premium: float
    Policy_Sales_Channel : float
    Vintage: int




class Output(BaseModel):
    Response: int

@app.post("/predict")
def predict(data: Input) -> Output:
    X_input = pd.DataFrame([[data.Gender,data.Age,data.Driving_License,data.Region_Code,
              data.Previously_Insured,data.Vehicle_Age,data.Vehicle_Damage,data.Annual_Premium,
              data.Policy_Sales_Channel,data.Vintage]])

    X_input.columns = ['Gender','Age','Driving_License','Region_Code','Previously_Insured',
                       'Vehicle_Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel','Vintage']

    #load model
    model = joblib.load('insurance_model_pipeline_hyper1.pkl')

    #predict
    prediction = model.predict(X_input)

    #output
    return Output(Response = prediction)
