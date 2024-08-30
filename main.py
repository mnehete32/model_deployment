from fastapi import FastAPI,Body

from pydantic import BaseModel
import os
import pandas as pd
import pickle

from starter.ml.model import inference
from starter.ml.data import process_data

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI()

with open ("model/lb.pkl", "rb") as f:
    lb = pickle.load(f)

with open ("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
class item(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

@app.post("/inference/")
async def predict(
    item : item =  Body(...,
                        example={
                                "age":  39,
                                "workclass":  "State-gov",
                                "fnlgt":  77516,
                                "education":  "Bachelors",
                                "education_num":  13,
                                "marital_status":  "Never-married",
                                "occupation":  "Adm-clerical",
                                "relationship":  "Not-in-family",
                                "race":  "White",
                                "sex":  "Male",
                                "capital_gain":  2174,
                                "capital_loss":  0,
                                "hours_per_week":  40,
                                "native_country":  "United-States"
                        },


    ),):
    body = {k.replace("_","-"): [v] for k,v in item.dict().items()}


    test = pd.DataFrame.from_dict(body)
    test, _, _, _ = process_data(
    test, categorical_features=cat_features, training=False, encoder= encoder, lb = lb)

    pred = inference(model, test)
    salary = lb.inverse_transform(pred)[0]
    return {"salary": str(salary)}