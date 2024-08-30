from fastapi.testclient import TestClient
from main import app
import pytest

import json

client = TestClient(app)

@pytest.fixture
def data_low_salary():
    data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
    }
    return data



#31,Private,352465,Some-college,10,Married-civ-spouse,Exec-managerial,Husband,White,Male,15024,0,50,United-States
@pytest.fixture
def data_high_salary():
    data = {
    "age": 31,
    "workclass": "Private",
    "fnlgt": 352465,
    "education": "Some-college",
    "education_num": 10,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 15024,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States"
    }
    return data



def test_root():
    res = client.get("/")
    assert res.status_code == 200

def test_root_content():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"greeting": "Hello World!"}

def test_low_salary_test(data_low_salary):
    res = client.post("/inferece/",data=json.dumps(data_low_salary))
    assert res.status_code == 200
    assert "salary" in res.json(), "salary key is not present in the return of the api"
    assert res.json()["salary"] == "<=50K", "api return salary {}, the expected result was <=50K".format(res.json()["salary"])


def test_high_salary_test(data_high_salary):
    res = client.post("/inferece/",data=json.dumps(data_high_salary))
    assert res.status_code == 200
    assert "salary" in res.json(), "salary key is not present in the return of the api"
    assert res.json()["salary"] == ">50K", "api return salary {}, the expected result was >50K".format(res.json()["salary"])