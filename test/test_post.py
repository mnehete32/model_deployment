import requests
import json

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

low_salary_test = requests.post("https://my-udacity-mlops-app-0caba5ffe330.herokuapp.com/inference", data=json.dumps(data))
print("Result: {}".format(low_salary_test.json()))
print("Status Code: {}".format(low_salary_test.status_code))
assert "salary" in low_salary_test.json()
assert low_salary_test.json()["salary"] == "<=50K"



#31,Private,352465,Some-college,10,Married-civ-spouse,Exec-managerial,Husband,White,Male,15024,0,50,United-States
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

high_salary_test = requests.post("https://my-udacity-mlops-app-0caba5ffe330.herokuapp.com/inference", data=json.dumps(data))
print("Result: {}".format(high_salary_test.json()))
print("Status Code: {}".format(high_salary_test.status_code))
assert "salary" in high_salary_test.json()
assert high_salary_test.json()["salary"] == ">50K"