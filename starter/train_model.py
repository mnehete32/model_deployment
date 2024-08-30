# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from ml.data import process_data
from ml.model import *
import pandas as pd
import numpy as np



# Add the necessary imports for the starter code.

# Add code to load in the data.

data = pd.read_csv("data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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


X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


#np.save("test/testcases.npy", X_train[:32])


# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder= encoder, lb = lb
)

# Train and save a model.
model = train_model(X_train, y_train)
pred = model.predict(X_test)
precision, recall, fbeta = compute_model_metrics(y_test, pred)

print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("Fbeta: {}".format(fbeta))

with open('model/model.pkl','wb') as f:
    pickle.dump(model,f)

