import pandas as pd
import pickle
import os
from model import compute_model_metrics

df = pd.read_csv("data/test.csv")

with open ("model/lb.pkl", "rb") as f:
    lb = pickle.load(f)


def slice_validate(df, feature):
    """ Function for calculating descriptive stats on slices of the Iris dataset."""

    precisions = recalls = fbetas = []
    # temp_df = out_df.copy()
    for cls in df[feature].unique():
        # row_df = out_df.copy()
        temp = df[df[feature] == cls]
        y_true = lb.transform(temp["salary"].values)
        y_pred = temp["prediction"].values

        precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
        precision = round(precision,4)
        recall = round(recall,4)
        fbeta = round(fbeta,4)
        # row_df["precision"],row_df["recall"],row_df["fbeta"] = precision, recall, fbeta
        # row_df["feature_name"] = feature
        # row_df["cat_of_feature"] = cls
        with open(out_filename, "a") as f:
            f.write(f"{feature} {cls} {precision} {recall} {fbeta}\n")



if "__main__" == __name__:
    out_df = pd.DataFrame(columns=[
        "feature_name",
        "cat_of_feature",
        "precision",
        "recall",
        "fbeta"
    ])

    out_filename = "data/slice_output.txt"
    

    if os.path.isfile(out_filename):
        os.remove(out_filename)
    out_df.to_csv(out_filename, index=None, sep=' ', mode='w')
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
    for cat in cat_features:
        slice_validate(df, cat)
        # cat_df.to_csv(out_filename, header=None, index=None, sep=' ', mode='a')