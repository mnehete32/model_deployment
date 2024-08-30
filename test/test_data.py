import pandas as pd
import numpy as np
from starter.ml.model import compute_model_metrics,inference

def test_data_shape(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."

def test_all_columns_are_present_in_data(data):
    columns_names = ['age', 'workclass', 'fnlgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
    for col in columns_names:
        assert col in data, "Column {col} is not found in the data".format(col = col)

def test_model_return_values_range(model, test_data):
    "for some example test if the model return values which are between 0 and 1"

    pred = model.predict(test_data)
    assert pd.Series(pred).between(0,1).all()

def test_model_accuracy_metrics_on_testcases_data(model, test_data):
    test_data = np.load("test/testcases.npy")
    expected_labels = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
    pred = model.predict(test_data)
    precision, recall, fbeta = compute_model_metrics(expected_labels,pred)

    assert 0.5 < precision, "Precision {precision} is too low".format(precision= precision)
    assert precision <= 1.0, "Precision {precision} is greater than 1".format(precision= precision)

    assert 0.5 < recall, "Recall {recall} is too low".format(recall= recall)
    assert recall <= 1.0, "Recall {recall} is greater than 1".format(recall= recall)

    assert 0.5 < fbeta, "fbeta {fbeta} is too low".format(fbeta= fbeta)
    assert fbeta <= 1.0, "fbeta {fbeta} is greater than 1".format(fbeta= fbeta)
    assert (pred == expected_labels).all()

def test_inference_function_returns_int64(model,test_data):
    pred = inference(model, test_data[0].reshape(1, -1))
    print("-"*20, type(pred))
    assert isinstance(pred, np.int64), "Inference funtion returned {pred} which is not Int64".format(pred = pred)
    

    
    

    


