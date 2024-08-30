import pytest
import pandas as pd
import numpy as np
import pickle
@pytest.fixture
def data():
    """ Simple function to generate some fake Pandas data."""
    df = pd.read_csv("data/census.csv")
    return df

@pytest.fixture
def model():
    """ load model"""
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@pytest.fixture
def test_data():
    test_data = np.load("test/testcases.npy")
    return test_data