import os
import sys
import pytest
import pickle
import pandas as pd
from flask import Flask,jsonify,request

# Adjust the path to ensure app and model can be imported
directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(directory, ".."))

from app import app, predict, model  # Ensure this import works

# Define the directory for current files
current_directory = os.getcwd()

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@pytest.fixture
def test_data():
    # Determine the path of the CSV file
    #path = os.path.join("dl.csv")
    data = pd.read_csv("dl.csv")
    # Verify that the DataFrame is not empty
    assert not data.empty, "Error loading data."
    return data

def test_model():
    # Determine the path of the file containing the trained model
    model_path = os.path.join(current_directory, "best_model.pkl")
    # Load the model from the file
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    # Verify that the model has been loaded correctly
    assert model is not None, "Error loading the model."

def test_predict_1(client, test_data):
    response = client.get('/predict?id=100001')
    assert response.status_code == 200
    response_data = response.get_json()
    assert 'probabilite' in response_data
    assert 'Pret' in response_data
    assert response_data['Pret'] in ['Accepté', 'Refusé']