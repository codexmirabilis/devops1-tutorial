import pickle
import pandas as pd
import predict

def func(x):
    return x + 2

def test_answer():
    assert func(3) == 5

def test_score():
    file_to_open = open('data/models/linear_regressor.pickle', 'rb')
    trained_model = pickle.load(file_to_open)
    file_to_open.close()

    # Load data that we want predictions for
    prediction_data = pd.read_csv('data/prediction-data.csv', sep=';')

    #print(trained_model.predict(prediction_data))
    y_test = [14.00, 15.00, 14.00, 24.00, 22.00, 18.00, 21.00, 27.00, 26.00, 25.00, 24.00]
    assert trained_model.score(prediction_data, y_test) > 0.75
