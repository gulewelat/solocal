import time
import pandas as pd
import numpy as np
import pickle

class Predict:
    
    def __init__(self, data):
        
        self.X_test = data
        self.pickled_model = pickle.load(open('model.pkl', 'rb'))
        self.prediction = self.pickled_model.predict(self.X_test)
        
        print('The prediction is {}'.format(self.prediction))
