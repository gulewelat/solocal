from utils import *
import time
import pandas as pd
import numpy as np
import pickle

start_time = time.time()
test_data = pd.read_json("test")
predict = Predict(test_data)
print("%s seconds of running time" % (time.time() - start_time))