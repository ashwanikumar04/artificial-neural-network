
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

sc = joblib.load('scaler.pkl')

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from keras.models import load_model
classifier = load_model("model.h5")


import numpy as np
new_prediction = classifier.predict(sc.transform(np.array([[0.,0,600,1,40,3,60000,2,1,1,1000]])))
new_prediction=(new_prediction>0.5)
print("The customer will leave" if new_prediction else"The customer will not leave")