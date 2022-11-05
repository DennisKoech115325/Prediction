import joblib
import pandas as pd
import numpy as np
import pickle
#from sklearn.neural_network import MLPRegressor

class my_performance_predictor():
  def __init__(self, reason ,traveltime ,studytime ,failures ,internet ,freetime ,goout ,health ,absences ,G1 ,G2):
    self.reason = reason
    self.traveltime = traveltime
    self.studytime = studytime
    self.failures =  failures
    self.internet =  internet
    self.freetime = freetime
    self.goout = goout
    self.health = health
    self.absences = absences
    self.G1 = G1
    self.G2 = G2
  def deserialize(self):
    filename = 'JL_Model.pkl'
    model = joblib.load(filename)
    return model

  def predict(self):
    #predset = dataPrepper(self.reason,self.traveltime,self.studytime,self.failures,self.internet,self.freetime,self.goout,self.health,self.absences,self.G1,self.G2)
    model = self.deserialize()
    #return model.predict([[self.reason,self.traveltime,self.studytime,self.failures,self.internet,self.freetime,self.goout,self.health,self.absences,self.G1,self.G2]])
    test = [[0, 1, 0, 0, 0, 1, 4, 5, 2, 9, 0, 9, 2, 2, 2]]
    np_array = np.array(test)
    return model.predict(np_array.reshape(1,-1)).tolist()