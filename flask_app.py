import json
import os
import numpy
from flask import Flask,jsonify,request
from flask_cors import CORS
from predictor import my_performance_predictor
app = Flask(__name__)
CORS(app)
@app.route("/pred",methods=["GET"])
def pred():
	reason = 'home'
	reason = numpy.dtype('object')
	traveltime = numpy.int64(1)
	studytime = numpy.int64(1)
	failures =  numpy.int64(0)
	internet =  'yes'
	internet = numpy.dtype('object')
	freetime = numpy.int64(4)
	goout = numpy.int64(3)
	health = numpy.int64(5)
	absences = numpy.int64(4)
	G1 = numpy.int64(12)
	G2 = numpy.int64(12)
	predicition = my_performance_predictor(reason,traveltime,studytime,failures,internet,freetime,goout,health,absences,G1,G2)
	price_dict = {
	'model':'MLV',
	'predicition': predicition.predict(),
	}
	return jsonify(price_dict)

@app.route("/")
def home():
	return "<a href = 'http://127.0.0.1:5000/pred'>Go to predictor</a>"

if __name__ == "__main__":
	app.run(debug=True)