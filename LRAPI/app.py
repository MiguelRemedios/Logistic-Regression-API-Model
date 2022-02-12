import numpy as np
from flask import Flask, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
model2 = pickle.load(open("modelTest.pkl", "rb"))


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/analyse', methods=["POST"])
def analyse():
    request_data = request.get_json()
    gender = request_data['gender']

    if gender == "Male":
        gender = 0
    else:
        gender = 1

    age = request_data['age']
    estimated_salary = request_data['estimatedSalary']
    features = [gender, age, estimated_salary]
    matrix = [np.array(features)]
    print(matrix)
    prediction = model.predict(matrix)
    print(prediction)

    return '''
               The gender value is: {}
               The age value is: {}
               The estimatedSalary version is: {}
               Prediction: {}'''.format(gender, age, estimated_salary, prediction)


@app.route('/analyse2', methods=["POST"])
def analyse2():
    request_data = request.get_json()
    age = request_data['age']
    features = [age]
    matrix = [np.array(features)]
    print(matrix)
    prediction = model2.predict(matrix)
    return '''
               The age value is: {}
               Prediction: {}'''.format(age, prediction)


if __name__ == '__main__':
    app.run()
