from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import joblib


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('indexx.html')


def word_to_int(word):
    if word == 'Male':
        return 1
    elif word == 'Female':
        return 0
    elif word == 'Other':
        return 2
    elif word == 'No':
        return 0
    elif word == 'Yes':
        return 1
    elif word == 'Rural':
        return 0
    elif word == 'Urban':
        return 1
    elif word == 'Govt':
        return 0
    elif word == 'Unemployed':
        return 1
    elif word == 'Private':
        return 2
    elif word == 'Self Employed':
        return 3
    elif word == 'Student':
        return 4
    elif word == 'None':
        return 0
    elif word == 'Former Smoker':
        return 1
    elif word == 'Never':
        return 2
    else:
        return 3


@app.route('/predict',methods=['POST'])

def predict():
    '''
    For rendering results on HTML GUI
    '''

    if request.method == "POST":


        to_predict_list = request.form.values()

        to_predict_list = list(map(float, word_to_int(to_predict_list)))

    prediction = model.predict(np.array(to_predict_list).reshape(1,1))

    output = prediction
    return render_template('indexx.html', Result='Stroke chances are 0:No 1:Yes - {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

