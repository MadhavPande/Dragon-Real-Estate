import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import joblib

app = Flask(__name__)
model = joblib.load('app/Dragon.joblib')


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('app/templates/home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction[0])

    #output = round(prediction[0], 2)
    return render_template('app/templates/home.html', prediction_text="MEDV {}".format(prediction[0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



def run():
    app.run()