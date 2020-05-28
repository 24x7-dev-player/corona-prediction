import numpy as np
from flask import Flask, request, render_template, url_for
import pickle
import joblib

app = Flask(__name__)
# model = joblib.load("/Users/tushararora/Documents/Corona detector/Corona_detector.pkl")
model = pickle.load(open('model.pkl','rb'))
# print(model.predict([['102,1,22,0,1']]))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    value = np.array(input_features)
    output = model.predict_proba([value])[0][1]
    print(output)
    return render_template('index.html', Prediction_text = f"Patients probabilty of infection is {round(output*100)}%.")
if __name__ == "__main__":
    app.run(debug=True)
