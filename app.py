import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('finalized_model.pickle', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route("/")
def home():
    return  render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(scaler.transform(final_features))

        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text='Chance of Admission is  {:.0%}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)