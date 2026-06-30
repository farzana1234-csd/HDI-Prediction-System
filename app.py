from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('hdi_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        life_expectancy = float(request.form['life_expectancy'])
        expected_schooling = float(request.form['expected_schooling'])
        mean_schooling = float(request.form['mean_schooling'])
        gni = float(request.form['gni'])

        features = np.array([[life_expectancy, expected_schooling, mean_schooling, gni]])
        prediction = model.predict(features)[0]
        result = round(prediction, 4)

        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)