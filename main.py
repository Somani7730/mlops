from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('breast_cancer_model_selected_features.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form
        try:
            features = [float(x) for x in request.form.values()]
            features_array = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features_array)
            result = 'Cancer Detected' if prediction[0] == 1 else 'No Cancer Detected'
            return render_template('index.html', prediction=result)
        except ValueError:
            return render_template('index.html', error="Invalid input! Please ensure all fields are numeric.")

if __name__ == '__main__':
    app.run(debug=True)
