from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained models
with open('physical_isolation_model.pkl', 'rb') as file:
    physical_model = pickle.load(file)

with open('social_isolation_model.pkl', 'rb') as file:
    social_model = pickle.load(file)

with open('digital_isolation_model.pkl', 'rb') as file:
    digital_model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from the form
        input_values = [
            float(request.form['total_screen_time']),
            float(request.form['social_networking']),
            float(request.form['reading_and_research']),
            float(request.form['other']),
            float(request.form['productivity']),
            float(request.form['health_and_fitness']),
            float(request.form['entertainment']),
            float(request.form['creativity']),
            float(request.form['yoga']),
            float(request.form['movies']),
            float(request.form['gaming']),
            float(request.form['community_events']),
            float(request.form['family_time']),
            float(request.form['outdoor_activities']),
            float(request.form['volunteering']),
        ]

        # Convert the input into a numpy array for prediction
        input_array = np.array(input_values).reshape(1, -1)

        # Predict using the models
        physical_isolation = physical_model.predict(input_array)[0]
        social_isolation = social_model.predict(input_array)[0]
        digital_isolation = digital_model.predict(input_array)[0]

        # Pass predictions to the template
        return render_template('index.html', 
                               physical_isolation=physical_isolation,
                               social_isolation=social_isolation,
                               digital_isolation=digital_isolation)
    except Exception as e:
        # In case of error, return default or error message
        return render_template('index.html', 
                               physical_isolation=None,
                               social_isolation=None,
                               digital_isolation=None, 
                               error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
