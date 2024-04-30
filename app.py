from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('multi_label_rf_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    time = int(request.form['time'])
    score = int(request.form['score'])
    word_count = int(request.form['word_count'])
    
    # Make a prediction using the loaded model
    sample_data = pd.DataFrame({
        'Time (seconds)': [time],
        'Score': [score],
        'Word Count': [word_count]
    })
    predicted_categories = model.predict(sample_data)
    
    # Convert predicted categories to human-readable format
    categories = []
    if predicted_categories[0][0] == 1:
        categories.append("Fruits")
    if predicted_categories[0][1] == 1:
        categories.append("Vegetables")
    if predicted_categories[0][2] == 1:
        categories.append("Animals")
    if predicted_categories[0][3] == 1:
        categories.append("Commands")
    
    # Render the result template with predicted categories
    return render_template('result.html', categories=categories)

if __name__ == '__main__':
    app.run(debug=True)
