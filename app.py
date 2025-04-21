import pickle
import numpy as np
from flask import Flask, request, render_template
import nltk
from nltk.corpus import stopwords

# Download stopwords (only once)
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    
    # Preprocess
    message_transformed = vectorizer.transform([message])
    
    # Predict
    prediction = model.predict(message_transformed)
    
    result = 'SPAM' if prediction[0] == 1 else 'NOT SPAM'
    
    return render_template('index.html', prediction_text=f'Message is {result}')

if __name__ == "__main__":
    app.run(debug=True)
