from flask import Flask, render_template, request, jsonify
import pickle

# Load the saved model and vectorizer
with open('spam_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the JSON data from the request
        data = request.get_json()
        email_text = data['email']
        
        # Transform input text
        email_vectorized = vectorizer.transform([email_text])
        
        # Predict using the model
        prediction = model.predict(email_vectorized)
        
        # Map prediction to human-readable label
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        
        return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
