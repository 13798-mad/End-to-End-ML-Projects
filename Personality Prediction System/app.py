from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        features = [
            data['Time_spent_Alone'],
            1 if data['Stage_fear'].lower() == "yes" else 0,
            data['Social_event_attendance'],
            data['Going_outside'],
            1 if data['Drained_after_socializing'].lower() == "yes" else 0,
            data['Friends_circle_size'],
            data['Post_frequency']
        ]
        prediction = model.predict([features])[0]
        result = "Introvert" if prediction == 1 else "Extrovert"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
