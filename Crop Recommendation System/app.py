from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler from the pickle files
with open("/Users/madhurabhagat/Desktop/Crop Production App/project/model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("home.html")  

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["ph"]),
            float(request.form["rainfall"])
        ]
        ss=StandardScaler()
        features = np.array(features).reshape(1, -1)
        
        # Scale the features using the loaded scaler
        features_scaled = ss.fit_transform(features)
        
        # Make prediction with the scaled features
        prediction = model.predict(features_scaled)[0]

        return render_template("home.html", prediction_text=f"Predicted Crop: {prediction}")

    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)