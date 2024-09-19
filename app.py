from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the pre-trained models
model = joblib.load("./models/knn_model.lb")  # KNeighborsClassifier model
model2 = joblib.load("./models/std_scaler.lb")  # StandardScaler model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "POST":
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bloodPressure = int(request.form['bloodPressure'])
        skinThickness = int(request.form['skinThickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dbf = float(request.form['dbf'])  
        age = int(request.form['age'])
        
        data = np.array([[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, dbf, age]])
   
        scaled_data = model2.transform(data)
        print(scaled_data)

        prediction = model.predict(scaled_data)
        
        result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
        
        return render_template('home.html', prediction_text= result)

if __name__ == "__main__":
    app.run(debug=True)
