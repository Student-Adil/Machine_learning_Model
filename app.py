import numpy as np
from flask import Flask,request, render_template, jsonify
import pickle
app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))
@app.route('/')
def home():
    return render_template('index.html')  
@app.route('/predict',methods=['POST'])
def predict():
    major=request.form['major']
    minor=request.form['minor']
    cgpa=float(request.form['cgpa'])
    final_features=np.array([[major,minor,cgpa]])
    prediction=model.predict(final_features)
    output=prediction[0]
    return render_template('index.html',prediction_text='Predicted Course is {}'.format(output))
if __name__=="__main__":
    app.run(debug=True)
