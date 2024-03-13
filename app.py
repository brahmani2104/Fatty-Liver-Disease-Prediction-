import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

logistic_model = pickle.load(open('LR.pkl', 'rb'))
svm_model = pickle.load(open('SVC.pkl', 'rb'))
random_forest = pickle.load(open('RF.pkl', 'rb'))
hybrid_model = pickle.load(open('hybrid.pkl', 'rb'))


selected_model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(request.form[f'field{i}']) for i in range(1, 12)]
    selected_model = request.form.get('selected_model')
    accuracy = 0

    if selected_model is None:
        return render_template('index.html', prediction_text='Please select a model first.', accuracy_text='0%')

  
    final_features = [np.array(int_features)]

    if selected_model == 'Model 1':
        model = logistic_model
        accuracy = '94.92%'
    elif selected_model == 'Model 2':
        model = svm_model
        accuracy = '92.37%'
    elif selected_model == 'Model 3':
        model = random_forest
        accuracy = '96.81%' 
    else :
        model = hybrid_model
        accuracy = '98.45%'
    
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    if output == 0:
        return render_template('index.html', prediction_text='NO FLD' , Accuracy_test = accuracy)
    else:
        return render_template('index.html', prediction_text='FLD', Accuracy_test = accuracy)
    

if __name__ == "__main__":
    app.run(debug=True)