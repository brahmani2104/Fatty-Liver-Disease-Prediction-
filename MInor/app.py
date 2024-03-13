import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
#import joblib
#from tensorflow.keras.models import load_model




app = Flask(__name__)

#model = load_model('model.h5')

model = pickle.load(open('model3.pkl', 'rb'))
#model = joblib.load('model2.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    ## FOR NEURAL NETWORK
    #sequence_length = 10
    #int_features = [float(x) for x in request.form.values()]
    #final_features = np.array(int_features).reshape(1, sequence_length, X_train.shape[1])
    #prediction = model.predict(final_features)

    #output = round(prediction[0][0], 2)

    if output == 0:
        return render_template('index.html', prediction_text='There are no signs of Fatty Liver Disease')
    else:
        return render_template('index.html', prediction_text='There are signs of Fatty Liver Disease')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)