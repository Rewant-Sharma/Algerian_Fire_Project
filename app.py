#this app provides the hosting service for the model which can be visualised on a browser.
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from flask import Flask,request,jsonify,render_template

#how to deploy the model
app = Flask(__name__)

#import ridge and regressor files.

ridge_model = pickle.load(open('models/ridge_reg.pkl','rb')) #rb stands for read byte , we are decrypting our code here
scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')## if someone type the url and oput forward slash  it will open index page
def index():
    return render_template('index.html') #convert the html to form 

@app.route('/predictdata', methods=['Get', 'Post']) #when user is going to submit the button the page /predict data will open in URL,the get methord takes the value from html form and after calculation posts the data
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_sc = scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]]) #we are getting string values and then we standardise the given data and transform it
        result = ridge_model.predict(new_data_sc) # we predict on the transforemed data

        return render_template('index.html', result=result[0]) # why zero?, because we only want the first column of the array
    else:
        return render_template('index.html') #even if we dont get the result juist return the index.html


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)