import numpy as np
from flask import Flask, request
import joblib
import pandas as pd
import datetime
import os

app = Flask(__name__)

def get_model_path(model_folder="eve_models" , district="labuduwa"):
    prefix = ""
    suffix = ".pkl"

    if(model_folder=="eve_models"):
        prefix = ""+model_folder+"/eve_"
    elif(model_folder=="temp_models"):
        prefix = ""+model_folder+"/temp_"
    else:
        print("|||||||  None of the Conditions were Matched  |||||||||")
        prefix = ""+model_folder+"eve_"

    model_path = prefix+ district.capitalize()+suffix

    return model_path


# --------------------------------------- FUNCTIONS ---------------------------------------
def get_avg_for_next_week(model_folder="eve_models", district="kalpitiya"):
    
    model_path = get_model_path(
        model_folder=model_folder,
        district=district
    )

    loaded_model = joblib.load(model_path)
    print("------ Model Loaded SuccessFully --------")

    now = datetime.datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    date_list = pd.date_range(start=current_date, periods=7, freq='D')
    df = pd.DataFrame({'date': date_list})
    df = df.set_index('date')
    df['dayofyear'] = df.index.dayofyear
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year

    print("------ Feature DataFrame Creation Completed --------")

    #Getting the average 
    res = loaded_model.predict(df)
    arr_sum = sum(res)
    arr_len = len(res)
    arr_avg = arr_sum / arr_len

    return str(arr_avg)



# --------------------------------------- ENDPOINTS ---------------------------------------
@app.route('/')
def home():
    return "Hellow World"+joblib.__version__+""

@app.route('/check_model_path' , methods=['POST'])
def check_path():
    pdistrict = request.form['district']
    return get_model_path(
        model_folder="eve_models",
        district=pdistrict
    )



@app.route('/forecast-eve', methods=['POST'])
def forecast_eveporation():
    
    p_district = request.form['district']
    print("Obtaining Results for {} District".format(p_district))

    eve_results = get_avg_for_next_week(model_folder="eve_models" , district=p_district)

    return {"eveporation" : eve_results}



if __name__ == '__main__':
    app.run(debug=True)
