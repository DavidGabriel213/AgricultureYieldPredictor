from flask import Flask, render_template,request
import numpy as np
import pandas as pd
import joblib
import os

app=Flask(__name__)
processor_cat=joblib.load('preprocessor/processor_cat.joblib')
processor_reg=joblib.load('preprocessor/processor_reg.joblib')
cat_model=joblib.load('models/XGB_model.joblib')
reg_model=joblib.load('models/LinearRegresModel.joblib')
@app.route('/',methods=['GET','POST'])
def myfunc():
    predicted_yield=None
    Category=None
    cat_class=None
    if request.method=="POST":
        # numerical cols
        Temperature=float(request.form['temperature'])
        Humidity=float(request.form['humidity'])
        FarmExperience=float(request.form['experience'])
        NumWorkers=float(request.form['num_workers'])
        PreviousYield=float(request.form['prev_yield'])
        Rainfall=float(request.form['rainfall'])
        SoilPH=float(request.form['soil_PH'])
        Fertilizer=float(request.form['fertilizer'])
        Pesticide=float(request.form['pesticide'])
        DistanceToMarket=float(request.form['distance'])
        PlantingDensity=float(request.form['planting_density'])
        FarmSize=float(request.form['farm_size'])
        # categorical cols
        State=request.form['state']
        Crop=request.form['crop']
        Season=request.form['season']
        SoilType=request.form['soil_type']
        FarmingMethod=request.form['farming_method']
        IrrigationType=request.form['irrigation_type']
        # binary columns
        FertilizerUsed=float(request.form['fertilizer_used'])
        PesticideUsed=float(request.form['pesticide_used'])
        ImprovedSeed=float(request.form['improved_seed'])
        CreditAccess=float(request.form['credit_access'])
        ExtensionServices=float(request.form['extension_services'])
        # features engineered
        d={"Rainfall":Rainfall,"SoilPH":SoilPH,"Fertilizer":Fertilizer,"Pesticide":Pesticide,
           "DistanceToMarket":DistanceToMarket,"PlantingDensity":PlantingDensity, "FarmSize":FarmSize}
        for c in d:
            d[c]=np.log1p(d[c])
        FarmAccessComfort=(d['FarmSize']/d['DistanceToMarket'])
        EnvironmentComfort=(Temperature/Humidity)
        SoilIndex=(d['Rainfall']/(d['Fertilizer']+d['SoilPH']))
        IncreaseRate=(PreviousYield/NumWorkers)
        # features
        features=pd.DataFrame({"State":[State],"Crop":[Crop],"Season":[Season],"SoilType":[SoilType],
                               "FarmingMethod":[FarmingMethod],"IrrigationType":[IrrigationType],
                               "Temperature(°C)":[Temperature],"Humidity(%)":[Humidity],
                               "FarmExperience(Years)":[FarmExperience],"NumWorkers":[NumWorkers],
                               "PreviousYield(tons/ha)":[PreviousYield],"Rainfall(mm)_log":[Rainfall],
                               "SoilPH_log":[SoilPH],"Fertilizer(kg/ha)_log":[Fertilizer],
                               "Pesticide(L/ha)_log":[Pesticide],"DistanceToMarket(km)_log":[DistanceToMarket],
                               "PlantingDensity(plants/ha)_log":[PlantingDensity],"FarmSize(Hectares)_log":[FarmSize],
                               "FarmAccessComfort":[FarmAccessComfort],"EnvironmentComfort":[EnvironmentComfort],
                               "SoilIndex":[SoilIndex],"IncreaseRate":[IncreaseRate],
                               "FertilizerUsed":[FertilizerUsed],"PesticideUsed":[PesticideUsed],"ImprovedSeed":[ImprovedSeed],
                               "CreditAccess":[CreditAccess],"ExtensionServices":[ExtensionServices]})
        # features processing
        features_cat=processor_cat.transform(features)
        features_reg=processor_reg.transform(features)
        #predictions
        predicted_category=cat_model.predict(features_cat)
        y_reg_log=reg_model.predict(features_reg)
        # yield
        predicted_yield=np.expm1(y_reg_log)[0].round(2)
        # category
        if predicted_category==0:
            Category='High'
        elif predicted_category==1:
            Category='Low'
        elif predicted_category==2:
            Category='Medium'
        else:
            Category='Very low'
        class_map={
            'High':'high',
            'Medium': 'medium',
            'Low': 'low',
            'Very low':'verylow'
               }  
        cat_class=class_map.get(Category,'')
    return render_template('Crop_yield.html', Category=Category, predicted_yield=predicted_yield, cat_class=cat_class)
if __name__==('__main__'):
    port =int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=True)    
