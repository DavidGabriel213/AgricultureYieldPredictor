import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/storage/emulated/0/Download/nigerian_crop_yield_messy.csv")
df=df.drop_duplicates()
df["FarmID"]=df["FarmID"].astype(str).str.strip()
df["State"]=df["State"].astype(str).str.strip()
df["Crop"]=df["Crop"].astype(str).str.strip()
#Season
df["Season"]=df["Season"].astype(str).str.capitalize().str.strip()
Season_corrector={"Wet season":"Rainy Season","Rainy":"Rainy Season","Harmattan":"Dry Season","Dry":"Dry Season","Wet":"Rainy Season","Dry season":"Dry Season"}
df["Season"]=df["Season"].replace(Season_corrector)
#Soil Type
df["SoilType"]=df["SoilType"].astype(str).str.strip().str.capitalize()
SoilType_corrector={"C":"Clay","L":"Loamy","S":"Sandy","Silty":"Silt"}
df["SoilType"]=df["SoilType"].replace(SoilType_corrector)
#Farming Method
df["FarmingMethod"]=df["FarmingMethod"].astype(str).str.capitalize().str.strip()
FarmingMethod_corrector={"Rainfed":"Rain-fed","Irrigated":"Irrigation"}
df["FarmingMethod"]=df["FarmingMethod"].replace(FarmingMethod_corrector)
df["FarmingMethod"]=df["FarmingMethod"].apply(lambda x: "Rain-Fed" if "fed" in x else x)
#Irrigation
df["IrrigationType"]=df["IrrigationType"].astype(str).str.capitalize().str.strip()
IrrigationType_corrector={"Nan":np.nan,"Nil":np.nan}
df["IrrigationType"]=df["IrrigationType"].replace(IrrigationType_corrector)
df["IrrigationType"]=df["IrrigationType"].fillna(df.groupby(["State","Season"])["IrrigationType"].transform(lambda x: x.mode()[0]))
#FertizerUsed, PesticideUsed and Improved seed
for c in ["FertilizerUsed","PesticideUsed","ImprovedSeed"]:
    df[c]=df[c].astype(str).str. capitalize().str.strip()
    corrector={"1":"Yes","0":"No","Y":"Yes","N":"No","False":"No","True":"Yes"}
    df[c]=df[c].replace(corrector)
#Rainfall, Farmsize and Temperature
df["FarmSize(Hectares)"]=df["FarmSize(Hectares)"].astype(str).str.replace("-","").str.replace("ha","").str.strip()
def Accres_Hectares(c):
    if "acres" in c:
        k=float(c.replace("acres",""))
        return k*0.4047
    else:
        return c
df["FarmSize(Hectares)"]=df["FarmSize(Hectares)"].apply(lambda x: Accres_Hectares(x))
df["Rainfall(mm)"]=df["Rainfall(mm)"].astype(str).str.replace("-","").str.replace("mm","").str.strip()
df["Temperature(°C)"]=df["Temperature(°C)"].astype(str).str.replace("-","").str.replace("°","").str.replace("degees","").str.replace("C","").str.strip()
for c in ["FarmSize(Hectares)","Rainfall(mm)","Temperature(°C)"]:
    df[c]=pd.to_numeric(df[c],errors="coerce")
    max1=df[c].quantile(0.75)+1.5*(df[c].quantile(0.75)-df[c].quantile(0.25))
    min1=df[c].quantile(0.25)-1.5*(df[c].quantile(0.75)-df[c].quantile(0.25))
    df[c]=df[c].clip(min1,max1)
df["FarmSize(Hectares)"]=df["FarmSize(Hectares)"].fillna(df.groupby(["State","Crop"])["FarmSize(Hectares)"].transform("mean"))
for c in ["Rainfall(mm)","Temperature(°C)"]:
    df[c]=df[c].fillna(df.groupby(["State","Season"])[c].transform("mean"))
df["FarmSize(Hectares)"]=df["FarmSize(Hectares)"].round(2)
df["Rainfall(mm)"]=df["Rainfall(mm)"].round(2)
df["Temperature(°C)"]=df["Temperature(°C)"].round(2)
#Humidity
df["Humidity(%)"]=df["Humidity(%)"].astype(str).str.replace("%","").str.replace("percent","").str.replace("-","").str.strip()
df["Humidity(%)"]=pd.to_numeric(df["Humidity(%)"],errors="coerce")
df["Humidity(%)"]=df["Humidity(%)"].clip(20,99)
df["Humidity(%)"]=df["Humidity(%)"].fillna(df.groupby(["State","Season"])["Humidity(%)"].transform("mean"))
df["Humidity(%)"]=df["Humidity(%)"].round(1)
#SoilPH,Fertilizer(kg/ha) and Pesticide(L/ha)
for c in ["SoilPH","Fertilizer(kg/ha)","Pesticide(L/ha)"]:
    df[c]=df[c].astype(str)
    for r in ["pH","-","kg","/ha"]:
        df[c]=df[c].str.replace(r,"")
    df[c]=df[c].str.strip()
    df[c]=pd.to_numeric(df[c],errors="coerce")
    max1=df[c].quantile(0.75)+1.5*(df[c].quantile(0.75)-df[c].quantile(0.25))
    min1=df[c].quantile(0.25)-1.5*(df[c].quantile(0.75)-df[c].quantile(0.25))
    df[c]=df[c].clip(min1,max1)
df["SoilPH"]=df["SoilPH"].fillna(df.groupby(["State","SoilType","Season"])["SoilPH"].transform("mean"))
df["Fertilizer(kg/ha)"]=df["Fertilizer(kg/ha)"].fillna(df.groupby(["SoilType","FertilizerUsed"])["Fertilizer(kg/ha)"].transform("mean"))
for c in ["SoilPH","Fertilizer(kg/ha)"]:
    df[c]=(df[c]).round(2)
#Farm Experience
df["FarmExperience(Years)"]=np.abs(df["FarmExperience(Years)"])
#Num of workers
df["NumWorkers"]=np.abs(df["NumWorkers"])
#Distance to market
df["DistanceToMarket(km)"]=df["DistanceToMarket(km)"].astype(str).str.replace("km","").str.replace("-","").str.strip()
def distance_corrector(c):
        if "miles" in c:
            k=float(c.replace("miles",""))
            return k*1.609
        else:
            return c
df["DistanceToMarket(km)"]=df["DistanceToMarket(km)"].apply(lambda x: distance_corrector(x))
df["DistanceToMarket(km)"]=pd.to_numeric(df["DistanceToMarket(km)"],errors="coerce")
df["DistanceToMarket(km)"]=(df["DistanceToMarket(km)"].fillna(df.groupby(["State","Crop"])["DistanceToMarket(km)"].transform("mean"))).round(2)
#PreviousYield and ActualYield
for c in ["PreviousYield(tons/ha)","ActualYield(tons/ha)"]:
    df[c]=df[c].astype(str)
    for r in ["-","t/ha","tons"]:       
        df[c]=df[c].str.replace(r,"")
    df[c]=df[c].str.strip()
    df[c]=pd.to_numeric(df[c],errors="coerce")
    max1=df[c].quantile(0.75)+1.5*(df[c].quantile(0.75)-df[c].quantile(0.25))
    min1=df[c].quantile(0.25)-1.5*(df[c].quantile(0.75)-df[c].quantile(0.25))
    df[c]=df[c].clip(min1,max1)
df["PreviousYield(tons/ha)"]=df["PreviousYield(tons/ha)"].fillna(df.groupby(["SoilType","FarmingMethod","FertilizerUsed"])["PreviousYield(tons/ha)"].transform("mean"))
df["PreviousYield(tons/ha)"]=df["PreviousYield(tons/ha)"].round(2)
df["ActualYield(tons/ha)"]=df["ActualYield(tons/ha)"].fillna(df.groupby(["SoilType","FarmingMethod","FertilizerUsed","ImprovedSeed","Crop"])["ActualYield(tons/ha)"].transform("mean"))
df["ActualYield(tons/ha)"]=df["ActualYield(tons/ha)"].round(2)
#Planting density
df["PlantingDensity(plants/ha)"]=np.abs(df["PlantingDensity(plants/ha)"])

df["YieldCategory"]=df["YieldCategory"].astype(str).str.capitalize().str.strip()
YieldCategory_corrector={"L":"Low","Poor":"Low","Below average":"Low","Average":"Medium","Moderate":"Medium","H":"High","Excellent":"High","M":"Medium","Very poor":"Very low","Vl":"Very low","Minimal":"Very low"}
df["YieldCategory"]=df["YieldCategory"].replace(YieldCategory_corrector)
df.to_csv("Farm_Yield(cleaned data).csv",index=False)
#EDA's
fig,ax=plt.subplots(3,3,figsize=(9,9))
#yield by crop and state
state_crop=(df.groupby(["State","Crop"])["ActualYield(tons/ha)"].mean()).round(2).unstack()
state_crop.plot(kind="bar",ax=ax[0,0])
ax[0,0].set_ylabel("ActualYield")
ax[0,0].set_title("AverageYield_State&Crop")
#yield by crop and soil type
soil_crop=(df.groupby(["SoilType","Crop"])["ActualYield(tons/ha)"].mean()).round(2).unstack()
soil_crop.plot(kind="bar",ax=ax[0,1])
ax[0,1].set_ylabel("ActualYield")
ax[0,1].set_title("AverageYield_SoilType&Crop")
#yield by crop and fertilizerused
crop_fertilizer=(df.groupby(["FertilizerUsed","Crop"])["ActualYield(tons/ha)"].mean()).round(2).unstack()
crop_fertilizer.plot(kind="bar",ax=ax[0,2])
ax[0,2].set_ylabel("ActualYield")
ax[0,2].set_title("AverageYield_FertilizerUsed&Crop")
#CreditAccess_yield
creditacces_yield=(df.groupby("CreditAccess")["ActualYield(tons/ha)"].mean()).round(2)
creditacces_yield.plot(kind="bar",ax=ax[1,0])
ax[1,0].set_ylabel("ActualYield")
ax[1,0].set_title("AverageYield_Used&Crop")

#Feature Engineering
for c in ["Rainfall(mm)","SoilPH","Fertilizer(kg/ha)","Pesticide(L/ha)","DistanceToMarket(km)","PlantingDensity(plants/ha)","ActualYield(tons/ha)","FarmSize(Hectares)"]:
    df[c+"_log"]=(np.log1p(df[c])).round(3)
    df=df.drop(columns=[c])
df=df.drop(columns=["FarmID"])
df["FarmAccessComfort"]=(df["FarmSize(Hectares)_log"]/df["DistanceToMarket(km)_log"]).round(3)
df["EnvironmentComfort"]=(df["Temperature(°C)"]/df["Humidity(%)"]).round(3)
df["SoilIndex"]=(df["Rainfall(mm)_log"]/(df["Fertilizer(kg/ha)_log"]+df["SoilPH_log"])).round(3)
df["IncreaseRate"]=(df["PreviousYield(tons/ha)"]/df["NumWorkers"]).round(3)
df.to_csv("FarmYield(Engineering).csv",index=False)
import seaborn as sns
sns.heatmap(df.corr(numeric_only=True).round(2),annot=True)
print(df.corr(numeric_only=True))
plt.show()
