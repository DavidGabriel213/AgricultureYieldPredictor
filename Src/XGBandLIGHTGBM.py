import numpy as np
import pandas as pd
import joblib
import seaborn as sns 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report

df=pd.read_csv('Data(cleaned and raw)/FarmYield(Engineering).csv')
processor_reg=joblib.load('preprocessor/processor_reg.joblib')
processor_cat=joblib.load('preprocessor/processor_cat.joblib')
print('processors loaded!')

le=LabelEncoder()
for c in ["FertilizerUsed","PesticideUsed","ImprovedSeed"]:
    df[c]=le.fit_transform(df[c])
#binary columns
bin_cols=["FertilizerUsed","PesticideUsed","ImprovedSeed","CreditAccess","ExtensionServices"]
#Categorical columns
cat_cols=["State","Crop","Season","SoilType","FarmingMethod","IrrigationType"]
#Numerical columns
num_cols=["Temperature(°C)","Humidity(%)","FarmExperience(Years)","NumWorkers",
          "PreviousYield(tons/ha)","Rainfall(mm)_log","SoilPH_log","Fertilizer(kg/ha)_log",
          "Pesticide(L/ha)_log","DistanceToMarket(km)_log","PlantingDensity(plants/ha)_log",
          "FarmSize(Hectares)_log","FarmAccessComfort","EnvironmentComfort","SoilIndex","IncreaseRate"]
#Raw Features
X=df[cat_cols+num_cols+bin_cols]
#Targets
y_yield=df["ActualYield(tons/ha)_log"]
y_cat=le.fit_transform(df["YieldCategory"])
X_train,X_test,y_yield_train,y_yield_test=train_test_split(X,y_yield,test_size=0.2,random_state=7)
X_yield_train=processor_reg.transform(X_train)
X_yield_test=processor_reg.transform(X_test)
X_train,X_test,y_cat_train,y_cat_test=train_test_split(X,y_cat,test_size=0.2,random_state=7,stratify=y_cat)
X_cat_train=processor_cat.transform(X_train)
X_cat_test=processor_cat.transform(X_test)
XGB_model=XGBClassifier(
    n_estimators=200,
    max_depth=9,
    learning_rate=0.1,
    subsample=0.8,
    colsubsample_bytree=0.8,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=7,
    n_jobs=-1
)
# handling imballance
from sklearn.utils.class_weight import compute_sample_weight
weight=compute_sample_weight('balanced',y=y_cat_train)
XGB_model.fit(X_cat_train,y_cat_train, sample_weight=weight)
y_pred1=XGB_model.predict(X_cat_test)
accuracy1=accuracy_score(y_cat_test,y_pred1)
report1=classification_report(y_cat_test,y_pred1)
print(f"XGBClassifierAcuracy: {accuracy1*100:.2f}%")
print(f"{report1}")
joblib.dump(XGB_model,'XGB_model.joblib')
# LGBM classifier
LGBM_model=LGBMClassifier(
    n_estimators=300,
    max_depth=9,
    num_leaves=64,
    learning_rate=0.08,
    subsample=0.8,
    random_state=7,
    class_weight='balanced',
    n_jobs=-1
)
LGBM_model.fit(X_cat_train,y_cat_train, sample_weight=weight)
y_pred=LGBM_model.predict(X_cat_test)
accuracy=accuracy_score(y_cat_test,y_pred)
report=classification_report(y_cat_test,y_pred)
print(f"LGBClassifierAcuracy: {accuracy*100:.2f}%")
print(f"{report}")
joblib.dump(LGBM_model,'LGBM_model.joblib')
# XGBClassifier fine tunning
XGB_Params={
    'n_estimators':[200,250,350],
    'max_depth':[7,11,13],
    'learning_rate':[0.05,0.2,0.5],
    'subsample':[0.5,1],
}
XGB_tunned=GridSearchCV(
    XGBClassifier(use_label_encoder=False,
                  random_state=7,
                  eval_metric='mlogloss',
                  n_jobs=-1),
    XGB_Params,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)
XGB_tunned.fit(X_cat_train,y_cat_train,sample_weight=weight)
print(XGB_tunned.best_params_)
print(XGB_tunned.best_score_)
y_pred2=XGB_tunned.best_estimator_.predict(X_cat_test)
accuracy2=accuracy_score(y_cat_test,y_pred2)
report2=classification_report(y_cat_test,y_pred2)
joblib.dump(XGB_tunned.best_estimator_,'XGB_TunnedModel')
