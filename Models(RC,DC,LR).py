import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score,accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report,mean_squared_error

df=pd.read_csv("/storage/emulated/0/Download/StudentDropout/FarmYield(Engineering).csv")
#Label Encoding binary columns
le=LabelEncoder()
for c in ["FertilizerUsed","PesticideUsed","ImprovedSeed"]:
    df[c]=le.fit_transform(df[c])
#Binary columns
bin_cols=["FertilizerUsed","PesticideUsed","ImprovedSeed","CreditAccess","ExtensionServices"]
#Categorical columns
cat_cols=["State","Crop","Season","SoilType","FarmingMethod","IrrigationType"]
#Numerical columns
num_cols=["Temperature(°C)","Humidity(%)","FarmExperience(Years)","NumWorkers","PreviousYield(tons/ha)","Rainfall(mm)_log","SoilPH_log","Fertilizer(kg/ha)_log","Pesticide(L/ha)_log","DistanceToMarket(km)_log","PlantingDensity(plants/ha)_log","FarmSize(Hectares)_log","FarmAccessComfort","EnvironmentComfort","SoilIndex","IncreaseRate"]
#pipline
processor_reg=ColumnTransformer(transformers=[("ohe",OneHotEncoder(drop="first", sparse_output=False,handle_unknown="ignore"),cat_cols),("scaler",StandardScaler(),num_cols)],remainder="passthrough")
processor_cat=ColumnTransformer(transformers=[("ohe",OneHotEncoder(drop="first", sparse_output=False,handle_unknown="ignore"),cat_cols),("scaler",StandardScaler(),num_cols)],remainder="passthrough")
#Raw Features
X=df[cat_cols+num_cols+bin_cols]
#Targets
y_yield=df["ActualYield(tons/ha)_log"]
y_cat=le.fit_transform(df["YieldCategory"])
#Regression set
X_train,X_test,y_yield_train,y_yield_test=train_test_split(X,y_yield,test_size=0.2,random_state=7)
X_yield_train=processor_reg.fit_transform(X_train)
X_yield_test=processor_reg.transform(X_test)
#Categorical set
X_train,X_test,y_cat_train,y_cat_test=train_test_split(X,y_cat,test_size=0.2,random_state=7,stratify=y_cat)
X_cat_train=processor_cat.fit_transform(X_train)
X_cat_test=processor_cat.transform(X_test)
#saving processors
import joblib 
joblib.dump(processor_cat,"processor_cat.joblib")
joblib.dump(processor_reg,"processor_reg.joblib")
#Linear Regression
model=LinearRegression()
model.fit(X_yield_train,y_yield_train)
y_pred=model.predict(X_yield_test)
print(f"R²: {r2_score(y_yield_test,y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_yield_test,y_pred)):.4f}")
#LogisticRegression
model1=LogisticRegression(class_weight="balanced",max_iter=1000,random_state=7)
model1.fit(X_cat_train,y_cat_train)
y_pred1=model1.predict(X_cat_test) 
print(f"LogisticRegression: {accuracy_score(y_cat_test,y_pred1)*100:.2f}%")
print(classification_report(y_cat_test,y_pred1))
#DecisionTreeClassifier
model2=DecisionTreeClassifier(max_depth=9,random_state=7,class_weight="balanced")
model2.fit(X_cat_train,y_cat_train)
y_pred2=model2.predict(X_cat_test) 
print(f"DecisionTree: {accuracy_score(y_cat_test,y_pred2)*100:.2f}%")
print(classification_report(y_cat_test,y_pred2))
#RandomForestClassifier
model3=RandomForestClassifier(n_estimators=200,random_state=7,class_weight="balanced",max_depth=11)
model3.fit(X_cat_train,y_cat_train)
y_pred3=model3.predict(X_cat_test) 
print(f"RandomForest: {accuracy_score(y_cat_test,y_pred3)*100:.2f}%")
print(classification_report(y_cat_test,y_pred3))
#feature importance
feature_names = processor_cat.get_feature_names_out()
importances = model3.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
feat_imp['Original_Feature'] = feat_imp['Feature'].apply(
    lambda x: x.split('__')[-1].split('_')[0]
)
grouped = feat_imp.groupby('Original_Feature')['Importance'].sum().sort_values(ascending=False)
print(grouped)
#FineTuing best
params={"n_estimators":[50,150,250],"max_depth":[7,9,13],"min_samples_split":[10,15,20],"class_weight":["balanced",None]}
grid=GridSearchCV(RandomForestClassifier(random_state=7),params,cv=7,scoring="f1_weighted",verbose=1)
grid.fit(X_cat_train,y_cat_train)
print(grid.best_params_)
print(grid.best_score_)
y_pred=grid.best_estimator_.predict(X_cat_test)
print(accuracy_score(y_cat_test,y_pred))
print(classification_report(y_cat_test,y_pred))
