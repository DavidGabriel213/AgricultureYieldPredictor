# 🌾 Nigerian Crop Yield Predictor

A complete dual-target ML project predicting both the **actual crop yield in tons/ha** (regression) and **yield category** (classification) from Nigerian farm data — 20,000 rows, 10 crops, 15 states.

## 🌐 Live Demo
**[Try the app →](https://your-deployment-url.up.railway.app)**

---

## 📌 Project Overview
Nigerian agriculture contributes over 25% of GDP yet productivity remains low. This system helps farmers and agricultural officers predict expected yield based on farm conditions — enabling better planning, resource allocation and early intervention.

**Dual prediction in one app:**
- 🌾 **Actual Yield** → tons per hectare (regression)
- 📊 **Yield Category** → High / Medium / Low / Very Low (classification)

Both predictions are **independent** — two separate models trained in parallel, not chained. The crop type, season and state provide critical context that a single yield number alone cannot capture.

---

## 📊 Dataset
| Property | Value |
|---|---|
| Rows | 20,080 |
| Columns | 26 |
| Crops | 10 (Maize, Rice, Sorghum, Millet, Cassava, Yam, Cowpea, Groundnut, Wheat, Soybean) |
| States | 15 Nigerian states |
| Regression Target | ActualYield(tons/ha) |
| Classification Target | YieldCategory (4 classes) |

---

## 🧹 Data Cleaning Challenges
| Column | Problem | Solution |
|---|---|---|
| FarmSize | "2.5 ha", "6.2 acres", outliers ×10 | Acres × 0.4047 → hectares |
| Rainfall | "850mm", "850 mm", negatives | Strip mm, IQR clip |
| Temperature | "28°C", "28 degrees", "28C" — 3 formats | Strip all suffixes, pd.to_numeric |
| SoilPH | "6.5 pH" AND "pH 6.5" — prefix & suffix! | Strip both positions |
| Fertilizer | "80 kg/ha", "80kg", outliers ×10 | Strip all unit formats |
| DistanceToMarket | "18km", "18 km", "11miles" | Miles × 1.609 → km |
| PreviousYield | "2.5 tons", "2.5 t/ha", outliers | Strip all formats, IQR clip |
| YieldCategory | 28 different formats | str.capitalize() + dict map |

---

## ⚙️ Feature Engineering
| Feature | Formula | Agricultural Meaning |
|---|---|---|
| FarmSize_log | np.log1p(FarmSize) | Normalizes exponential distribution |
| Rainfall_log | np.log1p(Rainfall) | Normalizes skewed rainfall data |
| Fertilizer_log | np.log1p(Fertilizer) | Normalizes fertilizer distribution |
| FarmAccessComfort | FarmSize_log / DistanceToMarket_log | Big farm close to market = optimal |
| EnvironmentComfort | Temperature / Humidity | High temp + low humidity = crop stress |
| SoilIndex | Rainfall_log / (Fertilizer_log + SoilPH_log) | Natural vs chemical input balance |
| IncreaseRate | PreviousYield / NumWorkers | Yield efficiency per worker |

---

## 🤖 Models — Two Independent Pipelines

### Regression (ActualYield)
| Model | R² Score | RMSE |
|---|---|---|
| Linear Regression | 0.21 | 0.89 |
| **Random Forest Regressor** | **0.53** | **0.68** |

### Classification (YieldCategory)
| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 78.83% | Strong baseline |
| Decision Tree | 67.60% | max_depth limited |
| Random Forest | 81.37% | Good ensemble |
| **XGBoost** | **87.70%** | **Best — deployed** |
| LightGBM | 87.50% | Near identical to XGBoost |

---

## 🏗️ Two Independent Preprocessors
```python
# Regression preprocessor
preprocessor_reg = ColumnTransformer([
    ('ohe', OneHotEncoder(...), cat_cols),
    ('scaler', StandardScaler(), num_cols),
], remainder='passthrough')

# Classification preprocessor — separate instance!
preprocessor_cat = ColumnTransformer([
    ('ohe', OneHotEncoder(...), cat_cols),
    ('scaler', StandardScaler(), num_cols),
], remainder='passthrough')
```
Two separate ColumnTransformer instances — fitting one never affects the other.

---

## 🌐 Flask Dual Prediction
```python
# Regression
predicted_log = reg_model.predict(features_reg)[0]
predicted_yield = np.expm1(predicted_log)  # reverse log!

# Classification
predicted_cat_encoded = clf_model.predict(features_cat)[0]
predicted_category = label_encoder.inverse_transform(
    [predicted_cat_encoded])[0]
```

---

## 🏗️ Tech Stack
- **Language:** Python
- **ML:** Scikit-learn, XGBoost, LightGBM
- **Web Backend:** Flask
- **Frontend:** HTML5, CSS3 (Forest Green Theme)
- **Deployment:** Railway.app
- **Version Control:** GitHub

---

## 📁 Project Structure
```
CropYieldPredictor/
├── data/
│   └── nigerian_crop_yield_messy.csv
├── models/
│   ├── xgb_classifier.joblib
│   ├── rf_regressor.joblib
│   ├── preprocessor_cat.joblib
│   ├── preprocessor_reg.joblib
│   └── label_encoder.joblib
├── templates/
│   └── Crop_yield.html
├── static/
│   └── crop_yield.css
├── APP.py
├── requirements.txt
└── Procfile
```

---

## 🚀 Run Locally
```bash
git clone https://github.com/DavidGabriel213/CropYieldPredictor
cd CropYieldPredictor
pip install -r requirements.txt
python APP.py
```

---

## 💡 Key Learnings
1. **Independent models** — when targets are not directly derived from each other, parallel models are more honest than chaining
2. **Two separate ColumnTransformer instances** — same object reference means fitting one refits the other
3. **SoilPH both prefix and suffix** — "6.5 pH" AND "pH 6.5" in same column requires two separate strip operations
4. **Acres → Hectares** conversion — 1 acre × 0.4047 = 1 hectare
5. **SolarCapacity logic fill** — zero when RenewableEnergy=No, group mean when Yes
6. **log transform on regression target** — np.log1p() on skewed yield → np.expm1() in Flask

---

## 👨‍💻 About
**Gabriel David** | Mathematics Undergraduate | ATBU Bauchi
Self-taught ML Engineer — built during Industrial Training placement.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-gabriel--david--ds-blue)](https://linkedin.com/in/gabriel-david-ds)
[![GitHub](https://img.shields.io/badge/GitHub-DavidGabriel213-black)](https://github.com/DavidGabriel213)

