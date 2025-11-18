# 🌾 AgriSphere – AI-Based Crop Recommendation & Mandi Price Analysis


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python">
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit">
  <img src="https://img.shields.io/badge/SQLite-Database-green?logo=sqlite">
  <img src="https://img.shields.io/badge/PowerBI-Dashboard-yellow?logo=powerbi">
  <img src="https://img.shields.io/badge/ML-RandomForest-success?logo=scikitlearn">
</p>

---

## 🚀 Overview

AgriSphere is an **intelligent smart farming application** that recommends the most profitable and biologically suitable crop for Indian farmers using:

- Machine Learning  
- Live Weather API  
- Real-time Mandi Price Scraping  
- Soil Nutrient Analysis  
- SQL Logging  
- Power BI Analytics  
- Auto Location Detection

  <img width="1125" height="494" alt="image" src="https://github.com/user-attachments/assets/edb3c1fb-fd50-493f-bcae-812e83a8f251" />
  <img width="296" height="359" alt="image" src="https://github.com/user-attachments/assets/1d3d4fc7-95dd-4b7e-8c60-488a623389c3" />
  <img width="365" height="357" alt="image" src="https://github.com/user-attachments/assets/071ab581-a05b-42b1-a510-e4ade245d021" />



---

## ✨ Features

### 🔹 1. AI-Based Crop Recommendation
- Inputs: `N, P, K, pH, rainfall, temperature, humidity`
- ML Model: **RandomForestClassifier**
- Output: **Top 3 Recommended Crops + Suitability %**

---

### 🔹 2. Live Weather Integration
- Auto detects farmer’s location (browser/IP)
- Fetches live **temperature & humidity** using OpenWeatherMap API

---

### 🔹 3. Mandi Price Scraping (Agmarknet)
- Multi-strategy fallback scraper  
- Regex + table parsing  
- Cleans numeric values  
- Local caching for speed  

---

### 🔹 4. 🥇 Ranking Engine (ML + Economics)

**Hybrid Scoring Formula:**
Final Score = (0.6 × ML Suitability) + (0.4 × Normalized Mandi Price)



Also includes:
- Sanity checks  
- Temperature constraints  
- Price normalization  
- Final re-ranking  

---

### 🔹 5. SQL Database Logging
Stores:
- Requests  
- Predictions  
- Mandi prices

<img width="1125" height="621" alt="image" src="https://github.com/user-attachments/assets/8c0b5d2a-a6e1-4ce0-b1b0-34d10f3220fc" />


Used for Power BI insights.

---

### 🔹 6. 📈 Power BI Dashboard
Visualizes:
- Top recommended crops  
- State-wise distributions  
- Suitability trends  
- Price insights  
- User heatmaps

<img width="678" height="397" alt="image" src="https://github.com/user-attachments/assets/70b3a5bb-6be5-4066-9954-156f8a1bcdd9" />


---

## 🧠 Tech Stack

| Component | Technology | Description |
|----------|------------|-------------|
| Frontend | Streamlit | Web UI |
| Backend | Python | Core logic |
| ML Model | Scikit-learn | RandomForest classifier |
| Web Scraper | BeautifulSoup4 | Agmarknet price extraction |
| Database | SQLite | Local SQL storage |
| Analytics | Power BI | Dashboard visualization |
| API | OpenWeatherMap | Live weather service |

---

## 📁 Project Structure

```bash
AgriSphere/
│── app.py                    # ML model trainer
│── web_app.py                # Streamlit main application
│── db.py                     # SQLite DB functions
│── etl_export.py             # Export DB → CSV for Power BI
│── model.joblib              # Trained ML model
│── Crop_recommendation.csv   # Dataset
│── price_cache.json          # Cached mandi prices
│── requirements.txt          # Dependencies
│── /exports                  # Power BI CSV outputs
│── /screenshots              # UI/Dashboard images
└── README.md

```
## ⚙️ Installation & Running

### 🔧 1. Install Dependencies
`pip install -r requirements.txt`

### 🔑 2. Add OpenWeatherMap API Key
Create this file:
`.streamlit/secrets.toml`

Add your API key:
`OPENWEATHERMAP_API_KEY = "your_api_key_here"`

### ▶️ 3. Run the Streamlit Application
`streamlit run web_app.py`
The application will launch automatically in your browser.

---

## 📊 Power BI Integration

### 📤 Export SQLite Data to CSV
`python etl_export.py`

Exports will be generated in the `/exports/` folder:
- `requests.csv`
- `recommendations.csv`
- `prices.csv`

### 📥 Load into Power BI

- Open Power BI Desktop
- Click Get Data → Text/CSV
- Load all three files
- Create your dashboard
- Refresh anytime after re-running `etl_export.py`

---

## 🧪 Testing

| Component | Status | 
|----------|------------|
| Weather API | ✅ Working |
| ML Predictions | ✅ Accurate |
| Mandi Price Scraper | ⚠️ Has fallback handler |
| SQL Logging | ✅ Working |
| UI / Streamlit | ✅ Stable |

---

## 🔮 Future Enhancements
- Mobile App (Android / iOS)
- Mandi Price Forecasting using time-series models
- Crop Disease Detection (image-based)
- Local language support (Punjabi/Hindi)
- Satellite + IoT soil sensor integration
- Fertilizer recommendation engine

---

## 🏁 Conclusion
AgriSphere is a complete Machine-Learning–powered agricultural decision support system.
It improves farmer profitability by combining data science, real-time APIs, web scraping, SQL, and Power BI analytics into one unified platform.

---

## ⭐ Author

### 👨‍💻 Shubhdeep Bhole
### 🧾 Roll No: 24410998584
