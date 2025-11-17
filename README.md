🌾 AgriSphere – AI-Based Crop Recommendation & Mandi Price Analysis

Developer: Shubhdeep Bhole
Roll No: 24410998584

AgriSphere is an automated, intelligent decision-support system designed for Indian farmers.
It predicts the best crop for a farmer's land by combining:

✔ Machine Learning
✔ Live Weather Data
✔ Mandi Price Scraping
✔ Soil Nutrient Inputs
✔ SQL Logging
✔ Power BI Analytics Dashboard

🚀 Features
🔹 1. AI-Based Crop Recommendation

RandomForest ML model trained on the Crop Recommendation Dataset

Inputs: N, P, K, pH, rainfall, temperature, humidity

Outputs: Top 3 crops + suitability %

🔹 2. Live Weather Integration

Powered by OpenWeatherMap API

Automatically detects farmer’s location

Fetches temperature & humidity in real-time

🔹 3. Mandi Price Scraping (Agmarknet)

A hardened and multi-strategy web scraper that fetches real mandi prices:

Regex-based extraction

Multi-table fallback

Cached for performance

🔹 4. Ranking Engine (AI + Market Economics)

Final crop ranking =

0.6 × ML Suitability  +  0.4 × Price Normalization


Includes sanity checks (ex: cold regions won’t get tropical crops).

🔹 5. SQL Database Logging

All recommendations are logged into SQLite:

Requests

Recommendations

Mandi Prices

Exports available for Power BI analytics.

🔹 6. Power BI Dashboard

Visual insights:

Top crops

Average suitability

Price coverage %

State-wise trends

User location map

Price trend charts

🧠 Tech Stack
Component	Technology
Frontend	Streamlit
Backend	Python
Machine Learning	Scikit-learn
Web Scraping	BeautifulSoup4
Database	SQLite
Data Visualization	Power BI
APIs	OpenWeatherMap
📁 Project Structure
AgriSphere/
│── app.py                     -> ML model trainer
│── web_app.py                 -> Streamlit main app
│── db.py                      -> SQL models & saving logic
│── etl_export.py              -> Exports DB to CSV for Power BI
│── model.joblib               -> Trained ML model
│── Crop_recommendation.csv    -> Dataset (optional)
│── requirements.txt           -> Libraries
│── /exports                   -> Power BI CSV outputs
│── /screenshots               -> UI & dashboard screenshots
└── README.md

⚙️ Installation & Running
1. Install Dependencies
pip install -r requirements.txt

2. Add API Key

Create
.streamlit/secrets.toml

OPENWEATHERMAP_API_KEY = "your_api_key_here"

3. Run the App
streamlit run web_app.py


App will open automatically in the browser.

📊 Power BI Dashboard

Export updated CSVs:

python etl_export.py


Load the CSVs into Power BI

Auto-updated visuals appear.

🧪 Testing
Test	Status
Weather API	✅ Passed
ML Prediction	✅ Passed
Mandi Scraper	⚠️ Resilient (fallback)
SQL Logging	✅ Passed
UI Testing	✅ Passed
🔮 Future Enhancements

Mobile app version

Price prediction using time series forecasting

Local language support (Punjabi/Hindi)

Satellite data integration

IoT soil sensor compatibility

🏁 Conclusion

AgriSphere is a powerful example of integrating Machine Learning, live APIs, data scraping, SQL, and Power BI.
It provides farmers with accurate crop recommendations and real-time market insights, improving profitability and reducing risk.
