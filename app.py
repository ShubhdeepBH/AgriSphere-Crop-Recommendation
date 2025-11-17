
# app.py - Trainer for AgriSphere
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib, argparse
parser = argparse.ArgumentParser()
parser.add_argument('--csv', default='Crop_recommendation.csv')
parser.add_argument('--out', default='model.joblib')
args = parser.parse_args()
df = pd.read_csv(args.csv)
if 'label' not in df.columns:
    raise SystemExit('CSV must contain a "label" column.')
X = df.drop('label', axis=1); y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
joblib.dump({'model': model, 'features': X.columns.tolist()}, args.out)
print('Saved model to', args.out)
