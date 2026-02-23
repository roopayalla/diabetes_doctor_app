import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

print("Training diabetes model...")

# Load Excel
df = pd.read_excel("datamining.xlsx", sheet_name="datamining")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Gender fix
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})

# Clean data (exact column names)
zero_cols = ['Blood_pressure', 'Glucose_level', 'Insuline', 'Skin_Thickness']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan).fillna(df[col].median())

# Cholesterol fix
df.loc[df['Cholesterol'] > 500, 'Cholesterol'] = df['Cholesterol'].median()

# Features (EXACT Excel columns)
features = ['Age', 'Gender', 'Height', 'Weight', 'BMI', 'Blood_pressure', 
            'Glucose_level', 'Insuline', 'Skin_Thickness', 'Pregencies', 
            'D.P.F', 'Cholesterol', 'Heart_Rate']
X = df[features]
y = df['Outcome']

# Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model saved! Accuracy:", model.score(X_test, y_test))
print("Files ready for app.py!")
