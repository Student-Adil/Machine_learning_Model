import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("Collagedata.csv")

# ⚠️ FIX column name (remove trailing space)
df.columns = df.columns.str.strip()

X = df[['Major', 'Minor', 'Cgpa']]
y = df['Choose']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Encode categorical inputs
ct = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Major', 'Minor'])
    ],
    remainder='passthrough'
)

X_encoded = ct.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 🔥 SAVE EVERYTHING
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(ct, open("encoder.pkl", "wb"))
pickle.dump(le, open("label.pkl", "wb"))

print("✅ Model, encoder, and label saved")
