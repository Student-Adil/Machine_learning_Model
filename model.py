import pandas as pd
import numpy as np
df=pd.read_csv('Collagedata.csv')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df.columns=df.columns.str.strip()

x = df[['Major','Minor','Cgpa']]
y = df['Choose']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Major', 'Minor'])
    ],
    remainder='passthrough'
)

X_encoded = ct.fit_transform(x)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# prepare new sample and predict
new = pd.DataFrame([['Economics', 'Physics', 7.3]], columns=['Major', 'Minor', 'Cgpa'])
new_encoded = ct.transform(new)
pred = model.predict(new_encoded)
pred_label = le.inverse_transform(pred)


import pickle
pickle.dump(model,open("model.pkl","wb"))
print(pred_label[0])