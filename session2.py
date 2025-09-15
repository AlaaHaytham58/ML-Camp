"""
Name   courses_count   Gender city
Alaa    20    Male  mansoura
Sara         Female Giza
Omar   25    Male   Cairo
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler #normalization
from sklearn.model_selection import train_test_split #for testing

data = pd.DataFrame({
    'Name': ['Alaa', 'Doha', 'Ahmed'],
    'courses_count': [1, None, 5],
    'Gender': ['Female', 'Female', 'Male']
})

print("data before processing ")
print(data)

data['courses_count'].fillna(data['courses_count'].mean(), inplace=True)

print("data after processing")
print(data)
#-----------------------------------------
#Label Encoding

encoder = LabelEncoder()
data['courses_count'] = encoder.fit_transform(data['courses_count'])
print(data)
#----------------------------------------
#Normalization
scaler = MinMaxScaler()
data[['courses_count']] = scaler.fit_transform(data[['courses_count']])
print(data)
#----------------------------------------
#Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['courses_count']] = scaler.fit_transform(data[['courses_count']])
print(data)

#--------------------------------------
# testing
X = data[['courses_count']]
y = [0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train:", X_train)
print("X_test:", X_test)