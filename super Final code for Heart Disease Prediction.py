import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('heart_disease_data.csv')

# checking for missing values
heart_data.dropna(inplace=True)

# Splitting data into features and target
X = heart_data.drop(columns=['target'], axis=1)
y = heart_data['target']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training the Logistic Regression model
model = LogisticRegression()

# Applying Cross-Validation for better performance evaluation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

# Fitting the model and evaluating on test data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Fitting the model and evaluating on test data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate ROC-AUC Score
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC-AUC Score: {roc_auc:.4f}')

input_data = (29,1,1,130,204,0,0,202,0,0,2,0,2)
np_data = np.asarray(input_data)
reshaped_df = np_data.reshape(1,-1)

# Apply scaling here
scaled_input = scaler.transform(reshaped_df)
pred = model.predict(scaled_input)

# Prediction
st.title('Heart Disease Preidction Model')

input_text = st.text_input('Provide comma separated features to predict heart disease')
sprted_input = input_text.split(',')
img = Image.open('heart_img.jpg')
st.image(img,width=150)

try:
    np_df = np.asarray(sprted_input, dtype=float).reshape(1, -1)
    
    # Apply scaling (assuming 'scaler' is already fitted)
    scaled_input = scaler.transform(np_df)

    # Predict
    prediction = model.predict(scaled_input)
    
    if prediction[0] == 0:
        st.write("💖 This person doesn't have heart disease.")
    else:
        st.write("💔 This person does have heart disease.")

except ValueError:
    st.write('🚨 Please provide comma-separated numeric values.')

# About section
st.subheader("About Data")
st.write(heart_data)

# Performance section

st.subheader("Model Performance on Test Data :")
st.write(accuracy * 100)

st.subheader("F1 SCORE :")
st.write(f1 *100)

st.subheader("ROC-AUC SCORE :")
st.write(roc_auc * 100)