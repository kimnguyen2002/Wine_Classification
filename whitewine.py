# Student ID: 10914095
# Name: Kim
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# python -m streamlit run whitewine.py

##################################################################
# Part One: Load a dataset and Look at the summary of the dataset
##################################################################

# Read the dataset
df = pd.read_csv('winequality-white.csv', delimiter=';', quoting=1)

# Clean the column names by removing the quotation marks
df.columns = df.columns.str.replace('"', '')

# Clean the data by removing leading/trailing spaces and converting numeric values to appropriate data types
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df = df.apply(pd.to_numeric, errors='ignore')

# Shape of dataset
print (df.shape)
# Data columns values
print (df.columns)
# Data types of each column 
print (df.dtypes)
# Basic statistics of all numeric columns
print (df.select_dtypes("float64").describe().T)

column_names_list = list(df.columns.values)
print(type(column_names_list))
print(column_names_list)
print()

target = column_names_list[-1]
print(target)
print()

##################################################################
# Part Two: EDA (Exploratory Data Analysis) of the dataset
##################################################################

# Histogram
df.hist(bins=20, figsize=(10, 10))
plt.show()

# Count plot
plt.title("Count plot")
plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()

# Heatmap
plt.figure(figsize=(12, 12))
df_plot = sns.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()

new_df=df.drop('total sulfur dioxide',axis=1)
new_df.isnull().sum()
new_df.update(new_df.fillna(new_df.mean())) # catogerical vars 
next_df = pd.get_dummies(new_df,drop_first=True)
# display new dataframe
next_df

##################################################################
# Part Three: Machine Learning Model Training and Evaluation
##################################################################

# Split the dataset into features and target variable
X = df.drop(target, axis=1)
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

##################################################################
# Part Four: Streamlit Web App
##################################################################

# Streamlit Web App
st.title("White Wine Classification Web App")
st.write("Predict the quality of white wine using machine learning models.")

# Add a sidebar with widgets for model selection and feature input
st.sidebar.title("Model Selection")
model_name = st.sidebar.selectbox("Select a machine learning model:", ["Random Forest", "Support Vector Machine", "Logistic Regression"])

st.sidebar.title("Feature Input")
fixed_acidity = st.sidebar.slider("Fixed Acidity", float(df["fixed acidity"].min()), float(df["fixed acidity"].max()), float(df["fixed acidity"].mean()))
volatile_acidity = st.sidebar.slider("Volatile Acidity", float(df["volatile acidity"].min()), float(df["volatile acidity"].max()), float(df["volatile acidity"].mean()))
citric_acid = st.sidebar.slider("Citric Acid", float(df["citric acid"].min()), float(df["citric acid"].max()), float(df["citric acid"].mean()))
residual_sugar = st.sidebar.slider("Residual Sugar", float(df["residual sugar"].min()), float(df["residual sugar"].max()), float(df["residual sugar"].mean()))
chlorides = st.sidebar.slider("Chlorides", float(df["chlorides"].min()), float(df["chlorides"].max()), float(df["chlorides"].mean()))
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", float(df["free sulfur dioxide"].min()), float(df["free sulfur dioxide"].max()), float(df["free sulfur dioxide"].mean()))
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", float(df["total sulfur dioxide"].min()), float(df["total sulfur dioxide"].max()), float(df["total sulfur dioxide"].mean()))
density = st.sidebar.slider("Density", float(df["density"].min()), float(df["density"].max()), float(df["density"].mean()))
pH = st.sidebar.slider("pH", float(df["pH"].min()), float(df["pH"].max()), float(df["pH"].mean()))
sulphates = st.sidebar.slider("Sulphates", float(df["sulphates"].min()), float(df["sulphates"].max()), float(df["sulphates"].mean()))
alcohol = st.sidebar.slider("Alcohol", float(df["alcohol"].min()), float(df["alcohol"].max()), float(df["alcohol"].mean()))

# Model evaluation
st.write("Model Evaluation:")
st.write(classification_report(y_test, y_pred))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

# Prepare the feature input for prediction
input_data = pd.DataFrame({
    "fixed acidity": [fixed_acidity],
    "volatile acidity": [volatile_acidity],
    "citric_acid": [citric_acid],
    "residual_sugar": [residual_sugar],
    "chlorides": [chlorides],
    "free_sulfur_dioxide": [free_sulfur_dioxide],
    "total_sulfur_dioxide": [total_sulfur_dioxide],
    "density": [density],
    "pH": [pH],
    "sulphates": [sulphates],
    "alcohol": [alcohol]
})

# Reorder the columns to match the order used during training
input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

# Make predictions based on the selected model
if model_name == "Random Forest":
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(input_data)
elif model_name == "Support Vector Machine":
    model = SVC()
    model.fit(X_train, y_train)
    prediction = model.predict(input_data)
elif model_name == "Logistic Regression":
    model = LogisticRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(input_data)

# Display the predicted wine quality
st.subheader("Predicted Wine Quality")
st.write("The predicted wine quality is:", prediction)

# Show a scatter plot of alcohol content vs. quality
plt.figure(figsize=(8, 6))
sns.scatterplot(x="alcohol", y="quality", data=df)
plt.title("Alcohol Content vs. Quality")
st.pyplot()

# Show a table of the first 10 rows of the dataset
st.subheader("Dataset Preview")
st.write("First 10 rows of the dataset", df.head(10))

# Show the classification report for the selected model
st.subheader("Classification Report")
st.write(classification_report(y_test, y_pred, zero_division=1))

# Show the confusion matrix for the selected model
st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))
