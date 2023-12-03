import streamlit as st
import pandas as pd
import seaborn as sns
import pickle

try:
    with open('breast_cancer_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file not found. Please provide the correct path.")

def predict_diagnosis(new_instance):
    prediction = model.predict(new_instance)
    return prediction
def main():
    st.title('Breast Cancer Prediction')
    st.write("Please input the following features:")
    columns = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
        'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
        'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
        'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    feature_values = {}
    for column in columns:
        feature_values[column] = st.sidebar.number_input(f'{column.capitalize().replace("_", " ")}')
    data_average = {column: feature_values[column] for column in feature_values}
    st.write("Average of all columns:")
    st.text(data_average)
    new_instance = pd.DataFrame(feature_values, index=[0])
    if st.button('Predict'):
        prediction = predict_diagnosis(new_instance)
        diagnosis = "Malignant" if prediction[0] == 'M' else "Benign"
        st.write(f"Predicted diagnosis for the given values: {diagnosis}")
if __name__ == "__main__":
    main()
