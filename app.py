import streamlit as st
import pickle
import numpy as np
import sklearn

# import the model
#load a package

# to load
with open('picklefolder/pipe.pkl', 'rb') as f:
    pipe = pickle.load(f)

with open('picklefolder/df.pkl', 'rb') as f:
    df = pickle.load(f)

st.title("Student flexibility predictor")

# create a sidebar
st.sidebar.title("Filters")

# brand
education_level = st.sidebar.selectbox('education',df['education_level'].unique())

# type of laptop
institution_type = st.sidebar.selectbox('Institute Type',df['institution_type'].unique())

# Gender
gender = st.sidebar.selectbox('gender',df['gender'].unique())

# age
age = st.sidebar.slider("age", 1, 90, value=25)
age = int(age)

device = st.sidebar.selectbox('device',df['device'].unique())

#IT STUDENT
it_student = st.sidebar.selectbox('it_student',['Yes','No'])

location = st.sidebar.selectbox('location',['Town','Rural'])

financial_condition = st.sidebar.selectbox('financial_condition',['Mid','Poor','Rich'])

internet_type = st.sidebar.selectbox('internet_type',['Wifi','Mobile Data'])

network_type = st.sidebar.selectbox('network_type',['4G','3G','2G'])

#prediction button
if st.button("Predict"):
    query = np.array([education_level,institution_type,gender,age,device,it_student,location,financial_condition,internet_type,network_type])
    query = query.reshape(1,10)
    prediction = pipe.predict(query)[0]
    
    # define the prediction labels dictionary
    prediction_labels = {0: 'Low', 1: 'Moderate', 2: 'High'}
    
    # map the prediction output value to the corresponding label
    prediction_label = prediction_labels[prediction]
    st.title("The predicted class for this configuration is " + prediction_label)