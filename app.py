import pandas as pd
import numpy as np 
import pickle as pk
import streamlit as st

# Load the machine learning model
model = pk.load(open('model.pkl', 'rb'))

# Set up Streamlit header and title
st.title('Car Price Prediction ML Model')

# Read the car details CSV file
cars_data = pd.read_csv('Cardetails.csv')

# Define a function to extract the brand name from the car name
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

# Apply the function to create a new 'name' column
cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Streamlit widgets for user input
st.sidebar.header('Enter Car Details')
name = st.sidebar.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.sidebar.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.sidebar.slider('No Of Kms Driven', 11, 200000)
fuel = st.sidebar.selectbox('Fuel Type', cars_data['fuel'].unique())
seller_type = st.sidebar.selectbox('Seller Type', cars_data['seller_type'].unique())
transmission = st.sidebar.selectbox('Transmission Type', cars_data['transmission'].unique())
owner = st.sidebar.selectbox('Owner', cars_data['owner'].unique())
mileage = st.sidebar.slider('Car Mileage', 10, 40)
engine = st.sidebar.slider('Engine CC', 700, 5000)
max_power = st.sidebar.slider('Max Power', 0, 200)
seats = st.sidebar.slider('No Of Seats', 5, 10)

# Perform prediction on button click
if st.sidebar.button("Predict"):
    # Create a DataFrame with input data
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    # Preprocess categorical variables
    input_data_model['owner'] = cars_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5])
    input_data_model['fuel'] = cars_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4])
    input_data_model['seller_type'] = cars_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3])
    input_data_model['transmission'] = cars_data['transmission'].replace(['Manual', 'Automatic'], [1, 2])
    input_data_model['name'] = cars_data['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra',
                                                           'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Audi',
                                                           'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',
                                                           'Kia', 'Fiat', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                                                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])

    # Predict car price
    car_price = model.predict(input_data_model)

    # Display predicted car price
    st.success('Predicted Car Price: $' + str(car_price[0]))

# Add background image and other styling
