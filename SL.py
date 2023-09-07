import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import hydralit_components as hc
from PIL import Image


# Load the data
data = pd.read_csv('star_cars_EDA.csv')
data = data.drop(columns=['sale_year', 'sale_month', 'sale_status', 'import_to_sale_months', 'genmodel_ID', 'date_of_import', 'date_of_sale', 'import_year', 'import_month','country_of_origin','electric_type'])
X = data.drop('listed_price', axis=1)

# Split the data into training and validation sets

# we use the whole data to train the model 
X_train = X
y_train = np.sqrt(data['listed_price'])

# Define pipeline function
def create_pipeline(scaler, classifier):
    # Define the numeric transformers with different scaling methods
    num_transformer = Pipeline(steps=[
        ('scaler', scaler)
    ])

    cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, X.select_dtypes(include=['float', 'int']).columns),
        ('cat', cat_transformer, X.select_dtypes(include=object).columns)
    ])

    # Define the pipeline with resampling strategies
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline



# Create the pipeline
scaler_choice = MinMaxScaler()
classifier_choice = XGBRegressor()
pipeline = create_pipeline(scaler_choice, classifier_choice)

# Load the trained pipeline
pipeline.fit(X_train, y_train)

st.set_page_config(layout='wide') 

# Define menu data
menu_data = [
    {"label": "Home"},
    {"label": "Prediction"}
]

# Create navigation bar
menu_id = hc.nav_bar(menu_definition=menu_data, sticky_mode='sticky')

# Define page contents based on the selected tab
if menu_id == "Home":
    st.title("Welcome to Car Price Prediction App")
    # Use columns to create a two-column layout
    col1, col2 = st.columns(2)  # Create two equal-width columns

    # Column 1: Image
    with col1:
        st.image('pic.png', use_column_width=False, width=600)
        # st.title("Welcome to Car Price Prediction App")

    # Column 2: Rest of the content
    with col2:
        st.write(".")
elif menu_id == "Prediction":
    st.title("Car Price Prediction")
    # Use columns to create a two-column layout
    col1, col2 = st.columns(2)  # Create two equal-width columns

    # Column 1: Image
    with col1:
        st.image('price.png', use_column_width=False, width=400)
        # st.title("Welcome to Car Price Prediction App")

    # Column 2: Rest of the content
    with col2:

        # Sidebar inputs
        st.header("Car Features")

        # Input fields
        input_features = {}

        # Numeric input features
        for col in X.select_dtypes(include=['float', 'int']).columns:
            if col in ['model_year', 'seat_num', 'door_num']:
                input_features[col] = st.number_input(col, value=int(X[col].mean()), step=1, format='%d')
            elif col == 'engine_size':  # Limit engine_size to two decimal places
                input_features[col] = st.number_input(col, value=X[col].mean(), step=0.01, format='%0.2f')
            else:
                input_features[col] = st.number_input(col, value=X[col].mean(), step=0.01, format='%0.2f')

        # Categorical input features
        for col in X.select_dtypes(include=object).columns:
            input_features[col] = st.selectbox(col, X[col].unique())

        # Predict button
        if st.button("Predict", key="predict_button"):
            input_df = pd.DataFrame([input_features])
            predicted_price = pipeline.predict(input_df)
            st.success(f"Predicted Price: {predicted_price[0]**2:.2f}")

        # Summary of Input Features
        st.header("Input Summary")
        st.subheader("Selected Input Features:")

        # Display selected input features and their values
        for feature, value in input_features.items():
            st.text(f"{feature}: {value}")
