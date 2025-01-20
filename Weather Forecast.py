import gradio as gr
import pandas as pd
import numpy as np
import os
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Dataset
def dataset():
    np.random.seed(42)
    information = {
        "Temperature": np.random.uniform(-10, 45, 1000),
        "Humidity": np.random.uniform(10, 100, 1000),
        "WindSpeed": np.random.uniform(0, 15, 1000),
        "Pressure": np.random.uniform(950, 1050, 1000),
    }
    climate = []
    for temperature, humidity, wind, press in zip(
        information["Temperature"], information["Humidity"], information["WindSpeed"], information["Pressure"]
    ):
        if temperature < -5:
            climate.append("Freezing")
        elif temperature < 0:
            climate.append("Snowy")
        elif humidity > 90 and temperature > 20:
            climate.append("Tropical Rain")
        elif humidity > 85:
            climate.append("Rainy")
        elif wind > 12:
            climate.append("Stormy")
        elif temperature > 35 and humidity < 30:
            climate.append("Heatwave")
        elif temperature > 30:
            climate.append("Sunny")
        elif humidity < 30 and temperature < 15:
            climate.append("Dry and Cold")
        elif humidity < 30:
            climate.append("Cloudy")
        else:
            climate.append("Overcast")
    information["Condition"] = climate
    return pd.DataFrame(information)

# Load the dataset
weather_forecast = dataset()

# Split the dataset into features and target
X = weather_forecast[["Temperature", "Humidity", "WindSpeed", "Pressure"]]
y = weather_forecast["Condition"]

# Encode the variables by using LabelEncoder
label_encodeR = LabelEncoder()
y_encoded = label_encodeR.fit_transform(y)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_scaled, y_train)

# Prediction and Weather Image Handling
def weather_image(condition):
    # Here, you can map conditions to specific images and captions (can be improved)
    images = {
        "Sunny": "sunny.gif",  
        "Cloudy": "cloudy.gif",
        "Rainy": "rainy.gif",
        "Snowy": "snowy.gif",
        "Stormy": "stormy.jpg",
        "Overcast": "overcast.gif",
        "Tropical Rain": "tropical_rain.jpg",
        "Dry and Cold": "dry_cold.jpg",
        "Heatwave": "heatwave.jpg",
        "Freezing": "freezing.jpg",
    }
    # Return the path of the image
    image_path = os.path.join("weather_images", images.get(condition, "unknown.gif"))
    caption = f"The current condition is: {condition}"
    return image_path, caption

def prediction(temp, humidity, windspeed, pressure):
    input_data = pd.DataFrame([[temp, humidity, windspeed, pressure]], columns=["Temperature", "Humidity", "WindSpeed", "Pressure"])
    input_data_scaled = scaler.transform(input_data)

    prediction_encoded = classifier.predict(input_data_scaled)[0]
    condition = label_encodeR.inverse_transform([prediction_encoded])[0]
    
    image, caption = weather_image(condition)
    
    return f"The predicted weather condition is: {condition}", image, caption

# Gradio Interface
def create_gradio_app():
    with gr.Blocks() as app:
        with gr.Tab("Main Page"):
            gr.Markdown(
                """
                <h1 style="text-align:center; font-size:3em; color:#2c3e50;">Weather Forecasting Prediction Model</h1>
                <p style="text-align:center; font-size:1.2em; color:#34495e;">
                    Accurately predicting weather conditions based on temperature, humidity, wind speed, and pressure.
                </p>
                """,
                elem_id="main-page",
            )
        with gr.Tab("Weather Conditions"):
            inputs = [
                gr.Number(label="Temperature (°C)"),
                gr.Number(label="Humidity (%)"),
                gr.Number(label="Wind Speed (m/s)"),
                gr.Number(label="Pressure (hPa)"),
            ]
            outputs = [
                gr.Textbox(label="Weather Data"),
                gr.Image(label="Generated Weather Image"),
                gr.Textbox(label="Detailed Caption"),
            ]

            gr.Interface(
                fn=prediction,
                inputs=inputs,
                outputs=outputs,
                title="Weather Forecasting",
                description="Enter weather metrics to predict conditions, generate an image, and get a detailed caption.",
            ).launch(share=True)

if __name__ == "__main__":
    create_gradio_app()




