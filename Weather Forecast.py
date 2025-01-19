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

# Evaluate the model
y_pred = classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Detailed image captions
detailed_captions = {
    "Sunny": [
        "A bright and clear day, perfect for outdoor activities.",
        "The sun is shining brightly, making it feel warm and cheerful.",
        "It’s a sunny day, ideal for enjoying the outdoors."
    ],
    "Cloudy": [
        "The sky is overcast with clouds, but no rain is expected.",
        "It’s cloudy today, with no direct sunlight breaking through.",
        "A gray sky hangs overhead, making the weather feel cool."
    ],
    "Rainy": [
        "It’s raining, so keep an umbrella handy!",
        "A rainy day, perfect for staying indoors with a warm drink.",
        "Heavy rain is falling, so expect wet conditions."
    ],
    "Snowy": [
        "Snow is falling, creating a winter wonderland outside.",
        "Expect snow flurries, so prepare for slippery roads.",
        "It’s a snowy day, with thick layers of snow on the ground."
    ],
    "Stormy": [
        "A storm is brewing, with strong winds and heavy rain.",
        "Prepare for stormy weather with gusty winds and rainfall.",
        "Thunderstorms are expected, so stay safe indoors."
    ],
    "Overcast": [
        "The sky is overcast, with no sunlight peeking through.",
        "An overcast sky means cooler temperatures are expected.",
        "Cloud cover is thick, and no sun is visible today."
    ],
    "Tropical Rain": [
        "A tropical rainstorm is in progress, with heavy downpours.",
        "Expect intense rainfall and high humidity with this tropical storm.",
        "A tropical rain is cooling down the area, but it’s quite heavy."
    ],
    "Dry and Cold": [
        "The air is dry and cold, with a chill that makes you bundle up.",
        "Cold and dry conditions make for an unpleasant outdoor experience.",
        "Expect dry, chilly air that requires warm clothing."
    ],
    "Heatwave": [
        "It’s a scorching heatwave, so stay hydrated and cool.",
        "Extreme heat is making the day feel sweltering.",
        "A heatwave is in progress, with temperatures soaring high."
    ],
    "Freezing": [
        "Freezing conditions are here, making it dangerously cold.",
        "Expect freezing temperatures, perfect for winter sports.",
        "It’s freezing outside, so bundle up to stay warm."
    ],
}

# Modify the weather_image function to include animations
def weather_image(condition):
    image_dir = "weather_images"
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

    image_file = images.get(condition, "unknown.gif")
    image_path = os.path.join(image_dir, image_file)

    if os.path.exists(image_path):
        return image_path, random.choice(detailed_captions.get(condition, ["Weather conditions are unclear."]))
    else:
        fallback_image = os.path.join(image_dir, "unknown.gif")
        if not os.path.exists(fallback_image):
            fallback_image = None
        return fallback_image, "Weather image not available."

# Modify the prediction function to remove background color
def prediction(temp, humidity, windspeed, pressure):
    input_data = pd.DataFrame([[temp, humidity, windspeed, pressure]], columns=["Temperature", "Humidity", "WindSpeed", "Pressure"])
    input_data_scaled = scaler.transform(input_data)

    prediction_encoded = classifier.predict(input_data_scaled)[0]
    condition = label_encodeR.inverse_transform([prediction_encoded])[0]
    
    image, caption = weather_image(condition)
    
    # Add emojis
    emojis = {
        "Sunny": "🌞",
        "Cloudy": "☁️",
        "Rainy": "🌧️",
        "Snowy": "❄️",
        "Stormy": "🌩️",
        "Overcast": "☁️",
        "Tropical Rain": "🌴🌧️",
        "Dry and Cold": "🌵❄️",
        "Heatwave": "🔥",
        "Freezing": "❄️🥶",
    }
    emoji = emojis.get(condition, "🌈")

    result = f"""
    Temperature: {temp} °C
    Humidity: {humidity} %
    Wind Speed: {windspeed} m/s
    Pressure: {pressure} hPa
    Predicted Condition: {condition} {emoji}
    """
    return result.strip(), image, caption

# Modify Gradio interface to exclude background color
def gradio_interface():
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

    interface = gr.Interface(
        fn=prediction,
        inputs=inputs,
        outputs=outputs,
        title="Weather Forecasting",
        description="Enter weather metrics to predict conditions, generate an image, and get a detailed caption.",
    )
    
    interface.launch(share=True)

if __name__ == "__main__":
    gradio_interface()

