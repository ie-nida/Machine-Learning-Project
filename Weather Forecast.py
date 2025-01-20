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
    "Sunny": ["A bright and clear day, perfect for outdoor activities."],
    "Cloudy": ["The sky is overcast with clouds, but no rain is expected."],
    "Rainy": ["It’s raining, so keep an umbrella handy!"],
    "Snowy": ["Snow is falling, creating a winter wonderland outside."],
    "Stormy": ["A storm is brewing, with strong winds and heavy rain."],
    "Overcast": ["The sky is overcast, with no sunlight peeking through."],
    "Tropical Rain": ["A tropical rainstorm is in progress, with heavy downpours."],
    "Dry and Cold": ["The air is dry and cold, with a chill that makes you bundle up."],
    "Heatwave": ["It’s a scorching heatwave, so stay hydrated and cool."],
    "Freezing": ["Freezing conditions are here, making it dangerously cold."],
}

# Weather image function
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

    fallback_image = os.path.join(image_dir, "unknown.gif")
    if not os.path.exists(fallback_image):
        fallback_image = None
    return fallback_image, "Weather image not available."

# Prediction function
def prediction(temp, humidity, windspeed, pressure):
    input_data = pd.DataFrame([[temp, humidity, windspeed, pressure]], columns=["Temperature", "Humidity", "WindSpeed", "Pressure"])
    input_data_scaled = scaler.transform(input_data)

    prediction_encoded = classifier.predict(input_data_scaled)[0]
    condition = label_encodeR.inverse_transform([prediction_encoded])[0]
    
    image, caption = weather_image(condition)
    
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

# Gradio interface setup
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

weather_descriptions = """
### Weather Conditions and Their Descriptions:

1. **Sunny**:
   - Temperature: Between 30°C and 45°C
   - Humidity: Less than 30%
   - Wind: Calm winds

2. **Cloudy**:
   - Temperature: Between 10°C and 25°C
   - Humidity: Low humidity
   - Wind: Moderate winds

3. **Rainy**:
   - Temperature: Between 15°C and 25°C
   - Humidity: Greater than 85%
   - Wind: Moderate winds

4. **Snowy**:
   - Temperature: Between -5°C and 0°C
   - Humidity: Low humidity
   - Wind: Light wind

5. **Stormy**:
   - Temperature: Between 20°C and 30°C
   - Wind: Strong winds (greater than 12 m/s)

6. **Overcast**:
   - Temperature: Between 10°C and 25°C
   - Humidity: Moderate humidity
   - Wind: No sun

7. **Tropical Rain**:
   - Temperature: Between 25°C and 35°C
   - Humidity: Above 90%
   - Weather: Strong rain

8. **Dry and Cold**:
   - Temperature: Below 15°C
   - Humidity: Low humidity
   - Wind: Cold wind

9. **Heatwave**:
   - Temperature: Above 35°C
   - Humidity: Very low humidity

10. **Freezing**:
    - Temperature: Below -5°C
    - Humidity: Extremely low humidity
"""

with gr.Blocks() as interface:
    with gr.Tab("Title"):
        title_html = """
        <div style="display: flex; justify-content: center; align-items: center; height: 100vh; background-color: transparent; margin: 0;">
            <h1 style="font-size: 40px; color: #ff6347; font-family: 'Arial', sans-serif; text-align: center; padding: 20px 25px; background-color: transparent; border-radius: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15); text-transform: uppercase; letter-spacing: 2px; width: 100%; max-width: 700px; border: 2px solid white;">
                Weather Forecasting Prediction Model
            </h1>
        </div>
        """
        gr.HTML(title_html)  
        
    with gr.Tab("Weather Descriptions"):
        gr.Markdown(weather_descriptions)
    
    with gr.Tab("Weather Forecasting"):
        gr.Interface(
            fn=prediction,
            inputs=inputs,
            outputs=outputs,
            title="Weather Forecasting",
            description="Enter weather metrics to predict conditions, generate an image, and get a detailed caption.",
        )

interface.launch(share=True)
