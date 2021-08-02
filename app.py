# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup,url_for,redirect,make_response
import numpy as np
import pandas as pd
from web import redirect
from disease_dic import disease_dic
from fertilizer_dic import fertilizer_dic
import requests
import json
from time import time
from random import random
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from model import ResNet9


# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']
disease_model_path = 'Trained_Model/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'Trained_Model/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    return render_template('index.html')

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html')

@ app.route('/crop-checkup')
def crop_checkup():
    return render_template('crop-checkup.html')

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer.html')

# render disease prediction input page


@ app.route('/disease')
def disease():
    return render_template('disease.html')

@ app.route('/weather')
def weather_recommendation():
    return render_template('weather2.html')

@app.route('/livedata', methods=["GET", "POST"])
def livedata():
    return render_template('test.html')

@app.route('/data', methods=["GET", "POST"])
def data():
    # Data Format
    # [TIME, Temperature, Humidity]

    Temperature = random() * 100
    Humidity = random() * 55
    Ph = random()*8
    N = random()*100
    Rainfall = random()*500

    data = [time() * 1000, Temperature, Humidity, Ph]

    response = make_response(json.dumps(data))

    response.content_type = 'application/json'

    return response



@ app.route('/potatoResult', methods=['POST'])
def potato_prediction():
    if request.method == 'POST':
        ph = float(request.form['addph'])
        M = float(request.form['addmoisture'])
        T = float(request.form['addtemperature'])
        H = float(request.form['addhumidity'])
    return render_template('potatoResult.html', ph = ph, M = M, T = T, H = H)

@ app.route('/potatoForm')
def potato_form():
    return render_template('potatoForm.html')

# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        state = request.form.get("stt")
        city = request.form.get("city")



        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]




            temperature2, humidity2 = weather_fetch(city)
            data2 = np.array([[N+50, P-10, K+30, temperature2, humidity2, ph+1.9, rainfall-100]])
            my_prediction2 = crop_recommendation_model.predict(data2)
            final_prediction2 = my_prediction2[0]

            temperature3, humidity3 = weather_fetch(city)
            data3 = np.array([[N + 30, P - 70, K + 29, temperature2, humidity2, ph - 1.9, rainfall + 100]])
            my_prediction3 = crop_recommendation_model.predict(data3)
            final_prediction3 = my_prediction3[0]

            temperature4, humidity4 = weather_fetch(city)
            data4 = np.array([[N - 20, P + 40, K + 5, temperature4, humidity4, ph + 2.6, rainfall - 200]])
            my_prediction4 = crop_recommendation_model.predict(data4)
            final_prediction4 = my_prediction4[0]

            temperature2, humidity2 = weather_fetch(city)
            data5 = np.array([[N + 35, P - 60, K - 77, temperature2, humidity2, ph + 1.9, rainfall - 100]])
            my_prediction5 = crop_recommendation_model.predict(data5)
            final_prediction5 = my_prediction5[0]

            hs = set([final_prediction,final_prediction2,final_prediction3,final_prediction4,final_prediction5])
            if state == 'Punjab':
                hs.add('Rice')
                hs.add('Maize')
            if state == 'Haryana':
                hs.add('Sugarcane')
                hs.add('Barley')
            if state == 'Rajasthan':
                hs.add('Wheat')
                hs.add('Sugarcane')
            if state == 'Uttar Pradesh':
                hs.add('Mushroom')
                hs.add('Betel Vine')
            if state == 'Bihar':
                hs.add('Wheat')
                hs.add('Pulses')
            if state == 'Gujarat':
                hs.add('Jowar')
                hs.add('Bajra')
            if state == 'Madhya Pradesh':
                hs.add('Moong')
                hs.add('Soyabean')
            if state == 'Maharashtra':
                hs.add('Cotton')
                hs.add('Sugarcane')
            if state == 'Chattisgarh':
                hs.add('Groundnut')
                hs.add('Pulses')
            if state == 'Jharkhand':
                hs.add('Rice')
                hs.add('Ragi')
            if state == 'Himachal Pradesh':
                hs.add('Potato')
                hs.add('Ginger')
            if state == 'Jammu & Kashmir':
                hs.add('Apple')
                hs.add('Wall Nuts')
            if state == 'West Bengal':
                hs.add('Jute')
                hs.add('Tea')
            if state == 'Karnataka':
                hs.add('Jowar')
                hs.add('Paddy')
            if state == 'Odisha':
                hs.add('Turmeric')

            hs.remove(final_prediction)

            predict_list = list(hs)





            return render_template('crop-result.html',prediction = final_prediction,predictionlist = predict_list)

        else:

            return render_template('try_again.html')

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/FertilizerData.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img = file.read()

        prediction = predict_image(img)

        prediction = Markup(str(disease_dic[prediction]))
        return render_template('disease-result.html', prediction=prediction)

    return render_template('disease.html')


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
