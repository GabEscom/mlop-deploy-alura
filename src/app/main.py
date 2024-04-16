from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
import os
from textblob import TextBlob
from googletrans import Translator
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

model = pickle.load(open('../../models/model.pkl', 'rb'))
columnas = ['area', 'modelo', 'estacionamiento']


app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')
basic_auth = BasicAuth(app)


@app.route('/')

def home():
    return 'Esta es mi primera API utilizando Flask en Debug mode = on'

@app.route('/sentimiento/<frase>')
@basic_auth.required

def sentimiento(frase):
    translator = Translator()
    tb = translator.translate(text=frase, src='es', dest='en').text
    tb = TextBlob(tb)
    polaridad = tb.sentiment.polarity
    return f'La polaridad de la frase es:  {polaridad}'

@app.route('/precio_casas/', methods=["POST"])
@basic_auth.required

def precio_casas():
    datos = request.get_json()
    datos_input = [datos[col] for col in columnas]
    precio = model.predict([datos_input])
    return jsonify(precios=precio[0])

app.run(debug=True, host='0.0.0.0')