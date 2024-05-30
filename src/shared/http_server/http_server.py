import json
import traceback
import requests

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array

from shared.http_server.format_predict_output import format_predict_output
from shared.http_server.validate_image_input import validate_image_input
from shared.machine_learning.load_latest_model import load_latest_model

vgg16_model = load_latest_model("vgg16")
inceptionv3_model = load_latest_model("inceptionv3")
convnet_model = load_latest_model("convnet")

import firebase_admin
from firebase_admin import credentials, auth

cred = credentials.Certificate("./service-account-key.json")
firebase_admin.initialize_app(cred)

app = Flask(__name__)
CORS(
    app,
    origins="http://192.168.4.12:3000",
    methods=["GET", "POST", "OPTIONS"],
    allow_headers="*",
)


@app.route("/")
def index():
    return render_template("index.html")


def _getRequestAuthToken(request):
    token = request.headers.get("Authorization").split(" ")[1]

    decoded_token = auth.verify_id_token(token)
    uid = decoded_token["uid"]
    return uid


@app.route("/secure-endpoint", methods=["POST"])
def secure_endpoint():
    try:
        uid = _getRequestAuthToken(request)
        # Busque os dados do usuário no Firebase
        user = auth.get_user(uid)
        return jsonify({"status": "success", "user": user.__dict__}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 401


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    try:
        file_path = "./google-services-key.json"
        with open(file_path, "r") as file:
            firebase_api_key = json.load(file)["apiKey"]

        FIREBASE_AUTH_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={firebase_api_key}"
        response = requests.post(
            FIREBASE_AUTH_URL,
            json={"email": email, "password": password, "returnSecureToken": True},
        )

        response_data = response.json()

        if "error" in response_data:
            return jsonify({"error": response_data["error"]["message"]}), 401

        id_token = response_data.get("idToken")
        return jsonify({"token": id_token}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def preprocess_image(img):
    try:
        is_valid, img = validate_image_input(img)

        if not is_valid:
            return None

        img_array = img_to_array(img)

        img_array = img_array.reshape(
            (1, img_array.shape[0], img_array.shape[1], img_array.shape[2])
        )

        img_array = img_array / 255.0

        return img_array
    except Exception as e:
        traceback.print_exc()


@app.route("/predict", methods=["POST"])
def predict_all_models():
    try:
        uid = _getRequestAuthToken(request)
        print(uid)
        img_array = preprocess_image(request.files.get("image"))
        if img_array is None:
            return jsonify({"error": "Imagem inválida"}), 400

        vgg16_prediction = vgg16_model.predict(img_array)
        vgg16_output = format_predict_output(vgg16_prediction[0])

        inceptionv3_prediction = inceptionv3_model.predict(img_array)
        inceptionv3_output = format_predict_output(inceptionv3_prediction[0])

        convnet_prediction = convnet_model.predict(img_array)
        convnet_output = format_predict_output(convnet_prediction[0])

        combined_output = {
            "vgg16": vgg16_output,
            "inceptionv3": inceptionv3_output,
            "convnet": convnet_output,
        }

        return jsonify(combined_output)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erro ao realizar a previsão: {str(e)}"}), 500


@app.route("/predict/vgg16", methods=["POST"])
def predict_vgg16():
    img_array = preprocess_image(request.files.get("image"))
    if img_array is None:
        return jsonify({"error": "Imagem inválida"}), 400

    try:
        prediction = vgg16_model.predict(img_array)
        formatted_output = format_predict_output(prediction[0])
        return jsonify(formatted_output)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erro ao realizar a previsão: {str(e)}"}), 500


@app.route("/predict/inceptionv3", methods=["POST"])
def predict_inceptionv3():
    img_array = preprocess_image(request.files.get("image"))
    if img_array is None:
        return jsonify({"error": "Imagem inválida"}), 400

    try:
        prediction = inceptionv3_model.predict(img_array)
        formatted_output = format_predict_output(prediction[0])
        return jsonify(formatted_output)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erro ao realizar a previsão: {str(e)}"}), 500


@app.route("/predict/convnet", methods=["POST"])
def predict_convnet():
    img_array = preprocess_image(request.files.get("image"))
    if img_array is None:
        return jsonify({"error": "Imagem inválida"}), 400

    try:
        prediction = convnet_model.predict(img_array)
        formatted_output = format_predict_output(prediction[0])
        return jsonify(formatted_output)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erro ao realizar a previsão: {str(e)}"}), 500


def run_http_server():
    app.run(host="0.0.0.0", port=3000)
