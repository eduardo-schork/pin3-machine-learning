import json
import traceback
import requests

from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array

from shared.http_server.format_predict_output import format_predict_output
from shared.http_server.validate_image_input import validate_image_input
from shared.machine_learning.load_latest_model import load_latest_model

from flask import Flask, request, jsonify, render_template
from firebase_admin import credentials, initialize_app, storage, db
from google.cloud import storage as gcs

vgg16_model = load_latest_model("vgg16")
inceptionv3_model = load_latest_model("inceptionv3")
convnet_model = load_latest_model("convnet")

import firebase_admin
from firebase_admin import credentials, auth

cred = credentials.Certificate("./service-account-key.json")
firebase_admin.initialize_app(
    cred,
    {
        "storageBucket": "gs://pin3-42a6f.appspot.com",
        "databaseURL": "https://pin3-42a6f-default-rtdb.firebaseio.com/",
    },
)

gcs_client = gcs.Client.from_service_account_json("./service-account-key.json")
bucket = gcs_client.bucket("post-images")


app = Flask(__name__)

CORS(
    app,
    origins="http://192.168.1.13:3000",
    methods=["GET", "POST", "OPTIONS"],
    allow_headers="*",
)


@app.route("/post-feed", methods=["POST"])
def post_feed():
    image = request.files.get["image"]
    if image is null:
        print("Image is required")
        return jsonify({"error": "Image is required"}), 400

    user_id = request.form.get("user_id")
    #send_notification_to_user(user_id)
    
    predicted_percentage = request.form.get("predicted_percentage")
    predicted_class = request.form.get("predicted_class")
    feedback_class = request.form.get("feedback_class")
    model_type = request.form.get("model_type")

    if (
        not user_id
        or not predicted_percentage
        or not predicted_class
        or not feedback_class
        or not model_type
    ):
        print("All fields are required")
        return jsonify({"error": "All fields are required"}), 400

    try:
        blob = bucket.blob(f"{user_id}/{image.filename}")
        blob.upload_from_file(image, content_type=image.content_type)
        image_url = blob.public_url

        ref = db.reference("tb_feed")
        new_post = ref.push(
            {
                "user_id": user_id,
                "image_url": image_url,
                "predicted_percentage": predicted_percentage,
                "predicted_class": predicted_class,
                "feedback_class": feedback_class,
                "model_type": model_type,
            }
        )

        return jsonify({"status": "success", "post_id": new_post.key}), 201

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500

def send_notification_to_user(uid):
    console.log("oi")
    user_ref = db.reference("users").child(user_id)
    user_data = user_ref.get()
    if user_data and "device_token" in user_data:
        device_token = user_data["device_token"]
        # Construir a mensagem da notificação
        message = messaging.Message(
            data={
                "title": "Nova postagem",
                "body": "Uma nova postagem foi feita.",
                #"post_id": new_post.key  # Se necessário, você pode incluir o ID da postagem na notificação
            },
            token=device_token,
        )
        # Enviar a notificação
        response = messaging.send(message)
        print("Successfully sent message:", response)

@app.route("/get-posts", methods=["GET"])
def get_posts():
    try:
        ref = db.reference("tb_feed")
        posts = ref.get()

        posts_list = []
        if posts:
            for key, value in posts.items():
                post = value
                post["id"] = key
                posts_list.append(post)

        return jsonify({"status": "success", "posts": posts_list}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


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

        user = auth.get_user(uid)

        user_data = {
            "uid": user.uid,
            "email": user.email,
            "name": user.display_name,
        }
        send_notification_to_user(user.id)
        return jsonify({"status": "success", "user": user_data}), 200
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

        print(response_data)

        if "error" in response_data:
            return jsonify({"error": response_data["error"]["message"]}), 401

        id_token = response_data.get("idToken")
        return jsonify({"token": id_token}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/register", methods=["POST"])
def register():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not email or not password or not name:
        return jsonify({"error": "Name, email and password are required"}), 400

    try:
        file_path = "./google-services-key.json"
        with open(file_path, "r") as file:
            firebase_api_key = json.load(file)["apiKey"]

        FIREBASE_AUTH_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={firebase_api_key}"
        response = requests.post(
            FIREBASE_AUTH_URL,
            json={"email": email, "password": password, "returnSecureToken": True},
        )

        response_data = response.json()

        if "error" in response_data:
            print(response_data)
            return jsonify({"error": response_data["error"]["message"]}), 401

        id_token = response_data.get("idToken")

        user = auth.get_user_by_email(email)
        auth.update_user(user.uid, display_name=name)

        return jsonify({"token": id_token}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logoff", methods=["POST"])
def logoff():
    try:
        user_id = _getRequestAuthToken(request)

        if not user_id:
            return jsonify({"error": "User ID is missing"}), 400

        auth.revoke_refresh_tokens(user_id)

        return jsonify({"status": "success"}), 200
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
