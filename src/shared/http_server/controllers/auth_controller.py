from flask import Blueprint, request, jsonify
import json
import requests
from firebase_admin import auth
from ..services.firebase_service import get_request_auth_token

auth_bp = Blueprint("auth_bp", __name__)


@auth_bp.route("/secure-endpoint", methods=["POST"])
def secure_endpoint():
    try:
        uid = get_request_auth_token(request)
        user = auth.get_user(uid)
        user_data = {"uid": user.uid, "email": user.email, "name": user.display_name}
        return jsonify({"status": "success", "user": user_data}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 401


@auth_bp.route("/login", methods=["POST"])
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
            print(response_data["error"]["message"])
            return jsonify({"error": response_data["error"]["message"]}), 401

        id_token = response_data.get("idToken")
        return jsonify({"token": id_token}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@auth_bp.route("/register", methods=["POST"])
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
            return jsonify({"error": response_data["error"]["message"]}), 401

        id_token = response_data.get("idToken")

        user = auth.get_user_by_email(email)
        auth.update_user(user.uid, display_name=name)

        return jsonify({"token": id_token}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@auth_bp.route("/anonymSign", methods=["POST"])
def anonymSign():
    try:
        file_path = "./google-services-key.json"
        with open(file_path, "r") as file:
            firebase_api_key = json.load(file)["apiKey"]

        FIREBASE_AUTH_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={firebase_api_key}"
        response = requests.post(FIREBASE_AUTH_URL, json={"returnSecureToken": True})

        response_data = response.json()

        if "error" in response_data:
            return jsonify({"error": response_data["error"]["message"]}), 401

        id_token = response_data.get("idToken")
        return jsonify({"token": id_token}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@auth_bp.route("/logoff", methods=["POST"])
def logoff():
    try:
        user_id = get_request_auth_token(request)

        if not user_id:
            return jsonify({"error": "User ID is missing"}), 400

        auth.revoke_refresh_tokens(user_id)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
