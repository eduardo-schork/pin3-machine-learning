from flask import Blueprint, request, jsonify
from ..services.firebase_service import (
    get_request_auth_token,
    save_image_to_bucket,
    save_post_to_db,
    get_posts_from_db,
)
from werkzeug.utils import secure_filename

post_bp = Blueprint("post_bp", __name__)


@post_bp.route("/post-feed", methods=["POST"])
def post_feed():
    try:
        image = request.files.get("image")

        if image is None:
            print("Image is required")
            return jsonify({"error": "Image is required"}), 400

        user_id = get_request_auth_token(request)

        feedback_class = request.form.get("feedback_class")
        predicted_class = request.form.get("predicted_class")
        predicted_percentage = request.form.get("predicted_percentage")
        model_type = request.form.get("model_type")

        if (
            not user_id
            or not feedback_class
            or not predicted_class
            or not predicted_percentage
            or not model_type
        ):
            print("All fields are required")
            return jsonify({"error": "All fields are required"}), 400

        image_filename = secure_filename(image.filename)
        image_content = image.read()
        image_type = image.content_type

        image_url = save_image_to_bucket(
            user_id, image_content, image_filename, image_type
        )

        save_post_to_db(
            user_id,
            image_url,
            feedback_class,
            predicted_class,
            predicted_percentage,
            model_type,
        )

        return jsonify({"status": "success"}), 201

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@post_bp.route("/get-posts", methods=["GET"])
def get_posts():
    try:
        uid = get_request_auth_token(request)
        posts_list = get_posts_from_db()
        return jsonify({"status": "success", "posts": posts_list}), 200
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)}), 500
