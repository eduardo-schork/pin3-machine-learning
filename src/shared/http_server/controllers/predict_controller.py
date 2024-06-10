from flask import Blueprint, request, jsonify
from ..services.image_service import preprocess_image
from shared.http_server.utils.format_predict_output import format_predict_output
from shared.machine_learning.load_latest_model import load_latest_model

predict_bp = Blueprint("predict_bp", __name__)

vgg16_model = load_latest_model("vgg16")
inceptionv3_model = load_latest_model("inceptionv3")
convnet_model = load_latest_model("convnet")


@predict_bp.route("/predict", methods=["POST"])
def predict_all_models():
    try:
        img_array = preprocess_image(request.files.get("image"))
        if img_array is None:
            return jsonify({"error": "Imagem inválida"}), 400

        vgg16_output = format_predict_output(vgg16_model.predict(img_array)[0])
        inceptionv3_output = format_predict_output(
            inceptionv3_model.predict(img_array)[0]
        )

        combined_output = {"vgg16": vgg16_output, "inceptionv3": inceptionv3_output}

        return jsonify(combined_output)
    except Exception as e:
        return jsonify({"error": f"Erro ao realizar a previsão: {str(e)}"}), 500


@predict_bp.route("/predict/vgg16", methods=["POST"])
def predict_vgg16():
    img_array = preprocess_image(request.files.get("image"))
    if img_array is None:
        return jsonify({"error": "Imagem inválida"}), 400

    try:
        prediction = vgg16_model.predict(img_array)
        formatted_output = format_predict_output(prediction[0])
        return jsonify(formatted_output)
    except Exception as e:
        return jsonify({"error": f"Erro ao realizar a previsão: {str(e)}"}), 500


@predict_bp.route("/predict/inceptionv3", methods=["POST"])
def predict_inceptionv3():
    img_array = preprocess_image(request.files.get("image"))
    if img_array is None:
        return jsonify({"error": "Imagem inválida"}), 400

    try:
        prediction = inceptionv3_model.predict(img_array)
        formatted_output = format_predict_output(prediction[0])
        return jsonify(formatted_output)
    except Exception as e:
        return jsonify({"error": f"Erro ao realizar a previsão: {str(e)}"}), 500


@predict_bp.route("/predict/convnet", methods=["POST"])
def predict_convnet():
    img_array = preprocess_image(request.files.get("image"))
    if img_array is None:
        return jsonify({"error": "Imagem inválida"}), 400

    try:
        prediction = convnet_model.predict(img_array)
        formatted_output = format_predict_output(prediction[0])
        return jsonify(formatted_output)
    except Exception as e:
        return jsonify({"error": f"Erro ao realizar a previsão: {str(e)}"}), 500
