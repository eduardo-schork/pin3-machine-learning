from flask import Flask
from flask_cors import CORS
from .controllers.auth_controller import auth_bp
from .controllers.post_controller import post_bp
from .controllers.predict_controller import predict_bp


def create_app():
    app = Flask(__name__)
    CORS(
        app,
        methods=["GET", "POST", "OPTIONS"],
        allow_headers="*",
    )

    app.register_blueprint(auth_bp)
    app.register_blueprint(post_bp)
    app.register_blueprint(predict_bp)

    return app


app = create_app()


def run_http_server():
    app.run(host="0.0.0.0", port=3000)
