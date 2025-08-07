from flask import Flask
from api.connect4 import bp as connect4_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(connect4_bp, url_prefix="/v1/connect4")
    return app

app = create_app()          # so gunicorn can import "api.app:app"
