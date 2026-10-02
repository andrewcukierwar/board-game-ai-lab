from flask import Flask, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from api.connect4 import bp as connect4_bp
from api.connect4.state import GameStore


def create_app(config=None):
    app = Flask(__name__)
    app.config.from_mapping(MAX_CONTENT_LENGTH=4096, GAME_SESSION_CAPACITY=128, GAME_SESSION_TTL=1800)
    if config:
        app.config.update(config)
    app.extensions['connect4_games'] = GameStore(
        capacity=app.config['GAME_SESSION_CAPACITY'], ttl=app.config['GAME_SESSION_TTL'])
    app.register_blueprint(connect4_bp, url_prefix='/v1/connect4')

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error='Request is too large. Send only the game configuration or move.',
                       code='request_too_large'), 413

    return app


app = create_app()
