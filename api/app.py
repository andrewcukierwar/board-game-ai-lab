import os
from pathlib import Path

from dotenv import load_dotenv

from flask import Flask, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from api.connect4 import bp as connect4_bp
from api.connect4.state import GameStore
from api.connect4.explanations import ExplanationService, environment_config
from api.cors import configure_cors


def victor_research_enabled():
    """Opt-in research agent flag; disabled unless explicitly true."""
    value = os.environ.get('VICTOR_RESEARCH_ENABLED', 'false').lower()
    if value not in ('true', 'false', '1', '0'):
        raise ValueError('VICTOR_RESEARCH_ENABLED must be true or false')
    return value in ('true', '1')


def create_app(config=None):
    # Explicit backend root file; real environment variables always take precedence.
    load_dotenv(Path(__file__).resolve().parent.parent / '.env', override=False)
    app = Flask(__name__)
    app.config.from_mapping(MAX_CONTENT_LENGTH=4096, GAME_SESSION_CAPACITY=128, GAME_SESSION_TTL=1800,
                            CORS_ALLOWED_ORIGINS=os.environ.get('CORS_ALLOWED_ORIGINS', ''),
                            SOURCE_COMMIT=os.environ.get('EVALUATION_SOURCE_COMMIT'),
                            VICTOR_RESEARCH_ENABLED=victor_research_enabled())
    app.config.update(environment_config())
    if config:
        app.config.update(config)
    app.extensions['connect4_games'] = GameStore(
        capacity=app.config['GAME_SESSION_CAPACITY'], ttl=app.config['GAME_SESSION_TTL'])
    app.extensions['connect4_explanations'] = ExplanationService(app.config)
    app.register_blueprint(connect4_bp, url_prefix='/v1/connect4')
    configure_cors(app)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error='Request is too large. Send a smaller JSON request.',
                       code='request_too_large'), 413

    return app


app = create_app()
