import os
from pathlib import Path
from threading import BoundedSemaphore

from dotenv import load_dotenv

from flask import Flask, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from api.connect4 import bp as connect4_bp
from api.connect4.state import GameStore
from api.connect4.explanations import ExplanationService, environment_config
from api.cors import configure_cors
from api.public_limits import PublicApiLimiter, environment_config as public_limit_config


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
    app.config.update(public_limit_config())
    if config:
        app.config.update(config)
    app.extensions['public_api_limiter'] = PublicApiLimiter(app.config)
    app.extensions['connect4_search_gate'] = BoundedSemaphore(
        app.config['PUBLIC_SEARCH_CONCURRENCY'])
    app.extensions['connect4_games'] = GameStore(
        capacity=app.config['GAME_SESSION_CAPACITY'], ttl=app.config['GAME_SESSION_TTL'])
    app.extensions['connect4_explanations'] = ExplanationService(app.config)
    app.register_blueprint(connect4_bp, url_prefix='/v1/connect4')
    configure_cors(app)

    @app.before_request
    def guard_public_api():
        from flask import request
        if not request.path.startswith('/v1/connect4/'):
            return None
        retry_after = app.extensions['public_api_limiter'].check(request.method, request.path)
        if retry_after is None:
            return None
        response = jsonify(
            error='The game API is handling too many requests. Try again shortly.',
            code='rate_limited')
        response.status_code = 429
        response.headers['Retry-After'] = str(retry_after)
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error='Request is too large. Send a smaller JSON request.',
                       code='request_too_large'), 413

    return app


app = create_app()
