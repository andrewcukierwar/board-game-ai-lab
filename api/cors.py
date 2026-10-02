"""Exact-origin CORS for the separately hosted Connect 4 frontend."""
from urllib.parse import urlsplit

from flask import request


def configure_cors(app):
    origins = set()
    for value in app.config['CORS_ALLOWED_ORIGINS'].split(','):
        origin = value.strip()
        if not origin:
            continue
        url = urlsplit(origin)
        if (url.scheme not in ('http', 'https') or not url.hostname or
                url.username or url.password or url.path or url.query or url.fragment or
                '*' in origin or any(char.isspace() for char in origin) or
                origin != f'{url.scheme}://{url.netloc}'):
            raise ValueError('CORS_ALLOWED_ORIGINS must contain comma-separated exact HTTP(S) origins, without paths or wildcards.')
        # Also reject malformed ports at startup.
        url.port
        origins.add(origin)

    @app.after_request
    def cors(response):
        if request.path.startswith('/v1/connect4/'):
            response.vary.add('Origin')
            origin = request.headers.get('Origin')
            if origin in origins:
                response.headers['Access-Control-Allow-Origin'] = origin
                if request.method == 'OPTIONS':
                    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
                    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            # No credentials or wildcard origins. CORS controls browser access,
            # not authentication; same-origin proxy clients need no CORS headers.
        return response
