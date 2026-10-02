"""One bounded, backend-only Responses API call; no retries or SDK globals."""
import json
from http.client import HTTPException, HTTPSConnection
from socket import SHUT_RDWR
from threading import Event, Timer
from time import monotonic

from .state import GameError


class OpenAIExplanationProvider:
    def __init__(self, connection_factory=HTTPSConnection):
        self.connection_factory = connection_factory

    def generate(self, *, api_key, model, max_tokens, timeout, instructions, evidence, schema):
        connection = response = timer = None
        expired = Event()
        deadline = monotonic() + timeout
        try:
            connection = self.connection_factory('api.openai.com', timeout=timeout)
            body = json.dumps({
                'model': model, 'store': False, 'max_output_tokens': max_tokens,
                'instructions': instructions,
                'input': [{'role': 'user', 'content': json.dumps(evidence, ensure_ascii=True)}],
                'text': {'format': {'type': 'json_schema', 'name': 'connect4_explanation',
                                    'strict': True, 'schema': schema}},
            }).encode('utf-8')
            # Socket timeouts alone reset on each read. A slow stream of headers
            # or body bytes must not keep a gameplay thread/permit indefinitely.
            # Connect first so the watchdog owns the socket even if HTTPResponse
            # detaches it for a Connection: close response. DNS uses OS timeouts.
            if hasattr(connection, 'connect') and connection.sock is None:
                connection.connect()
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError
            sock = getattr(connection, 'sock', None)
            if sock is not None:
                sock.settimeout(remaining)

            def interrupt():
                expired.set()
                if sock is not None:
                    try:
                        sock.shutdown(SHUT_RDWR)
                    except OSError:
                        pass

            timer = Timer(remaining, interrupt)
            timer.daemon = True
            timer.start()
            connection.request('POST', '/v1/responses', body=body, headers={
                'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json',
            })
            response = connection.getresponse()
            if expired.is_set() or monotonic() >= deadline:
                raise TimeoutError
            # Never forward provider bodies, headers, model names or credentials.
            if response.status != 200:
                raise GameError('explanation_unavailable', 'The explanation provider is unavailable. Try again later.', 503)
            raw = response.read(65537)
            if expired.is_set() or monotonic() >= deadline:
                raise TimeoutError
            if len(raw) > 65536:
                raise ValueError('oversize response')
            data = json.loads(raw)
            if data.get('status') != 'completed':
                raise ValueError('incomplete response')
            parts = [part for item in data['output'] if item['type'] == 'message'
                     for part in item['content']]
            if len(parts) != 1 or parts[0]['type'] != 'output_text':
                raise ValueError('missing text or refusal')
            return json.loads(parts[0]['text'])
        except TimeoutError:
            raise GameError('explanation_timeout', 'The explanation timed out. You can retry.', 504) from None
        except (OSError, RuntimeError, HTTPException):
            if expired.is_set() or monotonic() >= deadline:
                raise GameError('explanation_timeout', 'The explanation timed out. You can retry.', 504) from None
            raise GameError('explanation_unavailable', 'The explanation provider is unavailable. Try again later.', 503) from None
        except (ValueError, KeyError, TypeError, AttributeError):
            raise GameError('invalid_explanation', 'The provider returned an unusable explanation. You can retry.', 502) from None
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()
