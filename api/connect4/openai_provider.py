"""One bounded, backend-only Responses API call; no retries or SDK globals."""
import json
from http.client import HTTPSConnection

from .state import GameError


class OpenAIExplanationProvider:
    def __init__(self, connection_factory=HTTPSConnection):
        self.connection_factory = connection_factory

    def generate(self, *, api_key, model, max_tokens, timeout, instructions, evidence, schema):
        connection = self.connection_factory('api.openai.com', timeout=timeout)
        try:
            body = json.dumps({
                'model': model, 'store': False, 'max_output_tokens': max_tokens,
                'instructions': instructions,
                'input': [{'role': 'user', 'content': json.dumps(evidence, ensure_ascii=True)}],
                'text': {'format': {'type': 'json_schema', 'name': 'connect4_explanation',
                                    'strict': True, 'schema': schema}},
            }).encode('utf-8')
            connection.request('POST', '/v1/responses', body=body, headers={
                'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json',
            })
            response = connection.getresponse()
            # Never forward provider bodies, headers, model names or credentials.
            if response.status != 200:
                raise GameError('explanation_unavailable', 'The explanation provider is unavailable. Try again later.', 503)
            raw = response.read(65537)
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
        except (OSError, RuntimeError):
            raise GameError('explanation_unavailable', 'The explanation provider is unavailable. Try again later.', 503) from None
        except (ValueError, KeyError, TypeError, AttributeError):
            raise GameError('invalid_explanation', 'The provider returned an unusable explanation. You can retry.', 502) from None
        finally:
            connection.close()
