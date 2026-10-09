"""Bounded, process-local public API budgets.

Tailscale Funnel may present many users as one proxy address. Accordingly, these
limits are deliberately GLOBAL, not keyed to an untrusted forwarded-IP header.
Only one Gunicorn worker / one application instance is supported.
"""
from math import ceil, isfinite
import os
from threading import Lock
from time import monotonic


def environment_config():
    """Read nonsecret settings; validated after create_app test overrides."""
    defaults = {
        'PUBLIC_RATE_LIMIT_ENABLED': 'true',
        'PUBLIC_REQUEST_RATE_PER_SECOND': '60',
        'PUBLIC_REQUEST_BURST': '160',
        'PUBLIC_START_RATE_PER_SECOND': '8',
        'PUBLIC_START_BURST': '40',
        'PUBLIC_SEARCH_CONCURRENCY': '2',
    }
    result = {}
    for key, default in defaults.items():
        raw = os.environ.get(key, default)
        if key == 'PUBLIC_RATE_LIMIT_ENABLED':
            if raw.lower() not in ('true', 'false', '1', '0'):
                raise ValueError(f'{key} must be true or false')
            result[key] = raw.lower() in ('true', '1')
        else:
            try:
                result[key] = float(raw) if 'RATE_PER_SECOND' in key else int(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(f'{key} has an invalid numeric value') from error
    return result


def validate_config(config):
    enabled = config['PUBLIC_RATE_LIMIT_ENABLED']
    if type(enabled) is not bool:
        raise ValueError('PUBLIC_RATE_LIMIT_ENABLED must be boolean')
    for key in ('PUBLIC_REQUEST_RATE_PER_SECOND', 'PUBLIC_START_RATE_PER_SECOND'):
        value = config[key]
        if type(value) not in (int, float) or not isfinite(value) or not 0 < value <= 1000:
            raise ValueError(f'{key} must be finite and between 0 and 1000')
    for key in ('PUBLIC_REQUEST_BURST', 'PUBLIC_START_BURST'):
        value = config[key]
        if type(value) is not int or not 1 <= value <= 10000:
            raise ValueError(f'{key} must be an integer between 1 and 10000')
    searches = config['PUBLIC_SEARCH_CONCURRENCY']
    if type(searches) is not int or not 1 <= searches <= 3:
        raise ValueError('PUBLIC_SEARCH_CONCURRENCY must be an integer from 1 to 3')


class _Bucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last = None

    def replenish(self, now):
        if self.last is not None:
            self.tokens = min(self.capacity, self.tokens + max(0, now - self.last) * self.rate)
        self.last = now

    def wait_for_token(self):
        return max(0.0, 1.0 - self.tokens) / self.rate


class PublicApiLimiter:
    """Two constant-memory token buckets; decisions atomically reserve both tokens."""

    def __init__(self, config, clock=monotonic):
        validate_config(config)
        self.enabled = config['PUBLIC_RATE_LIMIT_ENABLED']
        self.clock = clock
        self._lock = Lock()
        self._requests = _Bucket(config['PUBLIC_REQUEST_RATE_PER_SECOND'],
                                 config['PUBLIC_REQUEST_BURST'])
        self._starts = _Bucket(config['PUBLIC_START_RATE_PER_SECOND'],
                               config['PUBLIC_START_BURST'])

    def check(self, method, path):
        """Return Retry-After seconds if denied; None if permitted.

        OPTIONS is excluded so browser CORS preflights cannot be exhausted by
        ordinary gameplay. Health probes are also excluded.
        """
        if not self.enabled or method == 'OPTIONS' or path == '/v1/connect4/health':
            return None
        is_start = method == 'POST' and path == '/v1/connect4/start_game'
        with self._lock:
            now = self.clock()
            self._requests.replenish(now)
            if is_start:
                self._starts.replenish(now)
            request_wait = self._requests.wait_for_token()
            start_wait = self._starts.wait_for_token() if is_start else 0.0
            wait = max(request_wait, start_wait)
            if wait > 0:
                return max(1, ceil(wait))
            self._requests.tokens -= 1.0
            if is_start:
                self._starts.tokens -= 1.0
        return None
