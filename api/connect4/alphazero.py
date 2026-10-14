"""Backend-only AlphaZero feature flag, one-time model readiness and safe public status.

Disabled by default. When disabled nothing here imports torch or reads model
files, and gameplay is unchanged. When enabled, the approved artifact is
validated and loaded once per process at application start; an agent is only
ever reported available after that strict load and self-check succeed.
Public responses carry stable codes, never paths, exceptions or model data.
"""
import os
from pathlib import Path

from .state import GameError

ALPHAZERO = 'alphazero'
ALPHAZERO_AGENT_VERSION = 1
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / 'models/alphazero/az2-hc-g6-cnn-s42-v1.json'
SETTINGS = ('type', 'tactical_guard')
FAILURE_CODES = ('runtime_unavailable', 'artifact_missing', 'artifact_integrity', 'artifact_contract',
                 'unsupported_architecture', 'self_check_failed', 'initialization_failed')


def environment_config():
    value = os.environ.get('ALPHAZERO_ENABLED', 'false').lower()
    if value not in ('true', 'false', '1', '0'):
        raise ValueError('ALPHAZERO_ENABLED must be true or false')
    return {'ALPHAZERO_ENABLED': value in ('true', '1'),
            'ALPHAZERO_MANIFEST': os.environ.get('ALPHAZERO_MANIFEST') or str(DEFAULT_MANIFEST)}


class AlphaZeroService:
    def __init__(self, config, logger):
        enabled = config.get('ALPHAZERO_ENABLED', False)
        if type(enabled) is not bool:
            raise ValueError('ALPHAZERO_ENABLED must be boolean')
        self.enabled, self.model, self.failure = enabled, None, None
        if not enabled:
            return
        try:
            from games.connect4.alphazero_v2.production import ArtifactError, load_once
        except ImportError:
            self.failure = 'runtime_unavailable'
            logger.error('AlphaZero is enabled but the PyTorch inference runtime is not installed')
            return
        try:
            self.model = load_once(config['ALPHAZERO_MANIFEST'])
        except ArtifactError as error:
            self.failure = error.code if error.code in FAILURE_CODES else 'initialization_failed'
            logger.error('AlphaZero unavailable (%s): %s', self.failure, error)
        except Exception:
            self.failure = 'initialization_failed'
            logger.exception('AlphaZero initialization failed')
        else:
            logger.info('AlphaZero model %s ready (artifact sha256 %s)', self.model.model_id,
                        self.model.artifact_sha256)

    @property
    def available(self):
        return self.model is not None

    def public_status(self):
        if not self.enabled:
            return dict(type=ALPHAZERO, enabled=False, available=False, code='alphazero_disabled')
        if not self.available:
            return dict(type=ALPHAZERO, enabled=True, available=False, code='alphazero_unavailable')
        manifest = self.model.manifest
        return dict(
            type=ALPHAZERO, enabled=True, available=True, agent_version=ALPHAZERO_AGENT_VERSION,
            model=dict(id=self.model.model_id, description=manifest['source']['description'],
                       architecture=manifest['architecture']['identity'],
                       parameters=manifest['architecture']['parameters'],
                       artifact_sha256=self.model.artifact_sha256,
                       source_checkpoint_sha256=manifest['source']['checkpoint_sha256']),
            search=dict(algorithm='puct', simulations=manifest['search']['simulations'],
                        exploration=manifest['search']['exploration'], root_noise=False,
                        final_selection='maximum-visit', deterministic=True),
            options=dict(tactical_guard=dict(type='boolean', default=False,
                                             description='One-ply root safeguard after the unchanged search: '
                                                         'take an immediate win, otherwise avoid a move that '
                                                         'allows an immediate winning reply when a safe move exists.')))

    def require(self):
        """Raise the controlled public error unless the model is loaded and validated."""
        if not self.enabled:
            raise GameError('invalid_agent', 'The AlphaZero agent is not enabled on this server.')
        if not self.available:
            raise GameError('agent_unavailable', 'The AlphaZero agent is unavailable on this server. '
                                                 'Choose another opponent.', 503)
        return self.model


def validate_player(value):
    """Strict AlphaZero player settings; returns the canonical stored configuration."""
    if set(value) - set(SETTINGS):
        raise GameError('invalid_agent', 'AlphaZero accepts only tactical_guard (true or false). '
                                         'Its simulations and search settings are fixed.')
    guard = value.get('tactical_guard', False)
    if type(guard) is not bool:
        raise GameError('invalid_agent', 'AlphaZero tactical_guard must be true or false.')
    return {'type': ALPHAZERO, 'tactical_guard': guard}
