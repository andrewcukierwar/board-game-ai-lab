import { test } from 'node:test';
import assert from 'node:assert/strict';
import { apiBase } from '../src/api-config.js';

test('local builds use the proxy by default; Vite dev always uses the proxy', () => {
  assert.equal(apiBase(), '');
  assert.equal(apiBase('https://api.example.com', { development: true }), '');
});

test('separate-host builds use a normalized origin and support local cross-origin verification', () => {
  assert.equal(apiBase(' https://api.example.com/ ', { render: true }), 'https://api.example.com');
  assert.equal(apiBase('http://localhost:8000'), 'http://localhost:8000');
});

test('Render requires an explicit HTTPS origin; endpoint paths and credentials fail', () => {
  for (const value of ['', 'http://api.example.com', '/v1', 'https://api.example.com/v1',
    'https://user:secret@api.example.com', 'https://api.example.com?q=1', 'https://api.example.com#x']) {
    assert.throws(() => apiBase(value, { render: true }), /VITE_API_BASE/);
  }
});
