import assert from 'node:assert/strict';
import test from 'node:test';
import { api } from '../src/api.js';

test('a temporary network failure does not end task status tracking', async () => {
  const oldFetch = globalThis.fetch;
  const oldStorage = globalThis.localStorage;
  let calls = 0;
  globalThis.localStorage = { getItem: () => null };
  globalThis.fetch = async () => {
    calls += 1;
    if (calls === 1) throw new TypeError('network unavailable');
    return { ok: true, status: 200, json: async () => ({ status: 'PROCESSING' }) };
  };
  try {
    assert.equal((await api.getTaskStatus(12)).status, 'PROCESSING');
    assert.equal(calls, 2);
  } finally {
    globalThis.fetch = oldFetch;
    globalThis.localStorage = oldStorage;
  }
});

test('a missing task reports its error without repeating the request', async () => {
  const oldFetch = globalThis.fetch;
  const oldStorage = globalThis.localStorage;
  let calls = 0;
  globalThis.localStorage = { getItem: () => null };
  globalThis.fetch = async () => {
    calls += 1;
    return { ok: false, status: 404, json: async () => ({ detail: 'not found' }) };
  };
  try {
    await assert.rejects(() => api.getTaskStatus(12), (error) => error.status === 404);
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = oldFetch;
    globalThis.localStorage = oldStorage;
  }
});
