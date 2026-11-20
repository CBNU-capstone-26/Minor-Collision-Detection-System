import assert from 'node:assert/strict';
import test from 'node:test';
import { api } from '../src/api.js';

test('bulk delete returns only failed video IDs after all requests finish', async () => {
  const oldFetch = globalThis.fetch;
  const oldStorage = globalThis.localStorage;
  globalThis.localStorage = { getItem: () => null };
  globalThis.fetch = async (url) => url.endsWith('/2')
    ? { ok: false, status: 500, json: async () => ({ detail: 'failed' }) }
    : { ok: true, status: 204 };
  try {
    assert.deepEqual(await api.deleteVideos([1, 2, 3]), { deletedCount: 2, failedIds: [2] });
  } finally {
    globalThis.fetch = oldFetch;
    globalThis.localStorage = oldStorage;
  }
});
