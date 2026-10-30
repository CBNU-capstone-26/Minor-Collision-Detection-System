import assert from "node:assert/strict";
import test from "node:test";

import { api } from "../src/api.js";

test("getVehicleDetectionTask requests the task status endpoint", async () => {
  const originalFetch = globalThis.fetch;
  const originalStorage = globalThis.localStorage;
  globalThis.localStorage = { getItem: () => null };
  globalThis.fetch = async (url, options) => {
    assert.equal(url, "/api/videos/7/vehicle-detection-tasks/12");
    assert.equal(options.method, "GET");
    return {
      ok: true,
      status: 200,
      json: async () => ({ task_id: 12, status: "PROCESSING" }),
    };
  };

  try {
    const result = await api.getVehicleDetectionTask(7, 12);
    assert.equal(result.task_id, 12);
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.localStorage = originalStorage;
  }
});
