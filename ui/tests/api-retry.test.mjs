import assert from "node:assert/strict";
import test from "node:test";

import { retryRequest, withRequestTimeout } from "../src/api.js";

test("retryRequest retries a transient list failure before succeeding", async () => {
  let attempts = 0;

  const result = await retryRequest(async () => {
    attempts += 1;
    if (attempts < 3) throw new Error("temporary failure");
    return "videos";
  }, { attempts: 3, delayMs: 0 });

  assert.equal(result, "videos");
  assert.equal(attempts, 3);
});

test("retryRequest returns the final failure after all attempts", async () => {
  let attempts = 0;

  await assert.rejects(
    () => retryRequest(async () => {
      attempts += 1;
      throw new Error("unavailable");
    }, { attempts: 3, delayMs: 0 }),
    /unavailable/,
  );
  assert.equal(attempts, 3);
});

test("withRequestTimeout aborts a request that does not answer", async () => {
  await assert.rejects(
    () => withRequestTimeout(
      (signal) => new Promise((_, reject) => {
        signal.addEventListener("abort", () => reject(new Error("timed out")));
      }),
      1,
    ),
    /timed out/,
  );
});
