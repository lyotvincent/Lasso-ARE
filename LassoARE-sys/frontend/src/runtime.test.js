import test from "node:test";
import assert from "node:assert/strict";

import { runtimeLabel, sampleActionLabel } from "./runtime.js";


test("runtimeLabel describes CPU and CUDA backends", () => {
  assert.equal(runtimeLabel({ profile: "cpu" }), "CPU · PyTorch + Scanpy");
  assert.equal(runtimeLabel({ profile: "cuda" }), "CUDA · PyTorch + RAPIDS");
  assert.equal(
    runtimeLabel({ profile: "cuda", degraded: true }),
    "CUDA · Scanpy fallback",
  );
});

test("sampleActionLabel describes load and download states", () => {
  assert.equal(sampleActionLabel({ action: "load" }), "Load sample");
  assert.equal(
    sampleActionLabel({ action: "download" }),
    "Download & load sample",
  );
  assert.equal(
    sampleActionLabel({ action: "unavailable" }),
    "Sample not configured",
  );
});
