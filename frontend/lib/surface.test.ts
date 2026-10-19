import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { interpolate, parseSurface, smilePoints, surfaceGrid } from "./surface.ts";
import type { Contract } from "./types.ts";

const snapshot = JSON.parse(readFileSync(new URL("./demo.json", import.meta.url), "utf8"));
const base = parseSurface(snapshot).contracts[0];
const quote = (strike: number, iv_pct: number, expiration = "2026-11-20", T = 1 / 12): Contract => ({ ...base, strike, iv_pct, expiration, T });

test("interpolation is linear and never extrapolates beyond observed strikes", () => {
  const points = [{ strike: 90, iv: 30 }, { strike: 110, iv: 20 }];
  assert.equal(interpolate(points, 100), 25);
  assert.equal(interpolate(points, 90), 30);
  assert.equal(interpolate(points, 89), null);
  assert.equal(interpolate(points, 111), null);
  assert.equal(interpolate([], 100), null);
});

test("duplicate strikes are averaged for display without modifying raw quotes", () => {
  const contracts = [quote(110, 25), quote(90, 30), quote(90, 32)];
  assert.deepEqual(smilePoints(contracts, "2026-11-20"), [{ strike: 90, iv: 31 }, { strike: 110, iv: 25 }]);
  assert.deepEqual(contracts.map((c) => c.iv_pct), [25, 30, 32]);
});

test("surface respects each expiration's independent strike coverage", () => {
  const grid = surfaceGrid([quote(90, 30), quote(110, 20), quote(100, 28, "2026-12-18", 0.2), quote(120, 24, "2026-12-18", 0.2)], 4);
  assert.deepEqual(grid.x, [90, 100, 110, 120]);
  assert.deepEqual(grid.z, [[30, 25, 20, null], [null, 28, 26, 24]]);
  assert.equal(grid.y[1], 73);
});

test("single-strike coverage does not invent a smile", () => {
  assert.equal(interpolate([{ strike: 100, iv: 25 }], 99), null);
  assert.equal(interpolate([{ strike: 100, iv: 25 }], 100), 25);
  assert.deepEqual(surfaceGrid([]).x, []);
});

test("saved snapshot conforms to the frontend data contract", () => {
  assert.equal(parseSurface(snapshot).metrics.symbol, "AAPL");
  assert.ok(parseSurface(snapshot).contracts.length > 0);
});

test("empty or invalid API data is rejected before replacing the current chart", () => {
  for (const bad of [null, {}, { ...snapshot, contracts: [] }, { ...snapshot, metrics: { ...snapshot.metrics, spot_price: NaN } },
    { ...snapshot, contracts: [{ ...base, iv_pct: Infinity }] }, { ...snapshot, warnings: [null] }]) {
    assert.throws(() => parseSurface(bad));
  }
});
