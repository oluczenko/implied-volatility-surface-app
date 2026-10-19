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


test("old snapshots normalize absent IV spread endpoints and quality counts", () => {
  const data = parseSurface(snapshot);
  assert.equal(data.quality, null);
  assert.ok(data.contracts.every((c) => c.iv_bid === null && c.iv_ask === null));
});

test("bid/ask IV endpoints remain distinct from last-trade estimates", () => {
  const midpoint = { ...base, price_source: "midpoint", iv_bid: .2, iv_ask: .4 };
  const valid = parseSurface({ ...snapshot, contracts: [midpoint] });
  assert.equal(valid.contracts[0].iv_bid, .2);
  for (const bad of [
    { ...midpoint, iv_bid: .5, iv_ask: .2 },
    { ...midpoint, iv_bid: Infinity },
    { ...midpoint, iv_ask: -1 },
    { ...midpoint, price_source: "last_trade" },
  ]) assert.throws(() => parseSurface({ ...snapshot, contracts: [bad] }));
  assert.equal(parseSurface({ ...snapshot, contracts: [{ ...midpoint, iv_bid: null }] }).contracts[0].iv_bid, null);
});

test("quality counts must account for every received and solved observation", () => {
  const quality = { received_contracts: 5, excluded_by_filters: 1, rejected_quotes: 1,
    eligible_contracts: 3, invalid_or_unidentifiable_prices: 1, unresolved_iv: 1, solved_contracts: 1 };
  const data = { ...snapshot, contracts: [base], quality };
  assert.equal(parseSurface(data).quality?.unresolved_iv, 1);
  for (const bad of [0, [], {}, { ...quality, unresolved_iv: -1 },
    { ...quality, received_contracts: 6 }, { ...quality, solved_contracts: 2 },
    { ...quality, eligible_contracts: 1.5 }]) {
    assert.throws(() => parseSurface({ ...data, quality: bad }));
  }
});


test("unsupported model counts are separate from solver failures and optional in older responses", () => {
  const quality = { received_contracts: 3, excluded_by_filters: 0, rejected_quotes: 0,
    eligible_contracts: 3, invalid_or_unidentifiable_prices: 0, unsupported_model: 1,
    unresolved_iv: 1, solved_contracts: 1 };
  const data = { ...snapshot, contracts: [base], quality };
  assert.equal(parseSurface(data).quality?.unsupported_model, 1);
  assert.throws(() => parseSurface({ ...data, quality: { ...quality, unsupported_model: -1 } }));
  assert.throws(() => parseSurface({ ...data, quality: { ...quality, unsupported_model: 2 } }));
  const { unsupported_model, ...legacy } = quality;
  assert.equal(parseSurface({ ...data, quality: { ...legacy, unresolved_iv: 2 } }).quality?.unsupported_model, 0);
});
