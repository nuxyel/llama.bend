// The README deck: llama.bend at the commit measured for BENCHMARK.md, next to transformers, llama.cpp and candle, and
// the same commit built with Bend 2.0.36 and 2.0.37. Every number comes from numbers.json (exported from the run's
// summary) or outputs/ (real runs).
//   node docs/media/deck.mjs docs/media/deck/readme.mjs docs/media/deck/frames
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const title = "llama.bend · benchmark";
const here = path.dirname(fileURLToPath(import.meta.url));
const N = JSON.parse(fs.readFileSync(path.join(here, "..", "numbers.json"), "utf8"));
const out = (f) => fs.readFileSync(path.join(here, "..", "outputs", f), "utf8").trim();
const LAWS = out("law_count.txt");
const FOOT = `${N.machine} · measured on ${N.date}, commit ${N.commit}`;
const MODELS = [["smollm2-135m", "SmolLM2-135M"], ["smollm2-360m", "SmolLM2-360M"], ["smollm2-1.7b", "SmolLM2-1.7B"]];
const B37 = "llama.bend (Bend 2.0.37)", B36 = "llama.bend (Bend 2.0.36)";

export default ({ frame, hbars, stats, table, C }) => {
  const ms = (v) => (v >= 100 ? `${Math.round(v)} ms` : `${v.toFixed(1)} ms`);
  const x = (a, b) => `${(a / b).toFixed(1)}×`;
  const pct = (a, b) => { const d = (b / a - 1) * 100; return `${d > 0 ? "+" : ""}${d.toFixed(0)}%`; };
  const P = N.per_token, BIG = P["smollm2-1.7b"];
  const slides = [];

  slides.push({ name: "title", html: frame({
    kicker: "llama.bend v1.0",
    title: `Llama models in Bend 2, <span class="v">measured</span>`,
    body: `<div style="height:40px"></div>` + stats([
      { big: LAWS, label: "laws proved", sub: "checked again by the Lean kernel" },
      { big: "3", label: "models, one package", sub: "SmolLM2 135M · 360M · 1.7B" },
      { big: "0", label: "different tokens", sub: "against transformers, 12 prompts per model" },
    ], 110) + `<div class="p" style="text-align:center;margin-top:60px">The same greedy tokens as PyTorch, next to the systems people use for these models.</div>`,
    foot: FOOT,
  }) });

  const row = (t) => [
    { label: "llama.cpp", sub: "F32", v: BIG[t]["llama.cpp"] },
    { label: "transformers", sub: "PyTorch 2.14", v: BIG[t].transformers },
    { label: "candle", sub: "F32", v: BIG[t].candle },
    { label: "llama.bend", sub: "Bend 2.0.37", v: BIG[t][B37], kind: "bend" },
  ];
  const max1 = Math.max(...row("1").map((r) => r.v)) * 1.05, max16 = Math.max(...row("16").map((r) => r.v)) * 1.05;
  slides.push({ name: "vs-references", html: frame({
    kicker: `SmolLM2-1.7B · ms per generated token over ${N.tokens["smollm2-1.7b"]} tokens · lower is better`,
    title: `where Bend stands`,
    body: `<div style="display:flex;gap:60px">` + hbars({ w: 820, rowH: 70, labelW: 300, max: max1, fmt: ms, title: "1 thread", items: row("1") })
      + hbars({ w: 820, rowH: 70, labelW: 300, max: max16, fmt: ms, title: "16 threads", items: row("16") }) + `</div>`,
    takeaway: `llama.cpp is ${x(BIG["1"][B37], BIG["1"]["llama.cpp"])} ahead on one thread and ${x(BIG["16"][B37], BIG["16"]["llama.cpp"])} on 16`,
    foot: `same tokens everywhere · llama.bend's own clock at 16 threads: ${Math.round(N.own_clock["smollm2-1.7b"]["16"][B37])} ms · ` + FOOT,
  }) });

  const cell = (v, bend) => ({ t: ms(v), color: bend ? C.vio : C.b01, bold: bend });
  slides.push({ name: "three-models", html: frame({
    kicker: "ms per generated token · 1 thread / 16 threads",
    title: `one package, <span class="v">three sizes</span>`,
    body: table(["model", "llama.bend", "transformers", "llama.cpp", "candle"], MODELS.map(([m, label]) => [
      label, ...[B37, "transformers", "llama.cpp", "candle"].map((s) => ({ ...cell(P[m]["1"][s], s === B37), t: `${ms(P[m]["1"][s])} / ${ms(P[m]["16"][s])}` }))]),
      { size: 26, widths: [300, 330, 330, 330, 330] }),
    takeaway: `only the config changes between the three`,
    foot: `${N.tokens["smollm2-135m"]} tokens for the 135M and 360M, ${N.tokens["smollm2-1.7b"]} for the 1.7B · ` + FOOT,
  }) });

  slides.push({ name: "bend-release", html: frame({
    kicker: "the same commit, built with two Bend releases · ms per generated token",
    title: `Bend 2.0.36 → <span class="v">2.0.37</span>`,
    body: table(["model", "threads", "Bend 2.0.36", "Bend 2.0.37", "change"], MODELS.flatMap(([m, label]) => ["1", "16"].map((t) => [
      t === "1" ? label : "", t, ms(P[m][t][B36]), { t: ms(P[m][t][B37]), color: C.vio, bold: true }, pct(P[m][t][B36], P[m][t][B37])])),
      { size: 25, widths: [300, 160, 260, 260, 200] }),
    takeaway: `no change beyond the spread of the samples`,
    foot: `1.7B at 16 threads by its own clock: ${Math.round(N.own_clock["smollm2-1.7b"]["16"][B36])} and ${Math.round(N.own_clock["smollm2-1.7b"]["16"][B37])} ms; loading inflates the totals · ` + FOOT,
  }) });

  const G = N.growth;
  slides.push({ name: "flat-cost", html: frame({
    kicker: "512 generated tokens · llama.bend's own per-token timestamps",
    title: `a token costs <span class="g">the same</span> at position 500`,
    body: table(["model", "threads", "first tokens", "last tokens", "growth per position"], MODELS.flatMap(([m, label]) => ["1", "16"].map((t) => [
      t === "1" ? label : "", t, ms(G[m][t].first), ms(G[m][t].last), { t: `${G[m][t].ms_per_position.toFixed(2)} ms`, color: C.grn, bold: true }])),
      { size: 25, widths: [300, 160, 260, 260, 300] }),
    takeaway: `the key/value cache lives in flat arrays`,
    foot: FOOT,
  }) });

  const S = N.startup["smollm2-1.7b"];
  const BR = "llama.bend (binary-read, 2026-10-16)";
  const srow = (k, f) => [
    { label: "llama.cpp", v: f(S["llama.cpp"]) }, { label: "candle", v: f(S.candle) },
    { label: "transformers", v: f(S.transformers) }, { label: "llama.bend, before", v: f(S[B37]), kind: "bend" },
    { label: "llama.bend, now", v: f(S[BR]), kind: "bend" }];
  slides.push({ name: "startup", html: frame({
    kicker: "SmolLM2-1.7B · load and generate one token · 16 threads",
    title: `loading: <span class="o">66 s → 9 s</span>`,
    body: `<div style="display:flex;gap:60px">` +
      hbars({ w: 820, rowH: 66, labelW: 280, max: S[B37].s * 1.05, fmt: (v) => `${v} s`, title: "wall time", items: srow("s", (r) => r.s) }) +
      hbars({ w: 820, rowH: 66, labelW: 280, max: Math.max(...Object.values(S).map((r) => r.mb)) * 1.05, fmt: (v) => `${(v / 1000).toFixed(1)} GB`, title: "peak memory", items: srow("mb", (r) => r.mb) }) + `</div>`,
    takeaway: `the weights now go straight into their arrays (bend-ml-binary-read)`,
    foot: `${N.machine} · measured on ${N.date}, commit ${N.commit}; "now" on 2026-10-16, commit 8785791`,
  }) });

  return slides;
};
