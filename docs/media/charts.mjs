// Builds the README banner as SVG, in Bend's visual language (bend-lang.com), as bend-ml's does:
// warm paper, one monospace family (embedded, so GitHub renders it), violet for Bend, grey for
// the rest, hatched bars for values off the chart.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const out = (name, svg) => fs.writeFileSync(path.join(here, name), svg);
const b64 = (f) => fs.readFileSync(path.join(here, "fonts", f)).toString("base64");
const FONT = `@font-face{font-family:"Bend ML Mono";font-weight:400;src:url(data:font/woff2;base64,${b64("BendMLMono-Regular.woff2")}) format("woff2")}
@font-face{font-family:"Bend ML Mono";font-weight:700;src:url(data:font/woff2;base64,${b64("BendMLMono-Bold.woff2")}) format("woff2")}`;

const C = { bg: "#f2eee7", bg2: "#e9e4db", line: "#e4dfd5", rule: "#d9d3c7", b1: "#a9a59d", b00: "#87847d", b01: "#69665f", b02: "#4d4a44",
  vio: "#8b83b5", grn: "#7e9a5e", red: "#c46a60", ora: "#c4845c", bar: "#cdc7bc" };
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const T = (x, y, s, { size = 15, fill = C.b01, weight = 400, anchor = "start", extra = "" } = {}) =>
  `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" font-weight="${weight}" text-anchor="${anchor}" ${extra}>${esc(s)}</text>`;
const svg = (w, h, title, body) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${esc(title)}">
<title>${esc(title)}</title>
<style>${FONT}
text{font-family:"Bend ML Mono",ui-monospace,Menlo,"SF Mono",Consolas,"Liberation Mono",monospace}</style>
<defs><pattern id="hatch" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)"><rect width="9" height="9" fill="#efebe4"/><rect width="4" height="9" fill="#e2dcd1"/></pattern></defs>
<rect width="${w}" height="${h}" rx="14" fill="${C.bg}"/>
${body}
</svg>
`;
const fmt = (v) => (v < 0.1 ? `${Math.round(v * 1000)} ms` : v >= 10 ? `${v.toFixed(0)} s` : v >= 1 ? `${+v.toFixed(1)} s` : `${+v.toFixed(2)} s`);

// one Bend-style vertical bar chart: bars = [{label, sub, v, kind: "bend"|"other"|"over"}]
function bars(x0, y0, w, h, title, max, items) {
  const n = items.length, gap = 18, bw = (w - gap * (n - 1)) / n;
  let s = T(x0 + w / 2, y0, title, { size: 17, fill: C.b02, weight: 700, anchor: "middle" });
  const base = y0 + 40 + h;
  s += `<line x1="${x0 - 6}" y1="${base}" x2="${x0 + w + 6}" y2="${base}" stroke="${C.rule}" stroke-width="1"/>`;
  items.forEach((it, i) => {
    const x = x0 + i * (bw + gap), over = it.v > max, bh = over ? h : Math.max(2, (it.v / max) * h);
    const fill = over ? "url(#hatch)" : it.kind === "bend" ? C.vio : C.bar;
    s += `<rect x="${x.toFixed(1)}" y="${(base - bh).toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" fill="${fill}"/>`;
    if (over) s += `<path d="M${x} ${base - h + 14} l${bw / 4} -8 l${bw / 4} 8 l${bw / 4} -8 l${bw / 4} 8" fill="none" stroke="${C.bg}" stroke-width="5"/>`;
    const col = it.kind === "bend" ? C.vio : C.b01;
    s += T(x + bw / 2, base - bh - 10, fmt(it.v), { size: 15, fill: col, weight: it.kind === "bend" ? 700 : 400, anchor: "middle" });
    s += T(x + bw / 2, base + 24, it.label, { size: 14, fill: it.kind === "bend" ? C.vio : C.b00, weight: it.kind === "bend" ? 700 : 400, anchor: "middle" });
    if (it.sub) s += T(x + bw / 2, base + 42, it.sub, { size: 13, fill: C.b1, anchor: "middle" });
  });
  return s;
}

{
  const W = 1200, H = 360;
  let cells = "";
  const cs = 22, gp = 6, gx = W / 2 - (3 * cs + 2 * gp) / 2, gy = 52;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) cells += `<rect x="${gx + c * (cs + gp)}" y="${gy + r * (cs + gp)}" width="${cs}" height="${cs}" rx="4" fill="${r === 1 && c === 2 ? C.vio : C.bar}"/>`;
  let b = cells;
  // the wordmark and Bend's violet block cursor, centred together (the font advance is 0.6 em)
  const fs = 92, tw = 10 * 0.6 * fs, cw = 0.5 * fs, gapc = 0.12 * fs, x0 = W / 2 - (tw + gapc + cw) / 2;
  b += T(x0, 214, "llama.bend", { size: fs, fill: C.b02, weight: 700 });
  b += `<rect x="${(x0 + tw + gapc).toFixed(1)}" y="${214 - 0.78 * fs}" width="${cw}" height="${0.86 * fs}" fill="${C.vio}"/>`;
  b += T(W / 2, 270, "Llama models in Bend 2", { size: 26, fill: C.b00, anchor: "middle" });
  b += T(W / 2, 310, "shapes in the types · laws checked by the kernel · the same tokens as PyTorch", { size: 18, fill: C.b1, anchor: "middle" });
  out("banner.svg", svg(W, H, "llama.bend: Llama models in Bend 2. Shapes in the types, laws checked by the kernel, the same tokens as PyTorch.", b));
}
console.log("figures ok");
