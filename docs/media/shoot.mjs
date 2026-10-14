// Renders the real command outputs (outputs/*.txt, from scripts/capture_outputs.sh) as Bend-style code blocks on
// paper cards, at 2x, into shots/. Usage: node shoot.mjs [name ...]
import puppeteer from "puppeteer-core";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { blockHTML, esc } from "./term.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const read = (f) => fs.readFileSync(path.join(here, "outputs", f), "utf8").replace(/\n+$/, "");
export const browserPath = process.env.BRAVE || "/usr/bin/brave";

// the 1.7B's generation, condensed: the load, the prompt, the ids it chose, and the decoded text
const gen = () => {
  const ls = read("gen_17b.txt").split("\n");
  const ids = ls.filter((l) => /^\s*id /.test(l)).map((l) => l.match(/id (\d+)/)[1]);
  const t = ls.findIndex((l) => l.startsWith("text:"));
  return [ls[0], ls[1], `generated ids: ${ids.slice(0, 12).join(" ")} ...`, ...ls.slice(t)].join("\n");
};
const bad = (f, keep) => { const ls = read(f).split("\n"); return keep.map((i) => ls[i]).join("\n"); };
const capLine = (f) => read(f).split("\n").findIndex((l) => l.startsWith("Location:"));

// name, command label, text, lines to mark, caption and its class, width
const SHOTS = [
  { name: "bad-weight", cmd: "bend tests/bad_weight.bend", text: () => bad("bad_weight.txt", [0, 1, 2, 3]), mark: [2, 3],
    cap: "the query and key projections of the 135M swapped: 576 × 576 where 192 × 576 goes, rejected", capCls: "r", width: 1080 },
  { name: "bad-cap", cmd: "bend tests/bad_cap.bend", text: () => { const k = capLine("bad_cap.txt"); return bad("bad_cap.txt", [0, 1, 2, 3, k, k + 1, k + 2, k + 3]); }, mark: [2, 3],
    cap: "a third write into a cache made for two positions needs a proof that 2 < 2: it does not compile", capCls: "r", width: 1080 },
  { name: "law-gqa", cmd: "bend llama/main.bend --verdict", text: () => read("law_gqa.txt") + "\n\n" + read("verdict.txt"),
    cap: "with g query heads per key/value head, query head j·g + i reads key/value head j, checked again by the kernel", capCls: "g", width: 1080 },
  { name: "gen", cmd: './gen data/smollm2-1.7b "The capital of France is" 32', text: gen,
    cap: "SmolLM2-1.7B in Bend: the same 32 tokens as transformers", capCls: "g", width: 1080 },
];

export async function shoot(only) {
  const browser = await puppeteer.launch({ executablePath: browserPath, headless: "new", args: ["--no-sandbox", "--allow-file-access-from-files"] });
  try {
    for (const s of SHOTS) {
      if (only && !only.includes(s.name)) continue;
      const page = await browser.newPage();
      await page.setViewport({ width: s.width + 120, height: 900, deviceScaleFactor: 2 });
      const html = `<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="${pathToFileURL(path.join(here, "theme.css"))}">
<body style="background:transparent;padding:30px"><div id="w" class="card" style="width:${s.width}px">${blockHTML({ cmd: s.cmd, text: s.text(), mark: s.mark || [] })}<div class="cap ${s.capCls || ""}">${esc(s.cap)}</div></div>`;
      const tmp = path.join(here, ".tmp-shot.html");
      fs.writeFileSync(tmp, html);
      await page.goto(pathToFileURL(tmp).href);
      await page.evaluate(() => document.fonts.ready);
      await (await page.$("#w")).screenshot({ path: path.join(here, "shots", s.name + ".png"), omitBackground: true });
      await page.close();
      console.log("shot", s.name);
    }
  } finally {
    await browser.close();
    fs.rmSync(path.join(here, ".tmp-shot.html"), { force: true });
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await shoot(process.argv.slice(2).length ? process.argv.slice(2) : null);
