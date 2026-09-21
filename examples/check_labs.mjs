// Run with: node examples/check_labs.mjs
// Exercises the inline lab scripts with a minimal DOM and known numeric cases.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

class Element {
  constructor(value = "") {
    this.value = value;
    this.checked = true;
    this.textContent = "";
    this.style = {};
    this.children = [];
    this.listeners = {};
    this.className = "";
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  append(...children) { this.children.push(...children); }
  replaceChildren() { this.children = []; }
  getContext() {
    const noop = () => {};
    return { clearRect: noop, beginPath: noop, arc: noop, moveTo: noop,
      lineTo: noop, stroke: noop, fill: noop, fillText: noop, setLineDash: noop };
  }
}

function load(name, values) {
  const html = readFileSync(new URL(`../docs/assets/lab_${name}.html`, import.meta.url), "utf8");
  const script = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
  assert.ok(script, `${name} has inline script`);
  const nodes = new Map(Object.entries(values).map(([id, value]) => [id, new Element(value)]));
  const document = {
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); },
    createElement() { return new Element(); },
  };
  runInNewContext(script, { document, Math, Number, String });
  return id => document.getElementById(id);
}

const probability = load("probability", { a: "2", b: "1", c: "0", tau: "1", target: "0" });
assert.match(probability("readout").textContent, /loss = 0\.408/);
const firstBar = probability("bars").children[0].children[2].textContent;
probability("target").value = "2";
probability("target").listeners.input();
assert.match(probability("readout").textContent, /loss = 2\.408/);
assert.equal(probability("bars").children[0].children[2].textContent, firstBar);

const attention = load("attention", { row: "1", qx: "0", qy: "1", tau: "1", causal: "" });
assert.match(attention("readout").textContent, /weighted value output: \[0\.395, 0\.605\]/);
attention("row").value = "0";
attention("row").listeners.input();
assert.equal(attention("matrix").children[0].children[2].textContent, "masked");

const position = load("position", { omega: "0.1", pos: "8", delta: "4" });
assert.match(position("dot").textContent, /= 0\.921/);
const before = position("dot").textContent;
position("pos").value = "9";
position("pos").listeners.input();
assert.equal(position("dot").textContent, before);

const lstm = load("lstm", { prev: "0.5", forget: "0.8", input: "0.6", candidate: "0.5", output: "0.5" });
assert.match(lstm("readout").textContent, /0\.400 \+ 0\.300 = 0\.700/);
lstm("forget").value = "1";
lstm("input").value = "0";
lstm("forget").listeners.input();
assert.match(lstm("readout").textContent, /= 0\.500/);

const bpe = load("bpe", {});
assert.match(bpe("counts").textContent, /\(l,o\): 3/);
bpe("next").listeners.click();
assert.match(bpe("pieces").textContent, /lo \| w/);
bpe("next").listeners.click();
assert.match(bpe("pieces").textContent, /low \| <\/w>/);
bpe("reset").listeners.click();
assert.match(bpe("step").textContent, /step 0 of 2/);

console.log("Five labs passed initial-value and interaction checks.");
