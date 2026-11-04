// ---------- sliders ----------
const params = ["temperature", "top_k", "top_p", "max_tokens"];
params.forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener("input", () => (document.getElementById(id + "-val").textContent = el.value));
});
document.querySelectorAll(".chip-btn").forEach(b =>
  b.addEventListener("click", () => (document.getElementById("prompt").value = b.textContent)));

// ---------- real-time generation (Server-Sent Events) ----------
let source = null;
const goBtn = document.getElementById("go");

goBtn.addEventListener("click", () => {
  if (source) { source.close(); source = null; goBtn.textContent = "Generate"; return; }

  const prompt = document.getElementById("prompt").value;
  const query = new URLSearchParams({ prompt });
  params.forEach(id => query.set(id, document.getElementById(id).value));

  const out = document.getElementById("output");
  const stats = document.getElementById("stats");
  out.innerHTML = "";
  const promptSpan = document.createElement("span");
  promptSpan.className = "prompt-text";
  promptSpan.textContent = prompt;
  const genSpan = document.createElement("span");
  out.append(promptSpan, genSpan);

  let pieces = 0;
  const start = performance.now();
  goBtn.textContent = "Stop";

  source = new EventSource("/api/generate?" + query.toString());
  source.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.text !== undefined) {
      genSpan.textContent += msg.text;
      pieces++;
      const secs = (performance.now() - start) / 1000;
      stats.textContent = `${pieces} tokens · ${(pieces / secs).toFixed(1)} tokens/s`;
    }
    if (msg.done || msg.error) {
      if (msg.error) stats.textContent = "Error: " + msg.error;
      source.close(); source = null; goBtn.textContent = "Generate";
    }
  };
  source.onerror = () => { if (source) source.close(); source = null; goBtn.textContent = "Generate"; };
});

// ---------- model info ----------
fetch("/api/model-info").then(r => r.json()).then(info => {
  if (info.error) return;
  const box = document.getElementById("model-info");
  const items = {
    Layers: info.layers, Heads: info.heads, "d_model": info.d_model,
    Context: info.context + " tokens", Vocab: info.vocab,
    Params: info.params ? (info.params / 1e6).toFixed(2) + "M" : "—",
  };
  Object.entries(items).forEach(([k, v]) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.textContent = `${k}: ${v}`;
    box.append(c);
  });
});

// ---------- live training charts (Chart.js) ----------
const css = getComputedStyle(document.documentElement);
const C1 = css.getPropertyValue("--series-1").trim();
const C2 = css.getPropertyValue("--series-2").trim();
Chart.defaults.color = css.getPropertyValue("--muted").trim();
Chart.defaults.borderColor = css.getPropertyValue("--grid").trim();

function makeChart(id, title, series, opts = {}) {
  return new Chart(document.getElementById(id), {
    type: "line",
    data: {
      labels: [],
      datasets: series.map(([label, color]) => ({
        label, data: [], borderColor: color, backgroundColor: color,
        borderWidth: 2, pointRadius: 2, tension: 0.25,
      })),
    },
    options: {
      responsive: true, animation: { duration: 300 },
      plugins: { title: { display: true, text: title } },
      scales: { x: { title: { display: true, text: "epoch" } }, y: opts.y || {} },
    },
  });
}

const charts = {
  loss: makeChart("lossChart", "Cross-entropy loss", [["train", C1], ["validation", C2]]),
  ppl: makeChart("pplChart", "Perplexity (lower is better)", [["train", C1], ["validation", C2]],
                 { y: { type: "logarithmic" } }),
  acc: makeChart("accChart", "Next-token accuracy", [["train", C1], ["validation", C2]]),
  lr: makeChart("lrChart", "Learning rate", [["lr", C1]]),
};

function setData(chart, labels, ...arrays) {
  chart.data.labels = labels;
  arrays.forEach((a, i) => (chart.data.datasets[i].data = a));
  chart.update();
}

let lastLen = -1;
async function refreshHistory() {
  const h = await fetch("/api/history").then(r => r.json()).catch(() => null);
  const badge = document.getElementById("train-status");
  if (!h || h.status === "missing") { badge.textContent = "no history yet"; return; }
  badge.textContent = h.status === "training" ? "training…" : "complete";
  badge.className = "badge " + h.status;
  if (h.loss.length === lastLen) return;     // nothing new
  lastLen = h.loss.length;
  const epochs = h.loss.map((_, i) => i + 1);
  setData(charts.loss, epochs, h.loss, h.val_loss);
  setData(charts.ppl, epochs, h.perplexity, h.val_perplexity);
  setData(charts.acc, epochs, h.accuracy, h.val_accuracy);
  setData(charts.lr, epochs, h.lr);
}
refreshHistory();
setInterval(refreshHistory, 5000);   // poll every 5 s -> charts grow while training runs
