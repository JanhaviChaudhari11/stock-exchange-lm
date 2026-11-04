"""
Flask web app:
  GET /                      -> the dashboard page
  GET /api/generate?prompt=  -> Server-Sent Events stream, one event per generated piece of text
  GET /api/history           -> training_history.json (polled by Chart.js for live charts)
  GET /api/model-info        -> architecture details

Run:  python app.py   then open http://localhost:5000
"""
import json
import os
import threading

from flask import Flask, Response, jsonify, render_template, request

from generate import load_generator
from model import Config

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "training_history.json")

app = Flask(__name__)
_generator = None
_lock = threading.Lock()   # one generation at a time — TensorFlow model isn't shared across threads


def get_generator():
    global _generator
    if _generator is None:
        _generator = load_generator(CHECKPOINT_DIR)
    return _generator


def clamp(value, lo, hi, cast):
    try:
        return max(lo, min(hi, cast(value)))
    except (TypeError, ValueError):
        return lo


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate")
def api_generate():
    prompt = request.args.get("prompt", "The stock market")[:500]
    temperature = clamp(request.args.get("temperature", 0.8), 0.1, 2.0, float)
    top_k = clamp(request.args.get("top_k", 40), 0, 200, int)
    top_p = clamp(request.args.get("top_p", 0.95), 0.1, 1.0, float)
    max_tokens = clamp(request.args.get("max_tokens", 120), 1, 400, int)

    def events():
        try:
            gen = get_generator()
            with _lock:
                for piece in gen.stream(prompt, max_tokens, temperature, top_k, top_p):
                    yield f"data: {json.dumps({'text': piece})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    # SSE: the browser keeps the connection open and receives each token as soon as it's made.
    return Response(events(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/history")
def api_history():
    if not os.path.exists(HISTORY_PATH):
        return jsonify({"status": "missing"})
    with open(HISTORY_PATH) as f:
        return jsonify(json.load(f))


@app.route("/api/model-info")
def api_model_info():
    cfg_path = os.path.join(CHECKPOINT_DIR, "config.json")
    if not os.path.exists(cfg_path):
        return jsonify({"error": "No trained model yet — run train.py"}), 404
    cfg = Config.load(cfg_path)
    info = {"layers": cfg.n_layers, "heads": cfg.n_heads, "d_model": cfg.d_model,
            "d_ff": cfg.d_ff, "context": cfg.seq_len, "vocab": cfg.vocab_size}
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            info["params"] = json.load(f).get("params")
    return jsonify(info)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
