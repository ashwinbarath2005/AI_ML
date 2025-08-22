#=======================================================
# chatbot_flask.py — Flask REST API
# Run: python chatbot_flask.py  (http://127.0.0.1:8000)
#======================================================

from flask import Flask, request, jsonify
from inference_utils import load_artifacts, greedy_decode

MODEL_DIR = 'artifacts'

app = Flask(__name__)
model, tok, meta = load_artifacts(MODEL_DIR)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    message = data.get('message', '')
    reply = greedy_decode(model, tok, meta, message)
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
