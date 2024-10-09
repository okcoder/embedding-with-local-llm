from flask import Flask,jsonify,request
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

models = {
    "sup-simcse-ja-large": SentenceTransformer("cl-nagoya/sup-simcse-ja-large"),
    "sup-simcse-ja-base": SentenceTransformer("cl-nagoya/sup-simcse-ja-base"),
    "unsup-simcse-ja-large": SentenceTransformer("cl-nagoya/unsup-simcse-ja-large"),
    "unsup-simcse-ja-base": SentenceTransformer("cl-nagoya/unsup-simcse-ja-base")
}

@app.route("/embedding",methods=["POST"]) 
def embedding():
    req=request.json
    text = req['input']
    model= models[req['model']]
    embedding =  model.encode(text).tolist()
    return jsonify({
        "object": "list",
        "data": [
            {
            "object": "embedding",
            "index": 0,
            "embedding": embedding,
            }
        ],
        "model": req['model'],
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0
        }
        })
