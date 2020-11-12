from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)
model = pickle.load(open('naive_bayes.pkl', 'rb'))

@app.route('/test', methods=['GET'])
def test():
    data = request.get_json()
    output = {"test": "DISSERTATION"}
    return jsonify(output)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    return data

app.run()
