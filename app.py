from flask import Flask, request, jsonify
from langchain_google_vertexai import VertexAI
from dotenv import load_dotenv
from vertex_ai_tools import predict_no_context, predict_with_context
import os

load_dotenv()
app = Flask(__name__)

@app.route('/predict-no-context', methods=['POST'])
def no_context():

    data = request.get_json()
    query = data['query']

    prediction = predict_no_context(query)

    return prediction
    
@app.route('/predict-with-context', methods=['POST'])
def with_context():

    data = request.get_json()
    query = data['query']

    prediction = predict_with_context(query)

    return prediction



if __name__ == '__main__':
    app.run(debug=True)