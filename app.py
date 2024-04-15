from flask import Flask, request, jsonify
from langchain_google_vertexai import VertexAI
from dotenv import load_dotenv
from vertex_ai_tools import predict_with_text_model, predict_with_chat_model, design_prompt
import os

load_dotenv()
app = Flask(__name__)

@app.route('/predict-no-context', methods=['POST'])
def no_context():

    data = request.get_json()
    query = data['query']
    prompt_template= os.environ.get("PROMPT_NO_CONTEXT_INPUT_TASK")

    prediction = predict_with_text_model(f"{prompt_template}{query}")

    return prediction
    
@app.route('/predict-with-context', methods=['POST'])
def with_context():

    data = request.get_json()
    query = data['query']

    prediction = predict_with_chat_model(query)

    return prediction

@app.route('/ask-student-office', methods=['POST'])
def ask_student_office():

    data = request.get_json()
    query = data['query']

    prompt = design_prompt(query)
    result = predict_with_text_model(prompt)

    return result

if __name__ == '__main__':
    app.run(debug=True)