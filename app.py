from flask import Flask, request, jsonify
from langchain_google_vertexai import VertexAI
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)

@app.route('/ask', methods=['GET'])
def ask_question():

    model = VertexAI(model_name="gemini-pro")
    res = model.invoke("Hi, I would like to become student of Algebra University College in Zagreb. I am from Sweden. How can I apply?")

    return res
    
if __name__ == '__main__':
    app.run(debug=True)