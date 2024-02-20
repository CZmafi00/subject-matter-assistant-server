from langchain_google_vertexai import VertexAI

def predict_no_context(query):
    
    model = VertexAI(model_name="gemini-pro")
    return model.invoke(query)