import os
from urllib.parse import quote
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import PGVector
from vertexai.language_models import ChatModel

def predict_no_context(query):
    
    model_name= os.environ.get("VERTEX_AI_PREDICT_MODEL")
    model = VertexAI(model_name=model_name)
    return model.invoke(query)

def predict_with_context(query):
    
    context = _get_context(query)

    model_name = os.environ.get("VERTEX_AI_PREDICT_MODEL")
    chat_model = ChatModel.from_pretrained(model_name)

    chat = chat_model.start_chat(context=context)
    answer = chat.send_message(query)

    print(answer)

    return f"Answer: {answer.text} \n\n Context: \n{context}"

def _get_context(query):

    model = _create_embed_model()
    db_retriever = _create_db_retriever(model)
    documents = db_retriever.invoke(query)

    paragraphs = ["Answer the question only from the context. Otherwise, response with 'No Data':"]

    #TODO: Set threshold for cosine similarity and format context nicely
    # for i in range(len(documents)):
    #     paragraphs.append(documents[i].page_content)

    paragraphs.append(documents[0].page_content)

    context = "\n\n".join(paragraphs).join("Answer the following question.")

    return context

def _create_embed_model():
    project_number = os.environ.get("GOOGLE_PROJECT_NUMBER")
    embed_model = os.environ.get("VERTEXAI_EMBED_MODEL")
    
    return VertexAIEmbeddings(project=project_number, model_name=embed_model)

def _create_db_retriever(embed_model):
    collection=os.environ.get('PG_VECTOR_COLLECTION')
    connection_string = _create_pg_vector_connection_string()

    db = PGVector(embedding_function=embed_model, collection_name=collection, connection_string=connection_string)
    return db.as_retriever()

def _create_pg_vector_connection_string():
    password = os.environ.get('PG_VECTOR_PASSWORD')
    user = os.environ.get('PG_VECTOR_USER')
    host = os.environ.get('PG_VECTOR_HOST')
    port = os.environ.get('PG_VECTOR_PORT')
    db = os.environ.get('PG_VECTOR_DATABASE')

    return f"postgresql+psycopg2://{user}:%s@{host}:{port}/{db}" % quote(password)