import os
from urllib.parse import quote
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import PGVector
from vertexai.language_models import ChatModel

def predict_with_text_model(query):
    
    model_name= os.environ.get("VERTEX_AI_TEXT_MODEL")
    max_output_tokens = int(os.environ.get("VERTEX_AI_TEXT_MAX_OUTPUT_TOKENS"))
    temperature = float(os.environ.get("VERTEX_AI_TEXT_MODEL_TEMPERATURE"))
    top_k = int(os.environ.get("VERTEX_AI_TEXT_MODEL_TOP_K"))
    top_p = float(os.environ.get("VERTEX_AI_TEXT_MODEL_TOP_P"))

    model = VertexAI(model_name=model_name, max_output_tokens=max_output_tokens, top_k=top_k, top_p=top_p, temperature=temperature)

    prediction = model.invoke(query)

    return {"context": query, "answer": prediction}

def predict_with_chat_model(query):
    
    context = _prepare_context_for_multiple_documents(query)

    model_name = os.environ.get("VERTEX_AI_CHAT_MODEL")
    chat_model = ChatModel.from_pretrained(model_name)

    temperature = float(os.environ.get("VERTEX_AI_CHAT_MODEL_TEMPERATURE"))
    top_k = int(os.environ.get("VERTEX_AI_CHAT_MODEL_TOP_K"))
    top_p = float(os.environ.get("VERTEX_AI_CHAT_MODEL_TOP_P"))

    chat = chat_model.start_chat(context=context[0], temperature=temperature, top_k=top_k, top_p=top_p)
    prediction = chat.send_message(query)

    return {"context": [context[1]], "answer": prediction.text}

def design_prompt(query):

    pg_vector_db = _create_pg_vector()
    # https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.pgvector.PGVector.html#langchain_community.vectorstores.pgvector.PGVector.similarity_search_with_relevance_scores
    documents = pg_vector_db.similarity_search_with_relevance_scores(query)

    documents = _filter_out_relevant_documents(documents)
    context = ""

    if (len(documents) > 0):
        context = _prepare_context_for_multiple_documents(documents)
    else:
        context = _prepare_context_for_no_relevant_documents(query)

    return f"{context}{query}"

def _prepare_context_for_no_relevant_documents(query):
     
    prompt_task = os.environ.get('PROMPT_NO_CONTEXT_INPUT_TASK')

    return f"{prompt_task} Upit: {query}"

def _prepare_context_for_multiple_documents(documents):

    prompt_task = os.environ.get('PROMPT_CONTEXT_INPUT_TASK')

    relevant_docs = _filter_out_relevant_documents(documents)

    context = ""

    for doc in relevant_docs:
        context += "\n\n"
        context += doc[0].page_content

    context += f"\n\n{prompt_task}"

    return context

def _filter_out_relevant_documents(documents):

    similarity_treshold = float(os.environ.get('VERTEXAI_EMBED_MODEL_RELEVANCE_TRESHOLD'))

    sorted_documents = sorted(documents, key=lambda x: x[1], reverse=True)
    filtered_documents = []
    token_sum = 0

    for doc, score in sorted_documents:
        if token_sum + len(doc.page_content) <= 8000 and score >= similarity_treshold:
            filtered_documents.append((doc, score))
        
            token_sum += len(doc.page_content)
            
    return filtered_documents

def _create_embed_model():
    project_number = os.environ.get("GOOGLE_PROJECT_NUMBER")
    embed_model = os.environ.get("VERTEXAI_EMBED_MODEL")
    
    return VertexAIEmbeddings(project=project_number, model_name=embed_model)

def _create_pg_vector():

    collection=os.environ.get('PG_VECTOR_COLLECTION')
    connection_string = _create_pg_vector_connection_string()
    
    embed_model = _create_embed_model()
    db = PGVector(embedding_function=embed_model, collection_name=collection, connection_string=connection_string)

    return db

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