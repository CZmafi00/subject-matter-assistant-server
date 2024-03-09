# Subject Matter Assistant Server

This repository contains the code for the Subject Matter Assistant Server, which is a part of a graduation thesis for [Algebra University College](https://www.algebra.hr/). The solution is implemented in [Flask](https://flask.palletsprojects.com/en/3.0.x/) and utilizes the [LangChain](https://github.com/langchain-ai/langchain) framework along with [Vertex AI](https://cloud.google.com/vertex-ai) models. The application retrieves context from a [PGVector](https://github.com/pgvector/pgvector) database and uses it as context for Vertex AI Language Model (LLM) prediction.

## Overview

- **Framework:** Flask
- **LangChain Framework:** Included
- **Vertex AI Models:** Utilized for language modeling
- **PGVector:** Used for storing and retrieving context data
- **Dockerfile:** Included for containerization
- **Google Service Account Key:** Required for authentication

## Getting Started

1. **Clone the Repository:**
```bash
git clone git@github.com:CZmafi00/subject-matter-assistant-server.git
# alternatively use https
# git clone https://github.com/CZmafi00/subject-matter-assistant-server.git
cd subject-matter-assistant-server
```

2. **Install Dependencies:**
```
# Use a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

3. **Set Up Google Service Account:**

Obtain a Google Service Account Key and save it as service-account-key.json in the project root.

4. **Build Docker Image (Optional):**
```
docker build -t subject-matter-assistant-server .

```

5. **Run the Application:**
```
# Using python
python app.py
# Using Docker
docker run -p 5000:5000 -e GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json -e <list all environment variables> subject-matter-assistant-server
# using Docker compose
docker compose up -d
```

## Configuration

### Environment Variables

```markdown
| Variable                           | Description                                      |
|------------------------------------|--------------------------------------------------|
| `GOOGLE_APPLICATION_CREDENTIALS`    | Path to the Google Service Account Key JSON file |
| `GOOGLE_PROJECT_NUMBER`             | Google Cloud Project Number                     |
| `VERTEX_AI_TEXT_MODEL`              | Vertex AI Text Model ID                         |
| `VERTEX_AI_TEXT_MAX_OUTPUT_TOKENS`  | Maximum number of output tokens for Text Model   |
| `VERTEX_AI_TEXT_MODEL_TEMPERATURE`  | Temperature parameter for Text Model            |
| `VERTEX_AI_TEXT_MODEL_TOP_K`        | Top-K parameter for Text Model                   |
| `VERTEX_AI_TEXT_MODEL_TOP_P`        | Top-P parameter for Text Model                   |
| `VERTEX_AI_CHAT_MODEL`              | Vertex AI Chat Model ID                         |
| `VERTEX_AI_CHAT_MODEL_TEMPERATURE`  | Temperature parameter for Chat Model            |
| `VERTEX_AI_CHAT_MODEL_TOP_K`        | Top-K parameter for Chat Model                   |
| `VERTEX_AI_CHAT_MODEL_TOP_P`        | Top-P parameter for Chat Model                   |
| `VERTEXAI_EMBED_MODEL`              | Vertex AI Embed Model ID                        |
| `VERTEXAI_EMBED_MODEL_RELEVANCE_TRESHOLD` | Relevance threshold for Embed Model         |
| `PROMPT_NO_CONTEXT_INPUT_TASK`      | Input task for no context prompts               |
| `PROMPT_CONTEXT_INPUT_TASK`         | Input task for context prompts                  |
| `PG_VECTOR_HOST`                    | PGVector database host                          |
| `PG_VECTOR_PORT`                    | PGVector database port                          |
| `PG_VECTOR_DATABASE`                | PGVector database name                          |
| `PG_VECTOR_COLLECTION`              | PGVector collection name                        |
| `PG_VECTOR_USER`                    | PGVector database user                          |
| `PG_VECTOR_PASSWORD`                | PGVector database password                      |
