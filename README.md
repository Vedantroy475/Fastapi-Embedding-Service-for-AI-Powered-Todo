
# 🧠 AI Embedding Microservice
### FastAPI • Hugging Face • pgvector • CockroachDB • Docker

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Hugging Face](https://img.shields.io/badge/Model-Hugging%20Face-yellow?logo=huggingface)
![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Deployment-Google%20Cloud%20Run-4285F4?logo=googlecloud)

This repository contains a production-grade **vector embedding microservice** built for high-performance RAG (Retrieval-Augmented Generation) workflows.

Unlike wrappers around paid APIs, this service runs a **local, Open Source LLM** purely in-memory to generate dense vector embeddings, demonstrating complete control over the AI pipeline, data privacy, and latency.

> **🔗 Integration Context:**
> This service acts as the computational brain for the main full-stack application.
> View the **React/Node.js Monorepo** here: [**AI-Powered Todo App (Frontend + BFF)**](https://github.com/Vedantroy475/To-Do-App-AI-Powered-Edition.git)

---

# 🔬 Technical Architecture & AI Stack

This service leverages a specific set of high-performance Python libraries to bridge the gap between Natural Language Processing (NLP) and Relational Data.

### 1. The AI Model: Open Source & Local
We utilize **`sentence-transformers/all-MiniLM-L6-v2`**, hosted on **Hugging Face**.
* **Architecture:** A distilled BERT model (Siamese BERT-networks) optimized for semantic similarity.
* **Performance:** It maps sentences to a **384-dimensional dense vector space**.
* **Why this choice?** It offers the best trade-off between speed (latency < 20ms on CPU) and accuracy for semantic search tasks, eliminating the need for external API calls (e.g., OpenAI) for embeddings.

### 2. FastAPI (The Interface)
* Built on the **ASGI** standard for asynchronous concurrency.
* Uses **Pydantic** for strict data validation and type enforcing on `userId`, `todoId`, and payload structures.
* Auto-generates OpenAPI (Swagger) documentation.

### 3. Vector Storage Strategy (pgvector + psycopg2)
Instead of a separate vector database (like Pinecone/Milvus), we use **CockroachDB with pgvector**.
* **Library:** `psycopg2-binary` with `pgvector.psycopg2`.
* **Integration:** We use `register_vector(conn)` to allow Python `numpy` arrays to be inserted directly into SQL `VECTOR` columns without manual serialization.
* **Math:** Retrieval utilizes the **Cosine Distance operator (`<=>`)**, which effectively calculates `1 - CosineSimilarity`, allowing for efficient nearest-neighbor search.

---

# 🚀 Features

- 🔥 **Pure Python AI:** Runs Hugging Face models entirely within the container.
- ⚡ **Low Latency:** No external HTTP calls required for embedding generation.
- 🗄️ **Hybrid Storage:** Combines relational metadata (`userId`) with vector data in a single SQL transaction.
- ♻️ **Idempotent UPSERTs:** Atomic updates handle data consistency automatically.
- 🛡️ **Production Security:** API Key authentication and Google Secret Manager integration.
- 🐳 **Containerized:** Fully Dockerized for stateless deployment on Serverless platforms (Cloud Run).

---

# 📁 Repository Structure

```text
.
├── embedding_service_bin.py        # 🐍 Core Logic (Model Loading + Vector Math + API)
├── requirements.txt                # 📦 Dependencies (torch, transformers, fastapi, etc.)
├── start.sh                        # 🚀 Boot script (SSL Cert injection + Uvicorn start)
├── Dockerfile                      # 🐳 Multi-stage build instructions
├── .env.example                    # ⚙️ Config template
└── README.md                       # 📘 Technical Documentation
````

-----

# 🛠️ Local Development Guide

Follow these steps to run the AI service locally with hot-reloading.

### 1\. Prerequisites

  * Python 3.9+
  * Pip
  * A CockroachDB Serverless Cluster URL

### 2\. Clone the Repository

```bash
git clone [https://github.com/Vedantroy475/Fastapi-Embedding-Service-for-AI-Powered-Todo.git](https://github.com/Vedantroy475/Fastapi-Embedding-Service-for-AI-Powered-Todo.git)
cd Fastapi-Embedding-Service-for-AI-Powered-Todo
```

### 3\. Create a Virtual Environment

Isolate your extensive AI dependencies (PyTorch, Transformers) from your system Python.

**Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4\. Install AI & Server Dependencies

```bash
pip install -r requirements.txt
```

*This will install standard libraries plus `torch`, `transformers`, and `sentence-transformers`.*

### 5\. Configure Environment

Create a `.env` file:

```bash
cp .env.example .env
```

Update values in `.env`:

```env
EMBED_API_KEY="my_secret_key"
DATABASE_URL="postgresql://user:pass@hostname:26257/defaultdb?sslmode=require"
# You can swap this for any Hugging Face model ID
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

### 6\. Run the Server

Start the Uvicorn ASGI server. The `--reload` flag ensures the model doesn't need to reload from disk on every code change, but strictly on restart.

```bash
uvicorn embedding_service_bin:app --reload --host 0.0.0.0 --port 8000
```

Server is now running at `http://0.0.0.0:8000`.

-----

# 🧩 System Context (DFD Level-0)

The diagram below illustrates the high-level data flow. 
- **Repo A (Node.js)** acts as the client, sending raw text and queries.
- **Repo B (This Service)** acts as the processor, performing vector math using PyTorch/SentenceTransformers.
- **CockroachDB** acts as the state layer, storing vectors and performing the final similarity calculations.

![DFD Level- 0](<docs/DFD Level- 0.png>)

-----

# 🧩 DFD Level-1 Diagram (Internal Process View)

While Level-0 shows the system as a single "black box," DFD Level-1 explodes the Python Microservice to reveal its internal processing pipeline. It illustrates how data flows through the API layer, the computational AI model layer, and the database adapter layer.

### ▶️ DFD Level-1

![DFD Level- 1](<docs/DFD Level- 1.png>)

### Process Breakdown

The single circle from Level-0 is broken down into three distinct internal processes:

1.  **Process 1.0: API Request Handler (FastAPI/Pydantic)**

      * **Role:** The gateway. It handles authentication (validating the `x-api-key` header) and uses Pydantic models (`EmbedRequest`, `SearchRequest`) to validate incoming JSON payloads.
      * **Flow:** Invalid requests are immediately rejected (4xx error flow back to Node.js). Valid requests are routed to the next appropriate process.

2.  **Process 2.0: Text Vectorization Engine (SentenceTransformers/PyTorch)**

      * **Role:** The computational core. This process holds the Hugging Face model in memory. It accepts raw string text and converts it into a 384-dimensional dense numerical vector.
      * **Flow:** Used during both the "Write" path (embedding new todos) and the "Read" path (embedding search queries on-the-fly).

3.  **Process 3.0: SQL Execution Adapter (psycopg2)**

      * **Role:** The data access layer. It receives vectors from Process 2.0 and metadata from Process 1.0, constructs the necessary SQL queries, and executes them against CockroachDB.
      * **Flow (Write):** Executes idempotent `INSERT ... ON CONFLICT DO UPDATE` queries.
      * **Flow (Read):** Executes complex `SELECT` queries utilizing the `pgvector` cosine distance operator (`<=>`) for semantic search. It also handles formatting raw database rows back into clean Python dictionaries for the API response.

-----
# 📘 Code Documentation (`embedding_service_bin.py`)

### 1\. Model Instantiation

```python
# This happens once at startup (Cold Start)
model = SentenceTransformer(MODEL_NAME)
EMBED_DIM = model.get_sentence_embedding_dimension() # Auto-detects 384
```

We load the model into global memory space. This allows subsequent API requests to run inference immediately without disk I/O overhead.

### 2\. Vector Math & SQL

We interact with the database using raw SQL for maximum control over the vector operators.

**The Search Query:**

```sql
SELECT text, (embedding <=> %s) AS distance
FROM embeddings
ORDER BY distance ASC
LIMIT k;
```

  * `%s`: The query vector generated by Python.
  * `<=>`: The native pgvector Cosine Distance operator.
  * **Result:** We convert `distance` back to a `similarity_score` (0 to 1) before returning JSON to the frontend.

-----

# 🔌 API Reference

All requests must include: `x-api-key: <YOUR_KEY>`

### ▶️ `POST /embed`

**Description:** Vectorizes text and performs an UPSERT (Update if exists, Insert if new).
**Payload:**

```json
{ "userId": "u1", "todoId": "t1", "text": "Learn Neural Networks" }
```

### 🔍 `POST /search`

**Description:** Embeds the query string and finds nearest semantic neighbors.
**Payload:**

```json
{ "userId": "u1", "query": "AI study topics", "k": 3 }
```

**Response:**

```json
{
  "results": [
    { "todoId": "t1", "text": "Learn Neural Networks", "score": 0.89 }
  ]
}
```

-----

# 🐳 Docker & Cloud Deployment

### Build Image

```bash
docker build -t embedding-service .
```

### Run Container Locally

```bash
docker run -p 8000:8000 \
  -e DATABASE_URL="<YOUR_DB_URL>" \
  -e EMBED_API_KEY="secret" \
  embedding-service
```

### Deploy to Google Cloud Run

1.  **Build & Push:**
    ```bash
    gcloud builds submit --tag us-central1-docker.pkg.dev/<PROJECT>/<REPO>/embedding-service
    ```
2.  **Deploy:**
    ```bash
    gcloud run deploy embedding-service ...
    ```
3.  **Security:**
    The `start.sh` script automatically fetches the CockroachDB CA Certificate from **Google Secret Manager** and mounts it to `/root/.postgresql/root.crt` before the app starts, ensuring End-to-End SSL encryption.

-----

# 📦 Core Dependencies

| Library | Purpose |
| :--- | :--- |
| **`fastapi`** | High-performance web framework for building APIs with Python 3.7+ types. |
| **`uvicorn`** | ASGI server implementation, using `uvloop` for high concurrency. |
| **`sentence-transformers`** | Framework for state-of-the-art text and image embeddings. |
| **`torch`** | PyTorch deep learning library (backend for the transformer model). |
| **`psycopg2-binary`** | PostgreSQL database adapter for Python. |
| **`pgvector`** | Python support for pgvector extension (handling vector serialization). |

-----

# 🙌 Contact

If you are interested in discussing AI Engineering, RAG Architectures, or Full Stack Development, feel free to reach out.

  - **Email:** [vedantroy3@gmail.com](mailto:vedantroy3@gmail.com)
  - **LinkedIn:** [Vedant Roy](https://www.linkedin.com/in/vedant-roy-b58117227/)
  - **GitHub:** [Vedantroy475](https://github.com/Vedantroy475)

