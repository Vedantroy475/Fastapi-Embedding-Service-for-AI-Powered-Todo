# embedding_service_bin.py
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector # <-- Import pgvector helper
from datetime import datetime
import numpy as np
import os
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()  # loads .env in working directory

# --- Configuration ---
API_KEY = os.environ.get("EMBED_API_KEY", "dev-key")
# Use DATABASE_URL, which CockroachDB provides
DATABASE_URL = os.environ.get("DATABASE_URL")
MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in .env for embedding service")

# --- Load Embedding Model ---
try:
    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    # Get model dimension
    EMBED_DIM = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Dimension: {EMBED_DIM}")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    raise

# --- CockroachDB Connection & Schema Setup ---
try:
    print("Connecting to CockroachDB...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True  # Set autocommit for a serverless-friendly service
    register_vector(conn)  # <-- Register the VECTOR type
    print("CockroachDB connected.")

    with conn.cursor() as cur:
        # Enable the vector extension (idempotent)
        # Note: CRDB supports vector natively, but pgvector lib might expect this.
        # For CRDB v25.3.3+, this isn't strictly needed but doesn't hurt.
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create embeddings table with a VECTOR column
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                "userId" TEXT NOT NULL,
                "todoId" TEXT NOT NULL,
                text TEXT,
                embedding VECTOR({EMBED_DIM}),
                "createdAt" TIMESTAMPTZ DEFAULT now()
            );
        """)
        
        # Add unique constraint for upserting
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS 
            embeddings_user_todo_idx ON embeddings ("userId", "todoId");
        """)
        
        
        # Replicate the 24-hour TTL
        try:
            cur.execute("""
                ALTER TABLE embeddings SET (
                    ttl_expire_after = '24 hours',
                    ttl_job_cron = '@hourly'
                );
            """)
        except Exception as e:
            if "already exists" not in str(e):
                print(f"Warning: Could not set TTL on embeddings: {e}")

    print("Database schema checked and ready.")

except Exception as e:
    print(f"Error connecting to CockroachDB or setting up schema: {e}")
    raise

# --- FastAPI App Setup ---
app = FastAPI(title="Embedding service (CockroachDB + pgvector)")

# --- Add CORS Middleware (Same as before) ---
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://6905cbac10fb74bdc6151ca8--itaskvedant.netlify.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("CORS middleware added.")

# --- Pydantic Models ---
class EmbedRequest(BaseModel):
    userId: str
    todoId: str
    text: str

class SearchRequest(BaseModel):
    userId: str
    query: str
    k: Optional[int] = 5

class DeleteRequest(BaseModel):
    userId: str
    todoId: str

# --- Helper Function (No longer needed) ---
# def float32_array_to_bson_binary(vec: np.ndarray) -> Binary:
#   pgvector-python handles numpy arrays directly!

# --- API Endpoints ---
@app.post("/embed")
async def embed(req: EmbedRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY: raise HTTPException(status_code=403, detail="invalid api key")
    if not req.text or not req.todoId or not req.userId: 
        raise HTTPException(status_code=400, detail="userId, todoId and text required")

    try:
        vec = model.encode(req.text) # pgvector handles np.array

        # Use UPSERT (INSERT ... ON CONFLICT ... DO UPDATE)
        sql = """
            INSERT INTO embeddings ("userId", "todoId", text, embedding, "createdAt")
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT ("userId", "todoId")
            DO UPDATE SET
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding,
                "createdAt" = EXCLUDED."createdAt";
        """
        with conn.cursor() as cur:
            cur.execute(sql, (req.userId, req.todoId, req.text, vec, datetime.utcnow()))
        
        print(f"Embedded and saved doc for todoId: {req.todoId}")
        return {"status": "ok", "todoId": req.todoId, "dim": EMBED_DIM}
    except Exception as e:
        print(f"Error in /embed endpoint: {e}")
        conn.rollback() # Rollback on error
        raise HTTPException(status_code=500, detail="Failed to create embedding")

@app.post("/search")
async def search(req: SearchRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY: raise HTTPException(status_code=403, detail="invalid api key")
    if not req.query or not req.userId: 
        raise HTTPException(status_code=400, detail="userId and query required")

    try:
        # Generate query vector
        qvec = model.encode(req.query)

        # Use Cosine Distance operator `<=>` which matches our vector_cosine_ops index
        # It returns a "distance" from 0 (identical) to 2 (opposite)
        sql = """
            SELECT 
                "todoId", 
                text, 
                (embedding <=> %s) AS distance
            FROM embeddings
            WHERE "userId" = %s
            ORDER BY distance ASC
            LIMIT %s;
        """
        
        with conn.cursor() as cur:
            cur.execute(sql, (qvec, req.userId, req.k))
            results = cur.fetchall()

        # Format results to match the frontend's expectation
        # We convert distance (0-2) to a "score" (1 to -1)
        # score = 1.0 - distance
        formatted_results = [
            {"todoId": r[0], "text": r[1], "score": 1.0 - r[2]} 
            for r in results
        ]

        print(f"--- FastAPI /search ---")
        print(f"UserID: {req.userId}")
        print(f"Query: {req.query}")
        print(f"Found {len(formatted_results)} results")
        
        return {"results": formatted_results}
    
    except Exception as e:
        print(f"Error during SQL search: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail="Database search failed")

# --- NEW Endpoint for Deleting Embeddings ---
@app.post("/delete")
async def delete(req: DeleteRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY: raise HTTPException(status_code=403, detail="invalid api key")
    if not req.todoId or not req.userId:
        raise HTTPException(status_code=400, detail="userId and todoId required")

    try:
        sql = """
            DELETE FROM embeddings
            WHERE "userId" = %s AND "todoId" = %s;
        """
        with conn.cursor() as cur:
            cur.execute(sql, (req.userId, req.todoId))
            deleted_count = cur.rowcount
            
        print(f"Deleted {deleted_count} embedding(s) for todoId: {req.todoId}")
        return {"status": "ok", "deleted_count": deleted_count}
    except Exception as e:
        print(f"Error in /delete endpoint: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete embedding")


@app.get("/doc/{userId}/{todoId}")
async def get_doc(userId: str, todoId: str, x_api_key: str = Header(None)):
    if x_api_key != API_KEY: raise HTTPException(status_code=403, detail="invalid api key")
    try:
        with conn.cursor() as cur:
            # Cast vector to text for JSON serialization
            cur.execute(
                'SELECT "userId", "todoId", text, embedding::TEXT, "createdAt" FROM embeddings '
                'WHERE "userId" = %s AND "todoId" = %s',
                (userId, todoId)
            )
            res = cur.fetchone()
            if not res:
                raise HTTPException(status_code=404, detail="not found")
            
            # Format as dict
            doc = {
                "userId": res[0],
                "todoId": res[1],
                "text": res[2],
                "embedding_str": res[3], # Show embedding as string
                "createdAt": res[4]
            }
            return doc
    except Exception as e:
        print(f"Error in /doc endpoint: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to retrieve document")
    

class DeleteUserRequest(BaseModel):
    userId: str

@app.post("/delete-user")
async def delete_user(req: DeleteUserRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY: raise HTTPException(status_code=403, detail="invalid api key")
    if not req.userId:
        raise HTTPException(status_code=400, detail="userId required")

    try:
        sql = """
            DELETE FROM embeddings WHERE "userId" = %s;
        """
        with conn.cursor() as cur:
            cur.execute(sql, (req.userId,))
            deleted_count = cur.rowcount
        
        print(f"Deleted {deleted_count} embedding(s) for userId: {req.userId}")
        return {"status": "ok", "deleted_count": deleted_count}
    except Exception as e:
        print(f"Error in /delete-user endpoint: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete user embeddings")