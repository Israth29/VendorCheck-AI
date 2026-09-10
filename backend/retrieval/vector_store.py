import uuid
from langsmith import traceable
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

from config import QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_SIZE

qdrant = QdrantClient(path=QDRANT_PATH)

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE)
    )


def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


@traceable(name="get_embedding")
def get_embedding(text):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text
    )
    return result["embedding"]


def store_chunks_in_qdrant(chunks, metadata):
    points = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        vector = get_embedding(chunk)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk, **metadata}
        )
        points.append(point)
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


@traceable(name="retrieve_relevant_chunks")
def retrieve_relevant_chunks(query, doc_type=None, limit=3):
    """RAG retrieval step: embed the query and fetch the most semantically
    similar stored chunks from Qdrant, optionally filtered by document type
    (cv/supplier)."""
    query_vector = get_embedding(query)

    search_filter = None
    if doc_type:
        search_filter = Filter(
            must=[FieldCondition(key="type", match=MatchValue(value=doc_type))]
        )

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=limit
    ).points

    return results