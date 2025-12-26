from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np
from app.core.config import settings
from app.core.graph import graph_db
from app.core.gemini_service import expand_query
import time

# Global variables for lazy loading
client = None
model = None
collection_vectors = None
collection_names = None

def get_client():
    global client
    if client is None:
        if not settings.QDRANT_URL:
            print("Warning: QDRANT_URL not set. Search will fail.")
            return None
        try:
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=60
            )
        except Exception as e:
            print(f"Failed to initialize Qdrant client: {e}")
            return None
    return client

def get_model():
    global model
    if model is None:
        print("Loading Embedding Model...")
        try:
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            print("Model Loaded.")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            return None
    return model

def get_collection_data():
    global collection_vectors, collection_names
    if collection_vectors is None:
        m = get_model()
        if m:
            collection_names = list(settings.COLLECTIONS.keys())
            collection_descriptions = list(settings.COLLECTIONS.values())
            collection_vectors = m.encode(collection_descriptions, normalize_embeddings=True)
        else:
            return [], []
    return collection_vectors, collection_names

def get_embedding(text: str):
    m = get_model()
    if not m:
        return []
    return m.encode(text, normalize_embeddings=True).tolist()

def master_router(query: str, top_k: int = 3):
    """
    Stage 1: Semantic Routing
    Finds the most relevant collections based on vector similarity.
    """
    m = get_model()
    if not m:
        return []
        
    query_vector = m.encode(query, normalize_embeddings=True)
    
    vectors, names = get_collection_data()
    if not vectors is not None and len(vectors) > 0:
         return []

    # Compute cosine similarity
    scores = np.dot(vectors, query_vector)
    
    # Pair with names
    ranked_collections = []
    for name, score in zip(names, scores):
        ranked_collections.append({"name": name, "semantic_score": float(score)})
        
    # Sort by score
    ranked_collections.sort(key=lambda x: x["semantic_score"], reverse=True)
    
    return ranked_collections[:top_k]

async def graph_optimized_search(query: str, page: int = 1, limit: int = 15):
    """
    Stage 2 & 3: Graph Re-ranking & Search
    Combines semantic score with graph weights and searches Qdrant.
    """
    start_time = time.time()
    
    # 0. Query Expansion
    expanded_query = await expand_query(query)
    
    # 1. Get Semantic Candidates
    candidates = master_router(expanded_query, top_k=4)
    
    # 2. Apply Graph Weights (Re-ranking)
    final_routes = []
    for item in candidates:
        col_name = item["name"]
        semantic_score = item["semantic_score"]
        
        # Get learned weight from graph
        graph_weight = graph_db.get_weight(col_name)
        
        # Hybrid Score: 70% Semantic + 30% Graph (Log-scaled)
        # Using log(weight) to prevent runaway weights
        hybrid_score = semantic_score + (0.2 * np.log1p(graph_weight))
        
        final_routes.append({
            "name": col_name,
            "display_name": settings.COLLECTION_DISPLAY_NAMES.get(col_name, col_name),
            "score": hybrid_score,
            "semantic": semantic_score,
            "graph_w": graph_weight
        })
        
    # Sort by hybrid score
    final_routes.sort(key=lambda x: x["score"], reverse=True)
    
    # Select top collections to search (e.g., top 2 or 3)
    selected_collections = final_routes[:3]
    
    # Calculate pagination
    num_collections = len(selected_collections)
    per_collection_limit = max(1, limit // num_collections)
    offset = (page - 1) * per_collection_limit

    # 3. Perform Vector Search in Selected Collections
    query_vector = get_embedding(expanded_query)
    all_results = []
    
    c = get_client()
    if not c:
        return {"results": [], "latency": 0, "routed_to": []}

    for route in selected_collections:
        col_name = route["name"]
        display_name = route["display_name"]
        try:
            hits = c.query_points(
                collection_name=col_name,
                query=query_vector,
                limit=per_collection_limit,
                offset=offset,
                with_payload=True
            ).points
            
            # If we found results, update the graph (Feedback Loop)
            if hits:
                graph_db.update(col_name, reward=0.5)
            
            for hit in hits:
                payload = hit.payload
                all_results.append({
                    "id": hit.id,
                    "score": hit.score, # Vector similarity
                    "collection": col_name, # Raw collection name for API calls
                    "display_collection": display_name, # Pretty name for UI
                    "title": payload.get("title", "No Title"),
                    "abstract": payload.get("abstract", ""),
                    "year": payload.get("publication_year", "N/A"),
                    "date": payload.get("publication_date", ""),
                    "venue": payload.get("venue", "Unknown Venue"),
                    "citations": payload.get("citation_count", 0),
                    "url": payload.get("url", "#"),
                    "doi": payload.get("doi", ""),
                    "is_oa": payload.get("is_open_access", False),
                    "authors": payload.get("authors", []),
                    "concepts": payload.get("concepts", [])[:5] # Top 5 concepts
                })
                
        except Exception as e:
            print(f"Error searching collection {col_name}: {e}")
            
    # 4. Final Merge & Sort
    # Sort all results by their vector score
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    latency = time.time() - start_time
    
    return {
        "results": all_results[:limit],
        "latency": round(latency, 3),
        "routed_to": selected_collections,
        "page": page
    }

def get_suggestions(query: str):
    """
    Provides real-time suggestions and routing preview.
    """
    query = query.lower()
    suggestions = []
    
    # 1. Keyword Matching
    for col, keywords in settings.DOMAIN_KEYWORDS.items():
        for k in keywords:
            if query in k and k != query: # Simple substring match
                suggestions.append(k)
            if len(suggestions) >= 5:
                break
        if len(suggestions) >= 5:
            break
            
    # 2. Routing Prediction (Fast Semantic Check)
    # We use a smaller top_k=1 just to see where it would go
    predicted_route = master_router(query, top_k=1)[0]
    col_name = predicted_route["name"]
    
    return {
        "completions": suggestions[:3],
        "predicted_domain": settings.COLLECTION_DISPLAY_NAMES.get(col_name, col_name),
        "graph_weight": graph_db.get_weight(col_name)
    }

def get_paper_details(collection_name: str, paper_id: str):
    """
    Fetches a single paper's details from Qdrant by ID.
    """
    c = get_client()
    if not c:
        return None

    try:
        # Try to convert ID to integer if it looks like one
        # Qdrant requires integer IDs to be passed as integers, not strings
        if str(paper_id).isdigit():
            qdrant_id = int(paper_id)
        else:
            qdrant_id = paper_id

        # Qdrant retrieve API
        points = c.retrieve(
            collection_name=collection_name,
            ids=[qdrant_id],
            with_payload=True
        )
        
        if not points:
            return None
            
        point = points[0]
        payload = point.payload
        
        return {
            "id": point.id,
            "collection": collection_name,
            "display_collection": settings.COLLECTION_DISPLAY_NAMES.get(collection_name, collection_name),
            "title": payload.get("title", "No Title"),
            "abstract": payload.get("abstract", ""),
            "year": payload.get("publication_year", "N/A"),
            "date": payload.get("publication_date", ""),
            "venue": payload.get("venue", "Unknown Venue"),
            "citations": payload.get("citation_count", 0),
            "url": payload.get("url", "#"),
            "doi": payload.get("doi", ""),
            "authors": payload.get("authors", []),
            "concepts": payload.get("concepts", [])
        }
    except Exception as e:
        print(f"Error fetching paper {paper_id} from {collection_name}: {e}")
        return None
