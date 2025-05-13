"""
Utility functions for the Crawl4AI MCP server using LiteLLM/Ollama and Qdrant.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import uuid # For generating unique IDs for Qdrant points
import asyncio # For getting task name

# Qdrant imports
from qdrant_client import QdrantClient, models

# LiteLLM imports
from litellm import embedding as litellm_embedding, completion as litellm_completion
# from litellm import aembedding as litellm_aembedding, acompletion as litellm_acompletion # For async calls

# --- Qdrant Client Initialization ---
def get_qdrant_client() -> QdrantClient:
    """
    Get a Qdrant client with the URL and API key from environment variables.
    Initializes the collection if it doesn't exist.

    Returns:
        QdrantClient instance
    """
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333") # Default for local dev
    qdrant_api_key = os.getenv("QDRANT_API_KEY") # Optional
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "crawled_content")
    embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", "768")) # Default, ensure it matches your model

    if not qdrant_url or not collection_name:
        raise ValueError("QDRANT_URL and QDRANT_COLLECTION_NAME must be set in environment variables")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    try:
        client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception: # Catches specific Qdrant errors related to collection not found
        print(f"Collection '{collection_name}' not found. Attempting to create it.")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
            )
            print(f"Collection '{collection_name}' created successfully with vector size {embedding_dim}.")
        except Exception as create_e:
            print(f"Failed to create collection '{collection_name}': {create_e}")
            raise create_e
            
    return client

# --- LiteLLM Embedding Functions ---
def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts using LiteLLM, targeting an Ollama model.
    """
    if not texts:
        return []
    
    ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL")
    ollama_api_base = os.getenv("OLLAMA_API_BASE_URL") # URL of your FastAPI/Ollama server
    embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", "768"))

    if not ollama_embedding_model:
        raise ValueError("OLLAMA_EMBEDDING_MODEL environment variable must be set.")
    if not ollama_api_base:
        print("Warning: OLLAMA_API_BASE_URL is not set. LiteLLM will use its default for Ollama (usually http://localhost:11434).")


    try:
        # Ensure texts are not empty strings, as some Ollama models might error
        processed_texts = [t if t.strip() else "empty_string_placeholder" for t in texts]
        
        # LiteLLM uses 'ollama/' prefix for Ollama models if api_base is for an Ollama compatible endpoint
        # If your FastAPI mimics OpenAI, you might not need the 'ollama/' prefix or adjust model name.
        # Assuming your FastAPI exposes Ollama's native API or is OpenAI compatible and LiteLLM handles it.
        model_str = f"ollama/{ollama_embedding_model}" if "ollama/" not in ollama_embedding_model else ollama_embedding_model

        response = litellm_embedding(
            model=model_str, # e.g., "ollama/nomic-embed-text"
            input=processed_texts,
            api_base=ollama_api_base # Explicitly set the API base
        )
        return [item['embedding'] for item in response.data]
    except Exception as e:
        print(f"Error creating batch embeddings with LiteLLM/Ollama (model: {ollama_embedding_model}, base: {ollama_api_base}): {e}")
        return [[0.0] * embedding_dim for _ in range(len(texts))]

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using LiteLLM (Ollama).
    """
    embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * embedding_dim
    except Exception as e:
        print(f"Error creating single embedding with LiteLLM/Ollama: {e}")
        return [0.0] * embedding_dim

# --- LiteLLM Contextual Generation ---
def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk using LiteLLM (Ollama chat model).
    """
    model_choice = os.getenv("MODEL_CHOICE") # e.g., "ollama/llama3"
    ollama_api_base = os.getenv("OLLAMA_API_BASE_URL") # URL of your FastAPI/Ollama server

    if not model_choice:
        print("MODEL_CHOICE for contextual embedding not set. Using original chunk.")
        return chunk, False
    if not ollama_api_base:
         print("Warning: OLLAMA_API_BASE_URL is not set for contextual model. LiteLLM will use its default for Ollama.")

    try:
        prompt = f"""<document>
{full_document[:25000]}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. If the chunk is self-explanatory or very short, you can simply state 'no additional context needed' or restate the chunk."""
        
        # Adjust model string for ollama prefix if needed, similar to embeddings
        model_str = f"ollama/{model_choice}" if "ollama/" not in model_choice else model_choice

        response = litellm_completion(
            model=model_str, # e.g., "ollama/llama3"
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information for search retrieval."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150,
            api_base=ollama_api_base # Explicitly set the API base
        )
        
        context = response.choices[0].message.content.strip()
        
        if "no additional context needed" in context.lower() or not context or len(context) < 10:
            contextual_text = chunk
            was_contextualized = False
        else:
            contextual_text = f"Context: {context}\n---\nChunk: {chunk}"
            was_contextualized = True
        
        return contextual_text, was_contextualized
    
    except Exception as e:
        print(f"Error generating contextual embedding with LiteLLM/Ollama (model: {model_choice}, base: {ollama_api_base}): {e}. Using original chunk instead.")
        return chunk, False

def process_chunk_with_context(args):
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)

# --- Qdrant Data Storage ---
def add_documents_to_qdrant(
    client: QdrantClient,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 50
) -> None:
    """
    Add documents to the Qdrant collection in batches.
    Uses upsert, so existing points with the same ID will be updated.
    Generates a unique ID for each chunk based on URL and chunk_number.
    """
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "crawled_content")
    if not collection_name:
        raise ValueError("QDRANT_COLLECTION_NAME must be set.")

    # Optional: Delete existing points for these URLs if you want a clean replace
    # unique_urls_to_clear = list(set(urls))
    # if unique_urls_to_clear:
    #     print(f"Attempting to clear existing points for URLs: {unique_urls_to_clear}")
    #     try:
    #         client.delete(
    #             collection_name=collection_name,
    #             points_selector=models.FilterSelector(
    #                 filter=models.Filter(
    #                     should=[ # Using 'should' to match any of the URLs
    #                         models.FieldCondition(key="url", match=models.MatchValue(value=url_to_clear))
    #                         for url_to_clear in unique_urls_to_clear
    #                     ]
    #                 )
    #             )
    #         )
    #         print(f"Successfully cleared existing points for specified URLs.")
    #     except Exception as delete_e:
    #         print(f"Warning: Failed to clear some/all existing points for URLs: {delete_e}")


    model_choice_for_context = os.getenv("MODEL_CHOICE")
    use_contextual_embeddings = bool(model_choice_for_context)
    
    all_points_to_upsert: List[models.PointStruct] = []

    for i in range(0, len(contents), batch_size):
        batch_slice_end = min(i + batch_size, len(contents))
        
        current_batch_urls = urls[i:batch_slice_end]
        current_batch_chunk_numbers = chunk_numbers[i:batch_slice_end]
        current_batch_original_contents = contents[i:batch_slice_end]
        current_batch_metadatas = metadatas[i:batch_slice_end]
        
        contents_for_embedding = []
        if use_contextual_embeddings:
            process_args = []
            for j, content in enumerate(current_batch_original_contents):
                url = current_batch_urls[j]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_results = list(executor.map(process_chunk_with_context, process_args))

            for j, (contextual_content, success) in enumerate(future_results):
                contents_for_embedding.append(contextual_content)
                current_batch_metadatas[j]["contextual_embedding_applied"] = success
        else:
            contents_for_embedding = current_batch_original_contents
        
        batch_embeddings = create_embeddings_batch(contents_for_embedding)
        
        for j in range(len(current_batch_urls)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{current_batch_urls[j]}_{current_batch_chunk_numbers[j]}"))
            
            payload = {
                "url": current_batch_urls[j],
                "chunk_number": current_batch_chunk_numbers[j],
                "content": current_batch_original_contents[j], 
                "metadata": current_batch_metadatas[j],
            }
            
            # Ensure embedding has content before adding point
            if batch_embeddings and j < len(batch_embeddings) and any(batch_embeddings[j]):
                 all_points_to_upsert.append(
                    models.PointStruct(
                        id=point_id,
                        vector=batch_embeddings[j],
                        payload=payload
                    )
                )
            else:
                print(f"Warning: Skipping point for {current_batch_urls[j]} chunk {current_batch_chunk_numbers[j]} due to empty embedding.")

    if all_points_to_upsert:
        try:
            client.upsert(collection_name=collection_name, points=all_points_to_upsert)
            print(f"Upserted {len(all_points_to_upsert)} points to Qdrant collection '{collection_name}'.")
        except Exception as e:
            print(f"Error upserting batch to Qdrant: {e}")
            # Optionally, try to upsert one by one if batch fails, for debugging
            # for point_to_upsert in all_points_to_upsert:
            #     try:
            #         client.upsert(collection_name=collection_name, points=[point_to_upsert])
            #     except Exception as single_e:
            #         print(f"Error upserting single point {point_to_upsert.id}: {single_e}")


# --- Qdrant Document Search ---
def search_documents(
    client: QdrantClient,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None 
) -> List[Dict[str, Any]]:
    """
    Search for documents in Qdrant using vector similarity.
    """
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "crawled_content")
    if not collection_name:
        raise ValueError("QDRANT_COLLECTION_NAME must be set.")

    query_embedding = create_embedding(query)
    if not any(query_embedding): 
        print("Failed to generate query embedding. Returning empty results.")
        return []

    qdrant_filter_conditions = None
    if filter_metadata:
        must_conditions = []
        for key, value in filter_metadata.items():
            # This assumes metadata like 'source' is stored under payload.metadata.source
            must_conditions.append(
                models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value))
            )
        if must_conditions:
            qdrant_filter_conditions = models.Filter(must=must_conditions)
            
    try:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter_conditions,
            limit=match_count,
            with_payload=True 
        )
        
        results = []
        for hit in search_result:
            result_item = {
                "id": hit.id,
                "url": hit.payload.get("url") if hit.payload else None,
                "content": hit.payload.get("content") if hit.payload else None, 
                "metadata": hit.payload.get("metadata") if hit.payload else None,
                "similarity": hit.score 
            }
            results.append(result_item)
        return results
        
    except Exception as e:
        print(f"Error searching documents in Qdrant: {e}")
        return []
