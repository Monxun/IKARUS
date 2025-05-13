"""
MCP server for web crawling with Crawl4AI, using LiteLLM/Ollama and Qdrant.
"""
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from pathlib import Path
import requests
import asyncio
import json
import os
import re

# Qdrant client import
from qdrant_client import QdrantClient

# Crawl4AI imports
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

# Updated utils import
from utils import (
    get_qdrant_client,
    add_documents_to_qdrant,
    search_documents,
    # generate_contextual_embedding, # Not directly called here, but by add_documents_to_qdrant
    # create_embedding # Not directly called here, but by search_documents
)

# Load environment variables from the project root .env file if it exists
# For Docker, environment variables are typically passed differently (e.g., .env file with docker-compose)
# but this allows local execution too.
project_root = Path(__file__).resolve().parent
dotenv_path = project_root / '.env'
if dotenv_path.exists():
    print(f"Loading environment variables from {dotenv_path}")
    load_dotenv(dotenv_path, override=True)
else:
    print(f".env file not found at {dotenv_path}, relying on environment.")


# Helper functions for chunking and metadata (can be part of utils.py too)
def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)
    if not text or text.isspace(): # Handle empty or whitespace-only text
        return []

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
        
        current_slice = text[start:end]
        
        # Try to find a code block boundary first (```)
        # Search backwards from the end of the current_slice
        code_block_end_index = current_slice.rfind('```') 
        
        # Ensure the found '```' is the end of a block, not the start of one too close to 'start'
        if code_block_end_index != -1:
            # Check if it's a start or end marker. A simple heuristic:
            # count occurrences up to this point in the original text.
            num_code_markers_before = text[:start + code_block_end_index].count('```')
            if num_code_markers_before % 2 == 0: # Even number means this ``` closes a block
                 # Only break if we're past a certain threshold to avoid tiny chunks
                if code_block_end_index > chunk_size * 0.3:
                    end = start + code_block_end_index + 3 # Include the ```
            # If it's an opening ```, try to find its closing ``` within the chunk or extend
            # This part can get complex; for now, a simpler split is used.
            # A more robust solution would parse markdown structure.

        # If no suitable code block boundary, try to break at a paragraph
        # Search backwards for a double newline
        last_paragraph_break = current_slice.rfind('\n\n')
        if last_paragraph_break > chunk_size * 0.3: # Only break if we're past 30%
            end = start + last_paragraph_break + 2 # Include the \n\n

        # If no paragraph break, try to break at a sentence (less ideal but a fallback)
        # Search backwards for a period followed by a space
        elif '. ' in current_slice:
            last_sentence_break = current_slice.rfind('. ')
            if last_sentence_break > chunk_size * 0.3: # Only break if we're past 30%
                end = start + last_sentence_break + 1 # Include the period

        # Extract chunk and clean it up
        chunk_content = text[start:end].strip()
        if chunk_content: # Ensure chunk is not empty after stripping
            chunks.append(chunk_content)
        
        start = end
        if start >= text_length and chunk_content != text[start:].strip() and text[start:].strip(): # Add residue if any
             residue = text[start:].strip()
             if residue:
                chunks.append(residue)
                break


    return [c for c in chunks if c] # Filter out any empty strings that might have slipped through

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """Extracts headers and stats from a chunk."""
    headers = re.findall(r'^(#{1,6})\s+(.+)$', chunk, re.MULTILINE) # Limit to H1-H6
    header_str = '; '.join([f'{h[0]} {h[1].strip()}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

# Dataclass for application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    qdrant_client: QdrantClient
    
@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """Manages the Crawl4AI client and Qdrant client lifecycle."""
    print("Initializing Crawl4AI lifespan...")
    browser_config = BrowserConfig(
        headless=True,
        verbose=False # Set to True for more detailed crawl4ai logs
    )
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    print("Initializing Qdrant client...")
    qdrant_client_instance = get_qdrant_client() # From utils.py
    
    try:
        print("Crawl4AI and Qdrant clients initialized. Yielding context.")
        yield Crawl4AIContext(
            crawler=crawler,
            qdrant_client=qdrant_client_instance
        )
    finally:
        print("Cleaning up Crawl4AI lifespan...")
        await crawler.__aexit__(None, None, None)
        # Qdrant client (sync) typically doesn't require explicit close for HTTP.
        # If using an async client, it might have an `aclose()` method.
        # qdrant_client_instance.close() # If applicable
        print("Crawl4AI lifespan cleanup complete.")

# Initialize FastMCP server
mcp_host = os.getenv("HOST", "0.0.0.0")
mcp_port = int(os.getenv("PORT", "8051"))

mcp = FastMCP(
    "mcp-crawl4ai-rag-ollama",
    description="MCP server for RAG and web crawling with Crawl4AI, LiteLLM/Ollama, and Qdrant",
    lifespan=crawl4ai_lifespan,
    host=mcp_host,
    port=mcp_port
)
print(f"FastMCP server '{mcp.title}' configured to run on {mcp_host}:{mcp_port}")

def is_sitemap(url: str) -> bool:
    """Check if a URL is a sitemap."""
    parsed_url = urlparse(url)
    return url.endswith('sitemap.xml') or 'sitemap' in parsed_url.path.lower()

def is_txt(url: str) -> bool:
    """Check if a URL is a text file (simple check)."""
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    """Parse a sitemap and extract URLs."""
    urls = []
    try:
        # Added a User-Agent as some servers block requests without it
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; Crawl4AI-MCP-Bot/1.0; +http://example.com/bot)'}
        resp = requests.get(sitemap_url, headers=headers, timeout=10)
        resp.raise_for_status() # Raise an exception for HTTP errors
        
        # Basic XML parsing, consider robust libraries like `lxml` for complex sitemaps
        tree = ElementTree.fromstring(resp.content)
        # Namespace-agnostic search for 'loc' tags
        for loc_element in tree.findall('.//{*}loc'):
            if loc_element.text:
                urls.append(loc_element.text.strip())
    except requests.RequestException as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")
    except ElementTree.ParseError as e:
        print(f"Error parsing sitemap XML from {sitemap_url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while parsing sitemap {sitemap_url}: {e}")
    return list(set(urls)) # Return unique URLs

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Qdrant.
    Ideal for quickly retrieving content from a specific URL.
    """
    print(f"Tool 'crawl_single_page' called for URL: {url}")
    try:
        crawler: AsyncWebCrawler = ctx.request_context.lifespan_context.crawler
        qdrant_client_ctx: QdrantClient = ctx.request_context.lifespan_context.qdrant_client
        
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            print(f"Successfully crawled {url}. Content length: {len(result.markdown)}")
            chunks = smart_chunk_markdown(result.markdown)
            if not chunks:
                print(f"No content chunks generated for {url} after parsing markdown.")
                return json.dumps({"success": False, "url": url, "error": "No content chunks generated after parsing."}, indent=2)

            urls_list, chunk_numbers_list, contents_list, metadatas_list = [], [], [], []
            
            current_task = asyncio.current_task()
            crawl_time_str = current_task.get_name() if current_task else "N/A"

            for i, chunk_content in enumerate(chunks):
                urls_list.append(url)
                chunk_numbers_list.append(i)
                contents_list.append(chunk_content)
                
                meta = extract_section_info(chunk_content)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = urlparse(url).netloc
                meta["crawl_time"] = crawl_time_str
                metadatas_list.append(meta)
            
            url_to_full_document = {url: result.markdown}
            
            print(f"Adding {len(chunks)} chunks from {url} to Qdrant.")
            add_documents_to_qdrant(
                client=qdrant_client_ctx,
                urls=urls_list,
                chunk_numbers=chunk_numbers_list,
                contents=contents_list,
                metadatas=metadatas_list,
                url_to_full_document=url_to_full_document
            )
            
            return json.dumps({
                "success": True, "url": url, "chunks_stored": len(chunks),
                "content_length": len(result.markdown),
                "links_count": {"internal": len(result.links.get("internal", [])), "external": len(result.links.get("external", []))}
            }, indent=2)
        else:
            print(f"Failed to crawl {url}. Error: {result.error_message}")
            return json.dumps({"success": False, "url": url, "error": result.error_message or "Crawling failed, no markdown content."}, indent=2)
    except Exception as e:
        print(f"Exception in 'crawl_single_page' for {url}: {e}")
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)

# --- Helper functions for smart_crawl_url ---
async def _crawl_markdown_file_content(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """Helper to crawl a .txt or markdown file (assumes direct content)."""
    crawl_config = CrawlerRunConfig(extract_raw_html=False) # Assuming .txt is plain text
    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown: # crawl4ai might put text content in markdown field
        return [{'url': url, 'markdown': result.markdown}]
    elif result.success and result.raw_html: # Fallback for .txt if markdown is empty
         return [{'url': url, 'markdown': result.raw_html}] # Treat raw_html as markdown for .txt
    else:
        print(f"Failed to crawl text/markdown file {url}: {result.error_message}")
        return []

async def _crawl_batch_urls(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int) -> List[Dict[str, Any]]:
    """Helper to batch crawl multiple URLs."""
    if not urls: return []
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    # Using default dispatcher or configure MemoryAdaptiveDispatcher if needed
    # dispatcher = MemoryAdaptiveDispatcher(max_session_permit=max_concurrent)
    results = await crawler.arun_many(urls=urls, config=crawl_config) #, dispatcher=dispatcher)
    return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]

async def _crawl_recursive_internal(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int, max_concurrent: int) -> List[Dict[str, Any]]:
    """Helper for recursive internal link crawling."""
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    # dispatcher = MemoryAdaptiveDispatcher(max_session_permit=max_concurrent)
    
    visited_normalized = set()
    all_results_data = []
    
    current_urls_to_crawl = set(urldefrag(u)[0].lower() for u in start_urls)
    
    for depth in range(max_depth):
        if not current_urls_to_crawl:
            print(f"Recursive crawl: No new URLs to crawl at depth {depth}.")
            break
        
        # Filter out already visited URLs before crawling
        urls_for_this_depth = [url for url in current_urls_to_crawl if url not in visited_normalized]
        if not urls_for_this_depth:
            print(f"Recursive crawl: All URLs at depth {depth} already visited.")
            break

        print(f"Recursive crawl: Depth {depth}, crawling {len(urls_for_this_depth)} URLs.")
        
        # Add to visited *before* crawling to handle concurrent additions to next_level_urls
        for u_norm in urls_for_this_depth:
            visited_normalized.add(u_norm)

        # results = await crawler.arun_many(urls=list(urls_for_this_depth), config=run_config, dispatcher=dispatcher)
        results = await crawler.arun_many(urls=list(urls_for_this_depth), config=run_config)


        next_level_urls_normalized = set()
        for result in results:
            if result.success and result.markdown:
                all_results_data.append({'url': result.url, 'markdown': result.markdown})
                # Ensure internal links are properly formed and normalized
                for link_info in result.links.get("internal", []):
                    href = link_info.get("href")
                    if href:
                        normalized_link = urldefrag(href)[0].lower()
                        if normalized_link not in visited_normalized:
                            next_level_urls_normalized.add(normalized_link)
            # else:
            #     print(f"Recursive crawl: Failed or no markdown for {result.url} at depth {depth}. Error: {result.error_message}")
        
        current_urls_to_crawl = next_level_urls_normalized

    print(f"Recursive crawl finished. Total pages processed: {len(all_results_data)}")
    return all_results_data
# --- End of helper functions for smart_crawl_url ---

@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 2, max_concurrent_sessions: int = 5, chunk_size_chars: int = 4000) -> str:
    """
    Intelligently crawl a URL (webpage, sitemap, or .txt file) and store content in Qdrant.
    For regular webpages, recursively crawls internal links up to max_depth.
    """
    print(f"Tool 'smart_crawl_url' called for URL: {url}, max_depth: {max_depth}, max_concurrent: {max_concurrent_sessions}, chunk_size: {chunk_size_chars}")
    try:
        crawler: AsyncWebCrawler = ctx.request_context.lifespan_context.crawler
        qdrant_client_ctx: QdrantClient = ctx.request_context.lifespan_context.qdrant_client
        
        crawled_docs_data = []
        crawl_operation_type = "webpage" # Default
        
        if is_txt(url):
            print(f"Detected .txt file: {url}. Performing direct content fetch.")
            crawl_operation_type = "text_file"
            crawled_docs_data = await _crawl_markdown_file_content(crawler, url)
        elif is_sitemap(url):
            print(f"Detected sitemap: {url}. Parsing and crawling listed URLs.")
            crawl_operation_type = "sitemap"
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({"success": False, "url": url, "error": "No URLs found in sitemap or sitemap parsing failed."}, indent=2)
            print(f"Found {len(sitemap_urls)} URLs in sitemap. Batch crawling...")
            crawled_docs_data = await _crawl_batch_urls(crawler, sitemap_urls, max_concurrent_sessions)
        else:
            print(f"Detected regular webpage: {url}. Performing recursive crawl (max_depth: {max_depth}).")
            crawl_operation_type = "webpage_recursive"
            crawled_docs_data = await _crawl_recursive_internal(crawler, [url], max_depth, max_concurrent_sessions)
        
        if not crawled_docs_data:
            print(f"No content successfully crawled for {url} (type: {crawl_operation_type}).")
            return json.dumps({"success": False, "url": url, "error": "No content found or all crawl attempts failed."}, indent=2)
        
        print(f"Successfully crawled {len(crawled_docs_data)} documents for {url}. Processing and adding to Qdrant...")
        urls_list, chunk_numbers_list, contents_list, metadatas_list = [], [], [], []
        total_chunks_count = 0
        
        current_task = asyncio.current_task()
        crawl_time_str = current_task.get_name() if current_task else "N/A"

        for doc_data in crawled_docs_data:
            doc_url = doc_data['url']
            markdown_content = doc_data['markdown']
            chunks = smart_chunk_markdown(markdown_content, chunk_size=chunk_size_chars)
            
            for i, chunk_content in enumerate(chunks):
                urls_list.append(doc_url)
                chunk_numbers_list.append(i)
                contents_list.append(chunk_content)
                
                meta = extract_section_info(chunk_content)
                meta["chunk_index"] = i
                meta["url"] = doc_url
                meta["source"] = urlparse(doc_url).netloc
                meta["crawl_type"] = crawl_operation_type
                meta["crawl_time"] = crawl_time_str
                metadatas_list.append(meta)
                total_chunks_count += 1
        
        url_to_full_document_map = {doc['url']: doc['markdown'] for doc in crawled_docs_data if doc.get('markdown')}
        
        add_documents_to_qdrant(
            client=qdrant_client_ctx,
            urls=urls_list,
            chunk_numbers=chunk_numbers_list,
            contents=contents_list,
            metadatas=metadatas_list,
            url_to_full_document=url_to_full_document_map,
            # batch_size can be configured via env var or kept default in add_documents_to_qdrant
        )
        
        return json.dumps({
            "success": True, "url": url, "crawl_type": crawl_operation_type,
            "pages_crawled_count": len(crawled_docs_data), 
            "chunks_stored_count": total_chunks_count,
            "crawled_urls_sample": [doc['url'] for doc in crawled_docs_data][:5] + (["..."] if len(crawled_docs_data) > 5 else [])
        }, indent=2)
    except Exception as e:
        print(f"Exception in 'smart_crawl_url' for {url}: {e}")
        # import traceback
        # traceback.print_exc() # For more detailed error logging during development
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)

@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available unique source domains from Qdrant.
    Note: This scrolls through Qdrant points; may be slow on very large collections.
    """
    print("Tool 'get_available_sources' called.")
    try:
        qdrant_client_ctx: QdrantClient = ctx.request_context.lifespan_context.qdrant_client
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "crawled_content")
        if not collection_name:
            return json.dumps({"success": False, "error": "QDRANT_COLLECTION_NAME not set"}, indent=2)

        unique_sources = set()
        current_offset = None 
        scroll_limit = 250 # How many points to fetch per scroll request

        print(f"Scrolling Qdrant collection '{collection_name}' to find unique sources...")
        while True:
            try:
                # Fetch only the 'metadata.source' field if Qdrant's payload selection allows it.
                # Here, fetching the 'metadata' dictionary and then extracting 'source'.
                scroll_response = qdrant_client_ctx.scroll(
                    collection_name=collection_name,
                    offset=current_offset,
                    limit=scroll_limit,
                    with_payload=["metadata.source"], # Try to be specific
                    # If the above doesn't work due to qdrant version/setup, use:
                    # with_payload=models.PayloadSelector(include=["metadata"]),
                    with_vectors=False
                )
                
                hits = scroll_response[0] # scroll_result is a tuple (hits, next_page_offset)
                next_page_offset = scroll_response[1]

                if not hits:
                    print("No more hits from Qdrant scroll.")
                    break

                for hit in hits:
                    # Handle different ways 'metadata.source' might be returned by with_payload
                    source_val = None
                    if hit.payload:
                        if "metadata.source" in hit.payload: # If specific path worked
                            source_val = hit.payload["metadata.source"]
                        elif "metadata" in hit.payload and isinstance(hit.payload["metadata"], dict) and "source" in hit.payload["metadata"]:
                            source_val = hit.payload["metadata"]["source"]
                    
                    if source_val:
                        unique_sources.add(source_val)
                
                current_offset = next_page_offset
                if current_offset is None: 
                    print("Reached end of Qdrant scroll (next_offset is None).")
                    break
                # print(f"Scrolled {len(hits)} points, next offset: {current_offset}. Total unique sources so far: {len(unique_sources)}")

            except Exception as scroll_e:
                print(f"Error during Qdrant scroll for sources: {scroll_e}")
                break 

        sources_list = sorted(list(unique_sources))
        print(f"Found {len(sources_list)} unique sources.")
        return json.dumps({"success": True, "sources": sources_list, "count": len(sources_list)}, indent=2)
    except Exception as e:
        print(f"Exception in 'get_available_sources': {e}")
        return json.dumps({"success": False, "error": f"Failed to get sources from Qdrant: {str(e)}"}, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source_domain_filter: str = None, num_results: int = 5) -> str:
    """
    Perform a RAG query on Qdrant. Optionally filter by source domain.
    """
    print(f"Tool 'perform_rag_query' called. Query: '{query}', Source Filter: {source_domain_filter}, Num Results: {num_results}")
    try:
        qdrant_client_ctx: QdrantClient = ctx.request_context.lifespan_context.qdrant_client
        
        filter_conditions = None
        if source_domain_filter and source_domain_filter.strip():
            # This assumes 'source' is a field within the 'metadata' dict in your Qdrant payload
            filter_conditions = {"source": source_domain_filter.strip()} 
        
        print(f"Performing search in Qdrant with filter: {filter_conditions}")
        results = search_documents( # This is your updated utils.search_documents
            client=qdrant_client_ctx,
            query=query,
            match_count=num_results,
            filter_metadata=filter_conditions # Passed as a dict e.g. {"source": "example.com"}
        )
        
        print(f"Search returned {len(results)} results.")
        return json.dumps({
            "success": True, "query": query,
            "source_filter_applied": source_domain_filter if source_domain_filter and source_domain_filter.strip() else None,
            "results": results, # results are already formatted by search_documents
            "count": len(results)
        }, indent=2)
    except Exception as e:
        print(f"Exception in 'perform_rag_query': {e}")
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)

async def main():
    """Main function to run the MCP server."""
    transport_mode = os.getenv("TRANSPORT", "sse").lower()
    print(f"Starting MCP server with transport: {transport_mode}")
    if transport_mode == 'sse':
        await mcp.run_sse_async()
    elif transport_mode == 'stdio':
        await mcp.run_stdio_async()
    else:
        print(f"Warning: Unknown TRANSPORT mode '{transport_mode}'. Defaulting to SSE.")
        await mcp.run_sse_async()

if __name__ == "__main__":
    print("MCP application starting...")
    asyncio.run(main())
