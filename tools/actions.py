import requests
import os
from typing import List, Dict, Any
import json
from dotenv import load_dotenv
from urllib.parse import urlparse
import time
from atla_insights import tool

# Load environment variables from .env file
load_dotenv()


@tool
def google_search(query: str) -> str:
    """
    Search Google and return formatted results. Use this to find information about any topic.

    This tool searches the web using Google's Custom Search API and returns the top 10 results
    with titles, URLs, and snippets. Perfect for finding current information, news, tutorials,
    or any web content.

    Args:
        query: The search query string (e.g., "latest stock market news", "Python tutorial", "weather forecast")

    Returns:
        Formatted string with search results including titles, URLs, and descriptions
    """
    # Get API credentials from environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key:
        return "Error: Google API key not found. Please set GOOGLE_API_KEY environment variable."

    if not search_engine_id:
        return "Error: Google Search Engine ID not found. Please set GOOGLE_SEARCH_ENGINE_ID environment variable."

    # Google Custom Search API endpoint
    url = "https://www.googleapis.com/customsearch/v1"

    # Parameters for the API request
    params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": query,
        "num": 10,  # Number of results (max 10 for free tier)
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        # Extract search results
        results = []
        if "items" in data:
            for item in data["items"]:
                result = {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "displayLink": item.get("displayLink", ""),
                }
                results.append(result)
        else:
            return f"No results found for query: '{query}'"

        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = f"{i}. {result['title']}\n   URL: {result['link']}\n   {result['snippet']}\n"
            formatted_results.append(formatted_result)

        return (
            f"Top {len(results)} Google search results for '{query}':\n\n"
            + "\n".join(formatted_results)
        )

    except requests.exceptions.RequestException as e:
        return f"Error: API request failed: {str(e)}"
    except json.JSONDecodeError as e:
        return f"Error: Failed to parse API response: {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error: {str(e)}"


@tool
def open_webpage(url: str) -> str:
    """
    Open a webpage and extract its content. Use this to read articles, news, or any web content.

    This tool fetches the HTML content of a webpage and extracts readable text. Perfect for
    reading articles, news stories, documentation, or any web content you want to analyze.

    Args:
        url: The URL of the webpage to open (e.g., "https://example.com/article", "https://news.bbc.co.uk/story")

    Returns:
        Extracted text content from the webpage
    """
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return "Error: Invalid URL format. Please provide a complete URL starting with http:// or https://"

        # Set headers to mimic a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Fetch the webpage
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Check if response is HTML
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type:
            return (
                f"Error: URL does not return HTML content. Content type: {content_type}"
            )

        # Extract text content (basic implementation)
        # For better text extraction, you might want to use libraries like beautifulsoup4
        html_content = response.text

        # Simple text extraction (remove HTML tags)
        import re

        # Remove script and style elements
        html_content = re.sub(
            r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL
        )
        html_content = re.sub(
            r"<style[^>]*>.*?</style>", "", html_content, flags=re.DOTALL
        )

        # Remove HTML tags
        text_content = re.sub(r"<[^>]+>", " ", html_content)

        # Clean up whitespace
        text_content = re.sub(r"\s+", " ", text_content).strip()

        # Limit content length to avoid overwhelming the model
        max_length = 8000
        if len(text_content) > max_length:
            text_content = text_content[:max_length] + "... [Content truncated]"

        return f"Content from {url}:\n\n{text_content}"

    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch webpage: {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error while opening webpage: {str(e)}"


@tool
def search_and_read(query: str, num_articles: int = 2) -> str:
    """
    Search for information and read the top articles to provide comprehensive information.

    This tool combines search and webpage reading to give you detailed information from
    multiple sources. It searches for your query, then reads the top articles to provide
    comprehensive, well-sourced information.

    Args:
        query: The search query (e.g., "latest AI developments", "climate change news")
        num_articles: Number of articles to read (default: 2, max: 5)

    Returns:
        Comprehensive information from multiple web sources
    """
    # Use google_search to find articles
    search_results_str = google_search(query)
    urls = extract_urls_from_search_results(search_results_str)

    if not urls:
        return f"No articles found for query: '{query}'"

    # Read the top articles
    articles_content = []
    for i, url in enumerate(urls[:num_articles], 1):
        print(f"\nReading article {i} of {num_articles}...")
        article_content = open_webpage(url)
        articles_content.append(article_content)
        print(f"Finished reading article {i}.")

    # Combine all article content
    combined_content = "\n\n".join(articles_content)

    return (
        f"Comprehensive information from {num_articles} articles:\n\n{combined_content}"
    )


def extract_urls_from_search_results(search_results: str) -> List[str]:
    """
    Extract URLs from Google search results.

    Args:
        search_results: The formatted search results string

    Returns:
        List of URLs found in the search results
    """
    import re

    # Extract URLs from the search results
    url_pattern = r"URL:\s*(https?://[^\s\n]+)"
    urls = re.findall(url_pattern, search_results)
    return urls
