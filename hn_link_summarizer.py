import argparse
import csv
import json
import logging
import os
from typing import Optional, Tuple

import chromadb
import ollama
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup Ollama and ChromaDB clients (assuming you have initialized an Ollama client already)
chroma_client = chromadb.PersistentClient(path="./output/chromadb")
collection = chroma_client.get_or_create_collection(name="article_embeddings")
client = None


def setup_ollama_client(
    base_url: str,
    timeout: int,
    api_key: str,
) -> OpenAI:
    """
    Sets up the Ollama client with the specified parameters.

    Args:
        base_url (str): The base URL of the Ollama API. Default is "http://localhost:11434/v1".
        timeout (int): The timeout for API requests in seconds. Default is 600.
        api_key (str): The API key for authentication. Default is "ollama".

    Returns:
        OpenAI: The configured Ollama client.
    """
    return OpenAI(base_url=base_url, timeout=timeout, api_key=api_key)

def summarize_and_categorize_links_with_ollama(content: str, model: str) -> str:
    """
    Summarizes and categorizes the article content using Ollama.

    Args:
        content (str): The content of the article to summarize and categorize.

    Returns:
        str: The summarized content along with categories.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """
You are a highly advanced model designed to process articles, blog posts, and other written content.
Your task is to summarize the provided content and categorize it based on the main themes, such as technology, finance, health, etc. When summarizing and categorizing, consider:

- The primary focus of the article and its relevance to specific categories.
- Key points and trends that can help in categorizing the content.
- Grouping similar articles together based on their main themes.

**Example of a Focused Summary with Categories:**

1. **Category**: Technology
   - Summary of articles related to new technological advancements, innovations, and their impact on society.

2. **Category**: Finance
   - Summary of articles focusing on financial markets, investment strategies, and economic trends.

Use this structure to ensure the summaries are consistent and meet the needs of users interested in specific categories.
""",
            },
            {"role": "user", "content": content},
        ],
    )
    return response.choices[0].message.content

def summarize_article_with_ollama(content: str, model: str) -> str:
    """
    Summarizes the article content using Ollama.

    Args:
        content (str): The content of the article to summarize.

    Returns:
        str: The summarized content.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """
You are a highly advanced model designed to summarize articles, blog posts, and other written content. Your task is to summarize the provided content by listing the key points as a markdown list. Include only the factual information presented in the text. If no content is provided, state: "No content provided for summarization."

Summarize: Summarize the content as a markdown article and extract the key points from the content and present them as a markdown list.
Fact-based: Ensure the summary includes only factual information from the content.
No Content: If no content is provided, state: "No content provided for summarization."
""",
            },
            {"role": "user", "content": content},
        ],
    )
    return response.choices[0].message.content


def configure_logging(debug: bool) -> None:
    """
    Configures logging to a file and optionally to the console if debug is enabled.

    Args:
        debug (bool): Whether to enable debug logging.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        filename="output/scraper.log", level=logging.INFO, format=log_format
    )

    log_level = logging.DEBUG if debug else logging.INFO
    console = logging.StreamHandler()
    console.setLevel(level=log_level)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logging.getLogger().setLevel(level=log_level)


def extract_article_content(url: str) -> Optional[str]:
    """
    Extracts the main content of an article from a given URL using newspaper3k.

    Args:
        url (str): The URL of the article.

    Returns:
        Optional[str]: The extracted article content, or None if extraction failed.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logging.error(f"Failed to extract article content from {url}: {e}")
        return None


def fallback_extraction(url: str) -> Optional[str]:
    """
    Fallback method to extract article content using BeautifulSoup if newspaper3k fails.

    Args:
        url (str): The URL of the article.

    Returns:
        Optional[str]: The extracted article content, or None if extraction failed.
    """
    try:
        response = requests.get(url, timeout=300)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        return "\n".join([para.get_text() for para in paragraphs])
    except Exception as e:
        logging.error(f"Fallback extraction failed for {url}: {e}")
        return None


def save_content_to_file(content: str, directory: str, filename: str) -> None:
    """
    Saves content to a file in the specified directory.

    Args:
        content (str): The content to save.
        directory (str): The directory to save the file in.
        filename (str): The name of the file to save.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def save_content_to_json(url: str, content: str, directory: str, filename: str) -> None:
    """
    Saves article content and URL to a JSON file in the specified directory.

    Args:
        url (str): The URL of the article.
        content (str): The extracted article content.
        directory (str): The directory to save the file in.
        filename (str): The name of the JSON file to save.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    data = {"url": url, "content": content}
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def fetch_raw_html(
    url: str, session: requests.Session
) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetches the raw HTML content from a given URL.

    Args:
        url (str): The URL to fetch the content from.
        session (requests.Session): The session to use for the request.

    Returns:
        Tuple[Optional[str], Optional[str]]: The raw HTML content and the content type, or None if the fetch failed.
    """
    try:
        response = session.get(url)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        return response.text, content_type
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch raw HTML content from {url}: {e}")
        return None, None


def save_binary_content(
    url: str, session: requests.Session, directory: str, filename: str
) -> None:
    """
    Saves binary content (e.g., PDF) from a given URL to a file in the specified directory.

    Args:
        url (str): The URL to fetch the content from.
        session (requests.Session): The session to use for the request.
        directory (str): The directory to save the file in.
        filename (str): The name of the file to save.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    try:
        response = session.get(url)
        response.raise_for_status()
        file_path = os.path.join(directory, filename)
        with open(file_path, "wb") as file:
            file.write(response.content)
        logging.info(f"Saved binary content for: {url}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch binary content from {url}: {e}")


def validate_csv_headers(reader: csv.DictReader, expected_headers: set) -> bool:
    if not expected_headers.issubset(reader.fieldnames):
        logging.error(
            f"Expected headers {expected_headers} not found in CSV. Found headers: {reader.fieldnames}"
        )
        return False
    return True


def setup_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def create_output_directory(output_dir: str, rank: str, hn_id: str) -> str:
    directory = os.path.join(output_dir, f"{rank}_{hn_id}")
    os.makedirs(directory, exist_ok=True)
    return directory


def process_link(
    row: dict,
    session: requests.Session,
    output_dir: str,
    debug: bool,
    ollama_model: str,
) -> None:
    rank = row["Rank"]
    hn_id = row["hn_id"]
    title = row["title"]
    link = row["link"]
    comments_link = row["comments_link"]

    if debug:
        logging.info(f"Processing: {title} ({link})")

    directory = create_output_directory(output_dir, rank, hn_id)

    raw_html, content_type = fetch_raw_html(link, session)
    if raw_html:
        handle_content(
            link, raw_html, content_type, directory, title, debug, ollama_model, session
        )
    else:
        logging.warning(f"No content fetched for {link}")

    save_content_to_file(comments_link, directory, "comments_link.txt")
    if debug:
        logging.info(f"Saved comments link for: {title}")


def handle_content(
    link: str,
    raw_html: str,
    content_type: str,
    directory: str,
    title: str,
    debug: bool,
    ollama_model: str,
    session: requests.Session,
) -> None:
    if "application/pdf" in content_type:
        save_binary_content(link, session, directory, "document.pdf")
    elif "text/plain" in content_type:
        save_content_to_file(raw_html, directory, "document.txt")
    elif "html" in content_type:
        save_content_to_file(raw_html, directory, "raw_html.html")
        if debug:
            logging.info(f"Saved raw HTML for: {title} : {len(raw_html)} bytes")

        content = extract_article_content(link)
        content_length = len(content) if content else 0
        if debug:
            logging.info(f"Extracted content for: {title} : {content_length} characters")

        if content and content_length > 1000:
            save_extracted_content(content, link, directory, debug, ollama_model)

            # Generate embedding for the content
            response = ollama.embeddings(model="mxbai-embed-large", prompt=content)
            embedding = response["embedding"]
            logging.info(f"Generated embedding for: {title} : {len(embedding)} bytes; {len(content)} characters")

            # Store the embedding and content in ChromaDB
            collection.add(
                ids=[link],  # Using the article link as a unique ID
                embeddings=[embedding],
                documents=[content],
            )
            if debug:
                logging.info(f"Stored embedding for: {title}")

        else:
            logging.warning(
                f"Extracted content is too small or failed for: {title}. Falling back to BeautifulSoup."
            )
            fallback_content = fallback_extraction(link)
            if fallback_content:
                save_extracted_content(
                    fallback_content, link, directory, debug, ollama_model
                )
            else:
                logging.warning(
                    f"Failed to extract content even with fallback for: {title}"
                )
    else:
        logging.warning(f"Unsupported content type for {link}: {content_type}")


def save_extracted_content(
    content: str, link: str, directory: str, debug: bool, ollama_model: str
) -> None:
    save_content_to_file(content, directory, "extracted_article.txt")
    save_content_to_json(link, content, directory, "extracted_article.json")

    # Summarize the content using Ollama
    summary = summarize_article_with_ollama(content, ollama_model)
    save_content_to_file(summary, directory, "summary.md")
    if debug:
        logging.info(f"Saved summary for: {link}")

def main(
    csv_file: str = "output/hacker_news_links.csv",
    md_file: str = "output/hacker_news_links.md",
    output_dir: str = "output/scrape",
    debug: bool = False,
    max_links: Optional[int] = None,
    ollama_model: str = "llama3",
) -> None:
    configure_logging(debug)

    if not os.path.isfile(csv_file):
        logging.error(f"CSV file not found: {csv_file}")
        return

    with open(csv_file, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        expected_headers = {
            "Rank",
            "hn_id",
            "title",
            "link",
            "score",
            "comments",
            "comments_link",
        }
        if not validate_csv_headers(reader, expected_headers):
            return

        titles = [row["title"] for row in reader]
        markdown_list = "\n".join([f"- {title}" for title in titles])
        link_summary = summarize_and_categorize_links_with_ollama(markdown_list, ollama_model)
        logging.info(f"Link summary: {link_summary}")

        # Write link summary to a markdown file
        summary_file = os.path.join(output_dir, "hacker_news_links_summary.md")
        with open(summary_file, "w", encoding="utf-8") as summaryFile:
            summaryFile.write(link_summary)

        session = setup_session()

        # Reset the reader to the beginning
        file.seek(0)
        reader = csv.DictReader(file)

        for row in reader:
            process_link(row, session, output_dir, debug, ollama_model)

if __name__ == "__main__":
    # Step 1: Initialize defaults
    defaults = {
        "csv_file": "output/hacker_news_links.csv",
        "md_file": "output/hacker_news_links.md",
        "output_dir": "output/scrape",
        "debug": False,
        "max_links": None,  # None signifies no limit unless specified
        "config": None,
        "ollama_base_url": "http://localhost:11434/v1",
        "ollama_timeout": 600,
        "ollama_api_key": "ollama",
        "ollama_model": "llama3",
    }

    # Parse arguments
    parser = argparse.ArgumentParser(description="Scrape articles from a CSV of links.")
    parser.add_argument(
        "csv_file", nargs="?", help="Path to the CSV file containing the links."
    )
    parser.add_argument("--output_dir", help="Directory to save the output.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--max_links", type=int, help="Maximum number of links to process."
    )
    parser.add_argument("--config", type=str, help="Path to the config JSON file.")
    args = parser.parse_args()

    # Load configuration from JSON
    if defaults["config"] and os.path.isfile(defaults["config"]):
        with open(defaults["config"], "r") as f:
            config = json.load(f)
        # Update defaults with configuration from JSON
        defaults.update(config)

    # Update defaults with any command-line arguments provided
    for arg in vars(args):
        if getattr(args, arg) is not None:
            defaults[arg] = getattr(args, arg)

    # Set up Ollama clients
    client = setup_ollama_client(
        defaults["ollama_base_url"],
        defaults["ollama_timeout"],
        defaults["ollama_api_key"],
    )

    # Use combined configuration
    main(
        defaults["csv_file"],
        defaults["md_file"],
        defaults["output_dir"],
        defaults["debug"],
        defaults["max_links"],
        defaults["ollama_model"],
    )
