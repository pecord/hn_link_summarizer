import argparse
import csv
import json
import logging
import os
from typing import Optional, Tuple

import requests
from bs4 import BeautifulSoup
from newspaper import Article
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

client = setup_ollama_client()

def summarize_with_ollama(content: str) -> str:
    """
    Summarizes the article content using Ollama.

    Args:
        content (str): The content of the article to summarize.

    Returns:
        str: The summarized content.
    """
    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": """
# System Prompt for Article Summarization

You are a highly advanced summarization model designed to process articles, blog posts, and other written content. Your task is to summarize the provided content with a specific focus on investing and technology. When summarizing, pay special attention to:

- Companies mentioned in the article and their relevance to investing.
- Technologies discussed, particularly those that might impact the market or have significant investment potential.
- Key points and trends related to investing and technology.

Ensure the summary is concise, informative, and highlights the most critical aspects relevant to investors and technology enthusiasts.

**Example of a Focused Summary:**

1. **Company Mentions**:
   - Discussed companies and their current market positions or innovations.
   - Any recent developments or news related to these companies that might affect their stock or market performance.

2. **Technology Highlights**:
   - Notable technologies mentioned in the article.
   - Potential impact of these technologies on various industries or the market.

3. **Investment Insights**:
   - Trends or insights related to investing.
   - Advice or opinions expressed in the article that might be of interest to investors.

Use this structure to ensure the summaries are consistent and meet the needs of users interested in investing and technology.
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

    if debug:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
        logging.getLogger().setLevel(logging.DEBUG)


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


def main(
    csv_file: str,
    output_dir: str = "output/scrape",
    debug: bool = False,
    max_links: Optional[int] = None,
    ollama_model: str = "llama3",
) -> None:
    """
    Reads a CSV file of links and processes each link to extract and save content.

    Args:
        csv_file (str): The path to the CSV file containing the links.
        output_dir (str): The directory to save the output. Default is "output/scrape".
        debug (bool): Whether to enable debug logging. Default is False.
        max_links (Optional[int]): The maximum number of links to process. Default is None (process all links).
    """
    configure_logging(debug)

    if not os.path.isfile(csv_file):
        logging.error(f"CSV file not found: {csv_file}")
        return

    with open(csv_file, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        # Check if the expected headers are present
        expected_headers = {
            "Rank",
            "hn_id",
            "title",
            "link",
            "score",
            "comments",
            "comments_link",
        }
        if not expected_headers.issubset(reader.fieldnames):
            logging.error(
                f"Expected headers {expected_headers} not found in CSV. Found headers: {reader.fieldnames}"
            )
            return

        # Initialize session with retry strategy
        session = requests.Session()
        retries = Retry(
            total=5, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504]
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))
        session.mount("https://", HTTPAdapter(max_retries=retries))

        for i, row in enumerate(reader):
            if max_links is not None and i >= max_links:
                break

            rank = row["Rank"]
            hn_id = row["hn_id"]
            title = row["title"]
            link = row["link"]
            comments_link = row["comments_link"]

            if debug:
                logging.info(f"Processing: {title} ({link})")

            # Create directory for each hn_id prefixed with rank within the output directory
            directory = os.path.join(output_dir, f"{rank}_{hn_id}")
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save raw HTML content
            raw_html, content_type = fetch_raw_html(link, session)
            if raw_html:
                if "application/pdf" in content_type:
                    save_binary_content(link, session, directory, "document.pdf")
                elif "text/plain" in content_type:
                    save_content_to_file(raw_html, directory, "document.txt")
                elif "html" in content_type:
                    save_content_to_file(raw_html, directory, "raw_html.html")
                    if debug:
                        logging.info(f"Saved raw HTML for: {title}")

                    # Extract and save article content
                    content = extract_article_content(link)
                    if content and len(content) > 1000:
                        save_content_to_file(
                            content, directory, "extracted_article.txt"
                        )
                        save_content_to_json(
                            link, content, directory, "extracted_article.json"
                        )

                        # Summarize and save the summarized content
                        summary = summarize_with_ollama(content)
                        save_content_to_file(summary, directory, "summary.md")
                        if debug:
                            logging.info(f"Saved summary for: {title}")
                    else:
                        logging.warning(
                            f"Extracted content is too small or failed for: {title}. Falling back to BeautifulSoup."
                        )
                        content = fallback_extraction(link)
                        if content:
                            save_content_to_file(
                                content, directory, "extracted_article.txt"
                            )
                            save_content_to_json(
                                link, content, directory, "extracted_article.json"
                            )

                            # Summarize and save the summarized content
                            summary = summarize_with_ollama(content)
                            save_content_to_file(summary, directory, "summary.md")
                            if debug:
                                logging.info(
                                    f"Saved summary using fallback for: {title}"
                                )
                        else:
                            logging.warning(
                                f"Failed to extract content even with fallback for: {title}"
                            )
                else:
                    logging.warning(
                        f"Unsupported content type for {link}: {content_type}"
                    )
            else:
                logging.warning(f"No content fetched for {link}")

            # Save comments link
            save_content_to_file(comments_link, directory, "comments_link.txt")
            if debug:
                logging.info(f"Saved comments link for: {title}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape articles from a CSV of links.")
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="output/hacker_news_links.csv",
        help="Path to the CSV file containing the links.",
    )
    parser.add_argument(
        "--output_dir", default="output/scrape", help="Directory to save the output."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--max_links", type=int, help="Maximum number of links to process."
    )
    parser.add_argument("--config", type=str, help="Path to the config JSON file.")
    args = parser.parse_args()

    # Load configuration from JSON file if provided
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            config = json.load(args.config)
        max_links = config.get("max_links", args.max_links)
        base_url = config.get("ollama_base_url", "http://localhost:11434/v1")
        timeout = config.get("ollama_timeout", 600)
        apikey = config.get("ollama_api_key", "ollama")
        model = config.get("ollama_model", "llama3")
        client = setup_ollama_client(base_url, timeout, apikey)
    else:
        max_links = args.max_links

    main(args.csv_file, args.output_dir, args.debug, max_links, model)
