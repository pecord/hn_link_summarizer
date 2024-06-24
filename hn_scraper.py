import json
import logging
import os
import time
from typing import Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_file: str) -> Dict:
    """
    Loads configuration from a JSON file.

    Args:
        config_file (str): Path to the JSON configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_file, "r", encoding="utf-8") as file:
        config = json.load(file)
    return config


def fetch_page(
    url: str, retries: int = 3, backoff_factor: float = 0.3, timeout: int = 10
) -> requests.Response:
    """
    Fetches a web page with retries and exponential backoff in case of failures.

    Args:
        url (str): The URL of the web page to fetch.
        retries (int): Number of retries before giving up.
        backoff_factor (float): Factor by which the delay increases after each retry.
        timeout (int): Timeout for the web request in seconds.

    Returns:
        requests.Response: The response object.

    Raises:
        requests.RequestException: If the request fails after all retries.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logging.error(f"Request failed for {url}: {e}")
            if attempt < retries - 1:
                sleep_time = backoff_factor * (2**attempt)
                logging.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                raise


def get_hacker_news_links(pages: int = 4, max_links: int = 100) -> List[Dict[str, str]]:
    """
    Scrapes Hacker News for the top links, including title, URL, points, and comments.

    Args:
        pages (int): Number of pages to scrape. Each page contains approximately 30 links.
        max_links (int): Maximum number of links to scrape.

    Returns:
        list: A list of dictionaries containing HN ID, title, link, points, comments, and comment link for the top links.
    """
    links: List[Dict[str, str]] = []
    try:
        for page in range(1, pages + 1):
            url = f"https://news.ycombinator.com/?p={page}"
            response = fetch_page(url)
            logging.info(f"Successfully fetched page {page}")

            soup = BeautifulSoup(response.text, "html.parser")
            items = soup.select(".athing")

            for item in items:
                title_element = item.select_one(".titleline a")
                subtext_element = item.find_next_sibling("tr").select_one(".subtext")

                if title_element and subtext_element:
                    title = title_element.get_text()
                    link = title_element["href"]
                    hn_id = item["id"]

                    # Validate URL
                    if not link.startswith("http"):
                        link = f"https://news.ycombinator.com/{link}"

                    score_element = subtext_element.select_one(".score")
                    score = score_element.get_text() if score_element else "0 points"

                    comments_element = subtext_element.select("a")[-1]
                    comments_text = (
                        comments_element.get_text()
                        if comments_element
                        else "0 comments"
                    )
                    comments_link = (
                        f"https://news.ycombinator.com/{comments_element['href']}"
                        if comments_element
                        else "#"
                    )
                    if "comment" not in comments_text:
                        comments_text = "0 comments"
                        comments_link = "#"

                    links.append(
                        {
                            "hn_id": hn_id,
                            "title": title,
                            "link": link,
                            "score": score,
                            "comments": comments_text,
                            "comments_link": comments_link,
                        }
                    )

                if len(links) >= max_links:
                    logging.info(f"Reached {max_links} links, stopping.")
                    return links[:max_links]
        logging.info(f"Reached page limit of {pages}, stopping.")
    except Exception as e:
        logging.error(f"An error occurred while scraping: {e}")

    logging.info(f"Finished scraping with {len(links)} links.")
    return links[:max_links]


def generate_markdown_table(links: List[Dict[str, str]]) -> str:
    """
    Generates a markdown table from the list of links.

    Args:
        links (list): A list of dictionaries containing HN ID, title, link, points, comments, and comment link.

    Returns:
        str: A string representing the markdown table.
    """
    markdown_list = [
        "| # | HN ID | Title & Link | Points | Comments |",
        "|---|-------|--------------|--------|----------|",
    ]
    for idx, link in enumerate(links, 1):
        title_link = f"[{link['title']}]({link['link']})"
        comments_link = f"[{link['comments']}]({link['comments_link']})"
        markdown_list.append(
            f"| {idx} | {link['hn_id']} | {title_link} | {link['score']} | {comments_link} |"
        )
    return "\n".join(markdown_list)


def save_as_csv(links: List[Dict[str, str]], filepath: str) -> None:
    """
    Saves the list of links as a CSV file.

    Args:
        links (list): A list of dictionaries containing HN ID, title, link, points, comments, and comment link.
        filepath (str): The path to the file where the CSV will be saved.
    """
    df = pd.DataFrame(links)
    df.index += 1  # Start index from 1
    df.to_csv(filepath, index_label="Rank", encoding="utf-8")
    logging.info(f"Saved CSV to {filepath}")


def save_as_markdown(links: List[Dict[str, str]], filepath: str) -> None:
    """
    Saves the list of links as a markdown file.

    Args:
        links (list): A list of dictionaries containing HN ID, title, link, points, comments, and comment link.
        filepath (str): The path to the file where the markdown will be saved.
    """
    markdown_table = generate_markdown_table(links)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_table)
    logging.info(f"Saved markdown to {filepath}")


def main(config_file: str, override_max_links: int = None) -> None:
    """
    Main function to scrape Hacker News, save the links as CSV and markdown files, and print the markdown table.

    Args:
        config_file (str): Path to the JSON configuration file.
        override_max_links (int, optional): Overrides the max_links setting from the config file.
    """
    config = load_config(config_file)

    pages = config.get("pages", 4)
    output_folder = config.get("output_folder", "output")
    csv_filename = config.get("csv_filename", "hacker_news_links.csv")
    markdown_filename = config.get("markdown_filename", "hacker_news_links.md")
    max_links = (
        override_max_links
        if override_max_links is not None
        else config.get("max_links", 100)
    )

    os.makedirs(output_folder, exist_ok=True)

    links = get_hacker_news_links(pages=pages, max_links=max_links)

    # Save as CSV
    csv_filepath = os.path.join(output_folder, csv_filename)
    save_as_csv(links, csv_filepath)

    # Save as Markdown
    markdown_filepath = os.path.join(output_folder, markdown_filename)
    save_as_markdown(links, markdown_filepath)

    # Print the markdown table to the console
    markdown_table = generate_markdown_table(links)
    print(markdown_table)


if __name__ == "__main__":
    config_path = "config.json"
    main(config_path)
