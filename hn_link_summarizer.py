import csv
import os
import requests
from newspaper import Article
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import argparse
import logging

# Configure logging
logging.basicConfig(filename='scraper.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to extract article content using newspaper3k
def extract_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logging.error(f"Failed to extract article content from {url}: {e}")
        return None

# Function to save content to a file in the specified directory
def save_content_to_file(content, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

# Function to get the raw HTML content
def fetch_raw_html(url, session):
    try:
        response = session.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch raw HTML content from {url}: {e}")
        return None

# Main function to read the CSV and process each link
def main(csv_file, output_dir="output/scrape"):
    with open(csv_file, newline='', encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        # Check if the expected headers are present
        expected_headers = {'Rank', 'hn_id', 'title', 'link', 'score', 'comments', 'comments_link'}
        if not expected_headers.issubset(reader.fieldnames):
            logging.error(f"Expected headers {expected_headers} not found in CSV. Found headers: {reader.fieldnames}")
            return

        # Initialize session with retry strategy
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))

        for row in reader:
            rank = row['Rank']
            hn_id = row['hn_id']
            title = row['title']
            link = row['link']
            comments_link = row['comments_link']
            
            logging.info(f"Processing: {title} ({link})")

            # Create directory for each hn_id within the output directory
            directory = os.path.join(output_dir, hn_id)

            # Save raw HTML content
            raw_html = fetch_raw_html(link, session)
            if raw_html:
                save_content_to_file(raw_html, directory, "raw_html.html")
                logging.info(f"Saved raw HTML for: {title}")

            # Extract and save article content
            content = extract_article_content(link)
            if content:
                save_content_to_file(content, directory, "extracted_article.txt")
                logging.info(f"Saved extracted article for: {title}")
            else:
                logging.warning(f"Failed to extract content for: {title}")

            # Save comments link
            save_content_to_file(comments_link, directory, "comments_link.txt")
            logging.info(f"Saved comments link for: {title}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape articles from a CSV of links.")
    parser.add_argument("csv_file", nargs='?', default="output/hacker_news_links.csv", help="Path to the CSV file containing the links.")
    parser.add_argument("--output_dir", default="output/scrape", help="Directory to save the output.")
    args = parser.parse_args()

    main(args.csv_file, args.output_dir)
