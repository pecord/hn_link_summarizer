import argparse
import glob
import json
import os
from openai import OpenAI
# Placeholder for Ollama summarization integration
# import ollama



def find_summary_files(base_path):
    return glob.glob(f"{base_path}/*/summary.md")

def extract_summary_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return content

def setup_ollama_client(
    base_url: str = "http://localhost:11434/v1",
    timeout: int = 600,
    api_key: str = "ollama",
) -> OpenAI:
    """
    Sets up the Ollama client with the specified parameters.
    """
    return OpenAI(base_url=base_url, timeout=timeout, api_key=api_key)

def summarize_with_ollama(content: str, model: str, client: OpenAI) -> str:
    """
    Summarizes the article content using Ollama.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """
### System Prompt

**Objective:** Summarize the summaries to extract and tally company mentions, technology highlights, and overarching investment insights. The summary should be divided into sections: Company Mentions, Technology Highlights, and Investment Insights. Additionally, tally the frequency of mentions for companies and categorize them as positive or negative. Highlight technologies and summarize key investment insights observed across multiple articles.

**Output Structure:**

1. **Company Mentions:**
   - Tally and categorize the frequency of company mentions as positive or negative.
   - Provide a brief description for each mention.

2. **Technology Highlights:**
   - List the technologies highlighted across the summaries.
   - Provide a brief description for each mention.

3. **Investment Insights:**
   - Summarize overarching investment insights seen across multiple articles.
""",
            },
            {"role": "user", "content": content},
        ],
    )
    return response.choices[0].message.content

def main():
    # Step 1: Initialize defaults
    defaults = {
        "csv_file": "output/hacker_news_links.csv",
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
    parser.add_argument("csv_file", nargs="?", help="Path to the CSV file containing the links.")
    parser.add_argument("--output_dir", help="Directory to save the output.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--max_links", type=int, help="Maximum number of links to process.")
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

    # Set up Ollama client
    client = setup_ollama_client()

    summary_files = find_summary_files(defaults["output_dir"])
    # print the number of summary files found 
    print(f"Found {len(summary_files)} summary files.")
    combined_summaries = " ".join([extract_summary_from_file(file) for file in summary_files])
    # print the combined summaries
    #print("combined_summaries: " + combined_summaries)
    final_summary = summarize_with_ollama(combined_summaries, defaults["ollama_model"], client)
    print(final_summary)

if __name__ == "__main__":
    main()