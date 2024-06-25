# Hacker News Link Summarizer
## Overview
This project is designed to scrape and summarize links from Hacker News, focusing on stock-related content. It utilizes Python for the main functionality, including web scraping, data processing, and summarization of articles. The project also includes functionality to process stock data and generate markdown reports.

## Features
Hacker News Scraper: Scrapes links from Hacker News and saves them to a CSV file.
Link Summarizer: Summarizes the content of scraped links using the Ollama API and saves the summaries in a markdown file.
Stock Data Processor: Processes stock data for specified symbols and generates a markdown report with historical data and technical indicators.

## Requirements
Before running the project, ensure you have Python installed on your system. Then, install the required Python packages using the following command:
```shell
pip install -r requirements.txt
```

## Configuration
The project can be configured using the config.json file. Here you can specify parameters such as the period for stock data, stock symbols of interest, and output settings.

Example `config.json`:
```json
{
    "recent_period": "1y",
    "stock_symbols": ["AAPL", "AMZN", "NFLX", "NVDA"],
    "pages": 2,
    "output_folder": "output",
    "csv_filename": "hacker_news_links.csv",
    "markdown_filename": "hacker_news_links.md",
    "max_links": 100
}
```

## Usage
To scrape Hacker News links, run: `./scrape_hn_links.sh`

To summarize the scraped links, run: `./summarize_hn_links.sh`

To process stock data and generate a markdown report, execute the stock_data_processor.py script: `python3 stock_data_processor.py`

## Output
The output will be saved in the output directory, including the CSV file with scraped links (hacker_news_links.csv), the markdown file with summaries (hacker_news_links.md), and the markdown report for stock data.

## Contributing
Contributions to this project are welcome. Please ensure you follow the coding standards and write tests for new features.

## License
This project is open-source and available under the MIT License