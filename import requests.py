import requests
from bs4 import BeautifulSoup
import json

class HNScraper:
    def __init__(self):
        self.url = 'https://news.ycombinator.com/'
        self.articles = []

    def scrape_hacker_news(self):
        for page in range(1, 4, 1):
            self.url = f'https://news.ycombinator.com/news?p={page}'
            #print(f"Scraping page {self.url}")
            response = requests.get(self.url)
            self.scrape_page(response.text)

    def scrape_page(self, responseText):
        soup = BeautifulSoup(responseText, 'html.parser')

        for item in soup.select('.title .titleline a'):
            if ('from?' in item.get('href')): continue
            title = item.get_text()
            link = f"{item.get('href')}"
            self.articles.append({'title': title, 'url': link})

    def print_output(self):
        print(json.dumps(self.articles, indent=2))

    def run_scraper(self):
        self.scrape_hacker_news()
        self.print_output()

if __name__ == "__main__":
    scraper = HNScraper()
    scraper.run_scraper()