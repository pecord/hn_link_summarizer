import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def get_hacker_news_links(pages=4):
    """
    Scrapes Hacker News for the top links, including title, URL, points, and comments.
    
    Args:
        pages (int): Number of pages to scrape. Each page contains approximately 30 links.
    
    Returns:
        list: A list of dictionaries containing HN ID, title, link, points, comments, and comment link for the top 100 links.
    """
    links = []
    for page in range(1, pages + 1):
        url = f'https://news.ycombinator.com/?p={page}'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        items = soup.select('.athing')
        for item in items:
            title_element = item.select_one('.titleline a')
            subtext_element = item.find_next_sibling('tr').select_one('.subtext')
            
            if title_element and subtext_element:
                title = title_element.get_text()
                link = title_element['href']
                hn_id = item['id']

                # Validate URL
                if not link.startswith('http'):
                    link = f'https://news.ycombinator.com/{link}'
                
                score_element = subtext_element.select_one('.score')
                score = score_element.get_text() if score_element else '0 points'
                
                comments_element = subtext_element.select('a')[-1]
                comments_text = comments_element.get_text() if comments_element else '0 comments'
                comments_link = f"https://news.ycombinator.com/{comments_element['href']}" if comments_element else '#'
                if 'comment' not in comments_text:
                    comments_text = '0 comments'
                    comments_link = '#'
                
                links.append({
                    'hn_id': hn_id,
                    'title': title,
                    'link': link,
                    'score': score,
                    'comments': comments_text,
                    'comments_link': comments_link
                })

            if len(links) >= 100:
                return links[:100]
    return links[:100]

def generate_markdown_table(links):
    """
    Generates a markdown table from the list of links.
    
    Args:
        links (list): A list of dictionaries containing HN ID, title, link, points, comments, and comment link.
    
    Returns:
        str: A string representing the markdown table.
    """
    markdown_list = [
        "| # | HN ID | Title & Link | Points | Comments |",
        "|---|-------|--------------|--------|----------|"
    ]
    for idx, link in enumerate(links, 1):
        title_link = f"[{link['title']}]({link['link']})"
        comments_link = f"[{link['comments']}]({link['comments_link']})"
        markdown_list.append(f"| {idx} | {link['hn_id']} | {title_link} | {link['score']} | {comments_link} |")
    return '\n'.join(markdown_list)

def save_as_csv(links, filepath):
    """
    Saves the list of links as a CSV file.
    
    Args:
        links (list): A list of dictionaries containing HN ID, title, link, points, comments, and comment link.
        filepath (str): The path to the file where the CSV will be saved.
    """
    df = pd.DataFrame(links)
    df.index += 1  # Start index from 1
    df.to_csv(filepath, index_label='Rank', encoding='utf-8')

def save_as_markdown(links, filepath):
    """
    Saves the list of links as a markdown file.
    
    Args:
        links (list): A list of dictionaries containing HN ID, title, link, points, comments, and comment link.
        filepath (str): The path to the file where the markdown will be saved.
    """
    markdown_table = generate_markdown_table(links)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_table)

def main():
    """
    Main function to scrape Hacker News, save the links as CSV and markdown files, and print the markdown table.
    """
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)
    
    links = get_hacker_news_links()
    
    # Save as CSV
    csv_filepath = os.path.join(output_folder, 'hacker_news_links.csv')
    save_as_csv(links, csv_filepath)
    
    # Save as Markdown
    markdown_filepath = os.path.join(output_folder, 'hacker_news_links.md')
    save_as_markdown(links, markdown_filepath)
    
    # Print the markdown table to the console
    markdown_table = generate_markdown_table(links)
    print(markdown_table)

if __name__ == '__main__':
    main()
