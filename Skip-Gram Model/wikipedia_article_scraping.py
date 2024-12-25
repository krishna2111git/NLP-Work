import requests
from bs4 import BeautifulSoup

def extract_wikipedia_text(url):
    # Fetch the Wikipedia page
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve page: {response.status_code}")
        return None
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the main content area
    content_div = soup.find('div', class_='mw-parser-output')
    
    if not content_div:
        print("Could not find the article content.")
        return None
    
    # Extract text from paragraphs
    paragraphs = content_div.find_all('p')
    article_text = ' '.join(paragraph.text.strip() for paragraph in paragraphs if paragraph.text.strip())
    
    return article_text

# Example usage
wikipedia_url_list = "https://en.wikipedia.org/wiki/India"
article_text = extract_wikipedia_text(wikipedia_url)

print(type(article_text))
if article_text:
    print("Executing")
    print(article_text[:500])  # Print the first 500 characters
