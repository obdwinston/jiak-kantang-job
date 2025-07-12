import os
import time
import json
import requests
from dotenv import load_dotenv
from datetime import datetime
from supabase import create_client, Client
from bs4 import BeautifulSoup

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")


def get_existing_urls() -> set[str]:
    """Get existing article URLs from Supabase."""

    if not SUPABASE_URL or not SUPABASE_API_KEY:
        print(
            "Warning: SUPABASE_URL or SUPABASE_API_KEY not found in environment variables"
        )
        return set()

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
        response = supabase.table("translations").select("article_url").execute()
        existing_urls = {row["article_url"] for row in response.data}

        print(f"Found {len(existing_urls)} existing URLs from Supabase")
        return existing_urls

    except Exception as e:
        print(f"Error fetching existing URLs: {e}")
        return set()


def scrape_urls() -> list[str]:
    """Scrape latest article URLs from Channel News Asia."""

    base_url = "https://www.channelnewsasia.com"
    latest_news_url = f"{base_url}/latest-news"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        existing_urls = get_existing_urls()

        response = requests.get(latest_news_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        a_tags = soup.find_all("a", class_=lambda x: x and "h6__link" in x)
        url_list = []
        for a in a_tags:
            href = a.get("href")
            if href:
                if not href.startswith("http"):
                    href = base_url + href
                if href not in existing_urls:
                    url_list.append(href)

        print(f"Found {len(url_list)} new URLs from Channel News Asia")
        return url_list

    except Exception as e:
        print(f"Error fetching new URLs: {e}")
        return []


def scrape_article(article_url: str) -> dict:
    """Scrape latest articles from Channel News Asia."""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(article_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # extract published datetime
        div_tag = soup.find("div", class_=lambda x: x and "article-publish" in x)
        published_at = None
        if div_tag:
            text = div_tag.text.strip().split("\n")[0]
            try:
                published_at = datetime.strptime(text, "%d %b %Y %I:%M%p").isoformat()
            except ValueError:
                published_at = text

        # extract image URL
        img_tag = soup.find("img", class_="image")
        image_url = None
        if img_tag:
            image_url = img_tag.get("src")

        # extract article title
        h1_tag = soup.find("h1", class_=lambda x: x and "h1--page-title" in x)
        title = None
        if h1_tag:
            title = h1_tag.text.strip()

        # extract article content
        p_tags = soup.find_all("p")
        content_list = []
        for p in p_tags:
            text = p.text.strip()
            if text and not text.startswith("Advertisement") and len(text) > 10:
                content_list.append(text)
        content = " ".join(content_list)

        return {
            "article_url": article_url,
            "image_url": image_url,
            "title": title,
            "category": article_url.split("/")[3],
            "content": content,
            "published_at": published_at,
        }

    except Exception as e:
        print(f"Error scraping {article_url}: {e}")
        return None


if __name__ == "__main__":
    urls = scrape_urls()

    articles = []
    for i, url in enumerate(urls):
        print(f"Scraping article {i + 1}/{len(urls)}: {url}")
        article = scrape_article(url)
        articles.append(article)
        if i < len(urls) - 1:
            time.sleep(1)  # add 1-second delay for courtesy

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"{len(articles)} articles saved to data.json")
