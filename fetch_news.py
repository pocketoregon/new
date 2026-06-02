import os
import json
import requests
import hashlib
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# STRUCTURE ENFORCEMENT (Forces the AI to limit its text length)
# ---------------------------------------------------------------------------
class StructuredArticle(BaseModel):
    title: str = Field(description="The rewritten punchy headline.")
    description: str = Field(description="A brief summary. Exactly 2 to 3 short sentences. Strict maximum of 35 words total.")
    category: str = Field(description="Must be one of: AI, Hardware, Software, Security, Science, Business, Policy, Startups, Gaming")
    source: str = Field(description="Keep exact original source name unchanged.")
    url: str = Field(description="Keep exact original URL unchanged.")
    image: str = Field(description="Keep exact original image URL unchanged.")
    published_at: str = Field(description="Keep exact original published_at timestamp unchanged.")
    key_points: List[str] = Field(description="Exactly 3 key takeaway bullet points. Each point must be strictly under 8 words.")
    impact: str = Field(description="One single impact sentence. Must be strictly under 20 words.")

class TechBriefingEdition(BaseModel):
    date: str
    generated_at: str
    edition: str = "Daily Tech Briefing"
    articles: List[StructuredArticle]

# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------
def generate_id(title, published_at, url=""):
    """Creates a unique fixed fingerprint ID for each article. URL is preferred."""
    raw_str = url if url else f"{title}_{published_at}"
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

def fetch_real_news(news_api_key):
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "category": "technology",
        "language": "en",
        "pageSize": 40,
        "apiKey": news_api_key
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "ok":
        raise ValueError(f"NewsAPI error: {data.get('message')}")

    articles = []
    for item in data.get("articles", []):
        title = item.get("title", "")
        if title and title != "[Removed]":
            articles.append({
                "title": title,
                "description": item.get("description", "") or "",
                "content": item.get("content", "") or "",
                "source": item.get("source", {}).get("name", "Unknown"),
                "url": item.get("url", ""),
                "image": item.get("urlToImage", "") or "",
                "published_at": item.get("publishedAt", "") or ""
            })
    return articles

def load_existing_ids(archive_path="archive.json"):
    """Reads archive to find all previously collected article IDs."""
    existing_ids = set()
    try:
        with open(archive_path, "r", encoding="utf-8") as f:
            archive = json.load(f)
        for batch in archive.get("batches", []):
            for article in batch.get("articles", []):
                if "id" in article:
                    existing_ids.add(article["id"])
    except Exception:
        pass
    return existing_ids

def process_articles(articles, client, today):
    """Rewrite articles with strict Pydantic enforcement."""
    if not articles:
        return {"date": today, "generated_at": datetime.now(timezone.utc).isoformat(), "articles": []}

    articles_text = ""
    for i, article in enumerate(articles):
        articles_text += f"""
Article {i+1}:
Title: {article['title']}
Source: {article['source']}
URL: {article['url']}
Image: {article['image']}
Published: {article['published_at']}
Description: {article['description']}
Content: {article['content'][:300] if article['content'] else ''}
---"""

    prompt = f"""You are an expert tech news copyeditor. Rewrite these {len(articles)} articles into brief, ultra-crisp executive summaries.

CRITICAL CONSTRAINTS:
- Keep everything simple, punchy, and direct.
- 'description' MUST be only 2-3 sentences and UNDER 35 words total. No long fluff paragraphs.
- 'key_points' MUST contain exactly 3 strings, each under 8 words.
- 'impact' MUST be exactly 1 sentence under 20 words.
- Keep original source names, URLs, and image URLs unchanged.

Today's Date: {today}
Articles to process: {articles_text}"""

    print(f"   Processing {len(articles)} fresh unseen articles with GPT-4o mini...")
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=TechBriefingEdition,
            temperature=0.1
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"   ⚠️ GPT processing failed: {e}")
        return {"date": today, "generated_at": datetime.now(timezone.utc).isoformat(), "articles": []}

def validate(news_data, original_articles):
    valid_categories = {"AI", "Hardware", "Software", "Security", "Science", "Business", "Policy", "Startups", "Gaming"}
    for i, article in enumerate(news_data["articles"]):
        if i < len(original_articles):
            if not article.get("url"):
                article["url"] = original_articles[i]["url"]
            if not article.get("image"):
                article["image"] = original_articles[i]["image"]
            if not article.get("published_at"):
                article["published_at"] = original_articles[i]["published_at"]
            if article.get("category") not in valid_categories:
                article["category"] = "Software"
    return news_data

def update_archive(new_batch, archive_path="archive.json"):
    try:
        with open(archive_path, "r", encoding="utf-8") as f:
            archive = json.load(f)
    except (FileNotFoundError, Exception):
        archive = {"batches": []}

    if new_batch["articles"]:
        archive["batches"].insert(0, new_batch)
        print(f"   Added new batch with {len(new_batch['articles'])} unseen articles.")
    else:
        print("   No new articles to append to archive during this cycle.")

    archive["batches"] = archive["batches"][:30]

    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)
    return archive

def update_feed(archive, feed_path="feed.json"):
    """Rebuild feed.json from the most recent batch in the archive."""
    batches = archive.get("batches", [])
    if not batches:
        articles = []
        generated_at = datetime.now(timezone.utc).isoformat()
    else:
        latest = batches[0]
        articles = latest.get("articles", [])
        generated_at = latest.get("fetched_at", datetime.now(timezone.utc).isoformat())

    feed = {
        "generated_at": generated_at,
        "articles": articles
    }
    with open(feed_path, "w", encoding="utf-8") as f:
        json.dump(feed, f, indent=2, ensure_ascii=False)
    print(f"   feed.json updated with {len(articles)} articles from latest batch.")

def generate_rss(articles, feed_path="feed.xml"):
    """Generates a valid RSS 2.0 feed from the latest articles."""
    now_rfc = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S +0000')
    items = ""
    for a in articles:
        title = (a.get("title") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        desc = (a.get("description") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        url = a.get("url") or ""
        pub = a.get("published_at") or now_rfc
        cat = a.get("category") or "Technology"
        source = (a.get("source") or "").replace("&", "&amp;")
        items += f"""
    <item>
      <title>{title}</title>
      <description>{desc}</description>
      <link>{url}</link>
      <guid isPermaLink="true">{url}</guid>
      <pubDate>{pub}</pubDate>
      <category>{cat}</category>
      <source url="{url}">{source}</source>
    </item>"""
    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>AI·Brief — Daily Tech Intelligence</title>
    <link>https://pocketoregon.github.io/new</link>
    <description>Automated AI-powered tech news briefings updated every 3 hours.</description>
    <language>en-us</language>
    <lastBuildDate>{now_rfc}</lastBuildDate>
    <atom:link href="https://pocketoregon.github.io/new/feed.xml" rel="self" type="application/rss+xml"/>
    {items}
  </channel>
</rss>"""
    with open(feed_path, "w", encoding="utf-8") as f:
        f.write(rss)
    print(f"   feed.xml generated with {len(articles)} articles.")

def fetch_news():
    news_api_key = os.environ.get("NEWS_API_KEY")
    github_token = os.environ.get("GITHUB_TOKEN")

    if not news_api_key or not github_token:
        raise ValueError("Environment keys NEWS_API_KEY or GITHUB_TOKEN are missing.")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now_iso = datetime.now(timezone.utc).isoformat()

    print(f"\n{'='*60}\nAI·Brief Pipeline — {today}\n{'='*60}\n")

    client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=github_token)

    print("[Step 1] Fetching fresh candidates from NewsAPI...")
    try:
        raw_candidates = fetch_real_news(news_api_key)
        print(f"   Fetched {len(raw_candidates)} total candidates.")
    except Exception as e:
        print(f"   ❌ NewsAPI fetch failed: {e}")
        exit(1)

    print("\n[Step 2] Filtering out existing duplicate IDs...")
    existing_ids = load_existing_ids()
    unseen_articles = []
    for article in raw_candidates:
        art_id = generate_id(article["title"], article["published_at"], article.get("url", ""))
        if art_id not in existing_ids:
            article["id"] = art_id
            unseen_articles.append(article)

    print(f"   Filtered down to {len(unseen_articles)} brand-new unique articles.")
    unseen_articles = unseen_articles[:15]  # Cap per run to avoid huge GPT prompts
    print(f"   Capped to {len(unseen_articles)} articles for this run.")

    if not unseen_articles:
        print("\n   [Notice] Zero fresh articles to pass forward this hourly iteration.")
        news_data = {"date": today, "generated_at": now_iso, "articles": []}
    else:
        print("\n[Step 3] Rewriting only unseen data...")
        news_data = process_articles(unseen_articles, client, today)
        news_data = validate(news_data, unseen_articles)
        news_data["generated_at"] = now_iso

        for i, article in enumerate(news_data.get("articles", [])):
            if i < len(unseen_articles):
                article["id"] = unseen_articles[i]["id"]

    print("\n[Step 4] Saving current news.json snapshot...")
    with open("news.json", "w", encoding="utf-8") as f:
        json.dump(news_data, f, indent=2, ensure_ascii=False)

    print("\n[Step 5] Synchronizing to archive...")
    new_batch = {
        "id": now_iso,
        "fetched_at": now_iso,
        "label": datetime.now(timezone.utc).strftime("%b %d, %Y — %I:%M %p UTC"),
        "articles": news_data["articles"]
    }
    archive = update_archive(new_batch)

    print("\n[Step 6] Updating feed.json from latest archive batch...")
    update_feed(archive)
    
    print("\n[Step 7] Generating RSS feed.xml...")
    generate_rss(news_data.get("articles", []))

    print(f"\n{'='*60}\nBackend Operations Complete!\n{'='*60}\n")

if __name__ == "__main__":
    try:
        fetch_news()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        exit(1)
