import os
import json
import requests
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# STRUCTURE ENFORCEMENT (Forces the AI to limit its text length)
# ---------------------------------------------------------------------------
class StructuredArticle(BaseModel):
    title: str = Field(description="The rewritten punchy headline.")
    summary: str = Field(description="A brief summary. Exactly 2 to 3 short sentences. Strict maximum of 35 words total.")
    category: str = Field(description="Must be one of: AI, Hardware, Software, Security, Science, Business, Policy, Startups")
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

class RankedFeed(BaseModel):
    generated_at: str
    articles: List[StructuredArticle]


# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------
def fetch_real_news(news_api_key):
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "category": "technology",
        "language": "en",
        "pageSize": 10,
        "apiKey": news_api_key
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    if data.get("status") != "ok":
        raise ValueError(f"NewsAPI error: {data.get('message')}")
        
    articles = []
    for item in data.get("articles", []):
        if item.get("title") and item.get("title") != "[Removed]":
            articles.append({
                "title": item.get("title", ""),
                "description": item.get("description", "") or "",
                "content": item.get("content", "") or "",
                "source": item.get("source", {}).get("name", "Unknown"),
                "url": item.get("url", ""),
                "image": item.get("urlToImage", "") or "",
                "published_at": item.get("publishedAt", "")
            })
    if not articles:
        raise ValueError("No articles returned from NewsAPI")
    print(f"   Fetched {len(articles)} real articles from NewsAPI")
    return articles[:8]

def process_articles(articles, client, today):
    """Pass 1 — simplify, shorten, and rewrite articles with strict Pydantic enforcement."""
    now_iso = datetime.now(timezone.utc).isoformat()
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
- 'summary' MUST be only 2-3 sentences and UNDER 35 words total. No long fluff paragraphs.
- 'key_points' MUST contain exactly 3 strings, each under 8 words.
- 'impact' MUST be exactly 1 sentence under 20 words.
- Keep original source names, URLs, and image URLs unchanged.

Today's Date: {today}
Articles to process:
{articles_text}"""

    print("   Pass 1: Simplifying with GPT-4o mini (Structured Outputs Mode)...")
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=TechBriefingEdition,
        temperature=0.1
    )
    return json.loads(completion.choices[0].message.content)

def rank_articles(new_articles, all_batches, client, today):
    """Pass 2 — rank and maintain the feed using Structured Outputs."""
    old_articles = []
    cutoff = datetime.now(timezone.utc).timestamp() - (48 * 3600)
    
    for batch in all_batches:
        try:
            batch_time = datetime.fromisoformat(batch["fetched_at"]).timestamp()
            if batch_time >= cutoff:
                for a in batch["articles"]:
                    a["fetched_at"] = batch["fetched_at"]
                    old_articles.append(a)
        except Exception as e:
            print(f"   Warning: Skipping batch due to error: {e}")
            pass
            
    all_articles = new_articles + old_articles
    now_iso = datetime.now(timezone.utc).isoformat()
    articles_text = ""
    for i, a in enumerate(all_articles):
        is_new = i < len(new_articles)
        articles_text += f"""
Article {i+1}:
Title: {a['title']}
Category: {a['category']}
Source: {a['source']}
URL: {a.get('url','')}
Image: {a.get('image','')}
Summary: {a['summary']}
Key Points: {a.get('key_points',[])}
Impact: {a.get('impact','')}
Is New: {'YES' if is_new else 'NO'}
Published At: {a.get('published_at', '')}
Fetched At: {a.get('fetched_at', now_iso)}
---"""

    prompt = f"""You are a news feed curator. Organize and sort these {len(all_articles)} articles for our tech news homepage. 
Rules:
- Order them by relevance, tech importance, and freshness.
- Prioritize brand new breaking developments over old items.
- Ensure variety; avoid grouping too many articles of the same category back-to-back.
- Maintain original lengths and formatting exactly as passed in.

Current time: {now_iso}
Articles to sort:
{articles_text}"""

    print("   Pass 2: Ranking feed with GPT-4o mini (Structured Outputs Mode)...")
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=RankedFeed,
        temperature=0.2
    )
    return json.loads(completion.choices[0].message.content)

def validate(news_data, original_articles):
    valid_categories = {"AI", "Hardware", "Software", "Security", "Science", "Business", "Policy", "Startups"}
    for i, article in enumerate(news_data["articles"]):
        if not article.get("url") and i < len(original_articles):
            article["url"] = original_articles[i]["url"]
        if not article.get("image") and i < len(original_articles):
            article["image"] = original_articles[i]["image"]
        if not article.get("published_at") and i < len(original_articles):
            article["published_at"] = original_articles[i]["published_at"]
        if article.get("category") not in valid_categories:
            article["category"] = "Software"
    return news_data

def update_archive(new_batch, archive_path="archive.json"):
    try:
        with open(archive_path, "r", encoding="utf-8") as f:
            archive = json.load(f)
        print(f"   Loaded existing archive with {len(archive['batches'])} batches")
    except (FileNotFoundError, Exception):
        print("   No or corrupt archive found, creating a fresh instance")
        archive = {"batches": []}

    archive["batches"].insert(0, new_batch)
    print(f"   Added new batch: {new_batch['label']}")

    cutoff = datetime.now(timezone.utc).timestamp() - (7 * 24 * 3600)
    original_count = len(archive["batches"])
    archive["batches"] = [b for b in archive["batches"] if datetime.fromisoformat(b["fetched_at"]).timestamp() >= cutoff]
    
    removed = original_count - len(archive["batches"])
    if removed > 0:
        print(f"   Removed {removed} old batches (>7 days aged)")

    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Archive saved: {len(archive['batches'])} total batches")
    return archive

def fetch_news():
    news_api_key = os.environ.get("NEWS_API_KEY")
    github_token = os.environ.get("GITHUB_TOKEN")
    if not news_api_key or not github_token:
        raise ValueError("Environment keys NEWS_API_KEY or GITHUB_TOKEN are missing.")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now_iso = datetime.now(timezone.utc).isoformat()

    print(f"\n{'='*60}\nAI·Brief News Fetch — {today}\n{'='*60}\n")

    client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=github_token)

    print("[Step 1] Fetching real news from NewsAPI...")
    original_articles = fetch_real_news(news_api_key)

    print("\n[Step 2] Processing with GPT-4o mini (Pass 1)...")
    news_data = process_articles(original_articles, client, today)
    news_data = validate(news_data, original_articles)
    news_data["generated_at"] = now_iso

    print("\n[Step 3] Saving news.json...")
    with open("news.json", "w", encoding="utf-8") as f:
        json.dump(news_data, f, indent=2, ensure_ascii=False)
    print(f"   ✅ news.json saved with {len(news_data['articles'])} articles")

    print("\n[Step 4] Updating archive...")
    new_batch = {
        "id": now_iso,
        "fetched_at": now_iso,
        "label": datetime.now(timezone.utc).strftime("%b %d, %Y — %I:%M %p UTC"),
        "articles": news_data["articles"]
    }
    archive = update_archive(new_batch)

    print("\n[Step 5] Ranking feed with GPT-4o mini (Pass 2)...")
    try:
        feed_data = rank_articles(news_data["articles"], archive["batches"], client, today)
        with open("feed.json", "w", encoding="utf-8") as f:
            json.dump(feed_data, f, indent=2, ensure_ascii=False)
        print(f"   ✅ feed.json saved with {len(feed_data['articles'])} articles")
    except Exception as e:
        print(f"   ⚠️ Warning: Feed ranking failed: {e}. Defaulting layout.")

    print(f"\n{'='*60}\n✅ Fetch Complete!\n{'='*60}\n")

if __name__ == "__main__":
    try:
        fetch_news()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        exit(1)
