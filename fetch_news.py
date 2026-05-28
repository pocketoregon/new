import os
import json
import requests
import google.generativeai as genai
from datetime import datetime, timezone

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


def process_with_gemini(articles, model, today):
    """Single pass — simplify, validate and rank all in one call."""
    now_iso = datetime.now(timezone.utc).isoformat()
    articles_text = ""
    for i, article in enumerate(articles):
        articles_text += f"""
Article {i+1}:
Title: {article['title']}
Source: {article['source']}
URL: {article['url']}
Image: {article['image']}
Description: {article['description']}
Content: {article['content'][:300] if article['content'] else ''}
---"""

    prompt = f"""You are a tech news editor. Rewrite these {len(articles)} articles in simple English.

RULES:
- Simple everyday words only
- Max 15 words per sentence
- Summary: 2-3 short sentences
- Key points: exactly 3, under 8 words each
- Impact: one sentence under 20 words
- Keep EXACT source name, URL and image URL unchanged
- Category must be one of: AI, Hardware, Software, Security, Science, Business, Policy, Startups
- Most important story first

Today: {today}

Articles:
{articles_text}

Respond with ONLY valid JSON, no markdown:
{{
  "date": "{today}",
  "generated_at": "{now_iso}",
  "edition": "Daily Tech Briefing",
  "articles": [
    {{
      "title": "string",
      "summary": "string",
      "category": "string",
      "source": "string",
      "url": "string",
      "image": "string",
      "key_points": ["string", "string", "string"],
      "impact": "string"
    }}
  ]
}}"""

    print("   Processing with Gemini (single pass)...")
    response = model.generate_content(prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        lines = [l for l in raw.split("\n") if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    return json.loads(raw)


def rank_with_gemini(new_articles, recent_batches, model, today):
    """Pass 2 — rank new + old articles for the feed."""
    old_articles = []
    cutoff = datetime.now(timezone.utc).timestamp() - (48 * 3600)
    for batch in recent_batches:
        try:
            batch_time = datetime.fromisoformat(batch["fetched_at"]).timestamp()
            if batch_time >= cutoff:
                for a in batch["articles"]:
                    a["fetched_at"] = batch["fetched_at"]
                    old_articles.append(a)
        except:
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
Fetched At: {a.get('fetched_at', now_iso)}
---"""

    prompt = f"""You are a news feed curator. Rank these {len(all_articles)} articles for a feed.

Pick best 30 (or all if less). For each article assign:
- relevance_score: 0-100
- is_new: true if fetched in last hour
- is_developing: true if story still evolving
- age_label: "Just now", "2 hours ago", "Yesterday" etc

Rules:
- New articles rank higher generally
- Developing stories can outrank boring new ones
- Variety — no 5 same category in a row
- Keep exact URL and image unchanged
- Respond ONLY valid JSON no markdown

Articles:
{articles_text}

Current time: {now_iso}

JSON:
{{
  "generated_at": "{now_iso}",
  "articles": [
    {{
      "title": "string",
      "summary": "string",
      "category": "string",
      "source": "string",
      "url": "string",
      "image": "string",
      "key_points": ["string", "string", "string"],
      "impact": "string",
      "relevance_score": 0,
      "is_new": true,
      "is_developing": false,
      "age_label": "string",
      "fetched_at": "string"
    }}
  ]
}}"""

    print("   Ranking articles for feed...")
    response = model.generate_content(prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        lines = [l for l in raw.split("\n") if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    return json.loads(raw)


def validate(news_data, original_articles):
    required_keys = {"date", "generated_at", "edition", "articles"}
    missing = required_keys - set(news_data.keys())
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    if not isinstance(news_data["articles"], list) or len(news_data["articles"]) == 0:
        raise ValueError("articles must be non-empty list")
    article_required = {"title", "summary", "category", "source", "url", "image", "key_points", "impact"}
    valid_categories = {"AI", "Hardware", "Software", "Security", "Science", "Business", "Policy", "Startups"}
    for i, article in enumerate(news_data["articles"]):
        missing_fields = article_required - set(article.keys())
        if missing_fields:
            raise ValueError(f"Article {i} missing: {missing_fields}")
        if not article.get("url") and i < len(original_articles):
            article["url"] = original_articles[i]["url"]
        if not article.get("image") and i < len(original_articles):
            article["image"] = original_articles[i]["image"]
        if article.get("category") not in valid_categories:
            article["category"] = "Software"
    return news_data


def update_archive(new_batch, archive_path="archive.json"):
    try:
        with open(archive_path, "r", encoding="utf-8") as f:
            archive = json.load(f)
    except:
        archive = {"batches": []}
    archive["batches"].insert(0, new_batch)
    cutoff = datetime.now(timezone.utc).timestamp() - (7 * 24 * 3600)
    archive["batches"] = [
        b for b in archive["batches"]
        if datetime.fromisoformat(b["fetched_at"]).timestamp() >= cutoff
    ]
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)
    print(f"   Archive updated: {len(archive['batches'])} batches stored")
    return archive


def fetch_news():
    news_api_key = os.environ["NEWS_API_KEY"]
    gemini_api_key = os.environ["GEMINI_API_KEY"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now_iso = datetime.now(timezone.utc).isoformat()

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    # Step 1 — Fetch real news
    print(f"[{today}] Step 1: Fetching real news from NewsAPI...")
    original_articles = fetch_real_news(news_api_key)

    # Step 2 — Single Gemini pass (simplify + validate)
    print(f"[{today}] Step 2: Processing with Gemini (Pass 1)...")
    news_data = process_with_gemini(original_articles, model, today)
    news_data = validate(news_data, original_articles)
    news_data["generated_at"] = now_iso

    # Step 3 — Save news.json
    with open("news.json", "w", encoding="utf-8") as f:
        json.dump(news_data, f, indent=2, ensure_ascii=False)
    print(f"   news.json saved with {len(news_data['articles'])} articles")

    # Step 4 — Update archive
    print(f"[{today}] Step 3: Updating archive...")
    new_batch = {
        "id": now_iso,
        "fetched_at": now_iso,
        "label": datetime.now(timezone.utc).strftime("%b %d, %Y — %I:%M %p UTC"),
        "articles": news_data["articles"]
    }
    archive = update_archive(new_batch)

    # Step 5 — Rank feed (second Gemini call)
    print(f"[{today}] Step 4: Ranking feed with Gemini (Pass 2)...")
    import time
    time.sleep(20)
    try:
        feed_data = rank_with_gemini(
            news_data["articles"],
            archive["batches"][1:],
            model,
            today
        )
        with open("feed.json", "w", encoding="utf-8") as f:
            json.dump(feed_data, f, indent=2, ensure_ascii=False)
        print(f"   feed.json saved with {len(feed_data['articles'])} articles")
    except Exception as e:
        print(f"   Warning: Feed ranking failed: {e}")

    print(f"\n✅ Done!")
    for idx, a in enumerate(news_data["articles"]):
        print(f"   [{idx+1}] [{a['category']}] {a['title']}")
        print(f"         URL: {a.get('url','NO URL')}")


if __name__ == "__main__":
    fetch_news()
