import os
import json
import requests
from openai import OpenAI
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


def call_gpt(client, prompt):
    """Call GitHub Models GPT-4o mini."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = [l for l in raw.split("\n") if not l.strip().startswith("```")]
            raw = "\n".join(lines).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"   ERROR in call_gpt: {e}")
        raise


def process_articles(articles, client, today):
    """Pass 1 — simplify and rewrite articles."""
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

Respond with ONLY valid JSON, no markdown, no extra text:
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
      "published_at": "string",
      "key_points": ["string", "string", "string"],
      "impact": "string"
    }}
  ]
}}"""

    print("   Pass 1: Simplifying with GPT-4o mini...")
    return call_gpt(client, prompt)


def rank_articles(new_articles, all_batches, client, today):
    """Pass 2 — rank new + old articles for the feed."""
    old_articles = []
    cutoff = datetime.now(timezone.utc).timestamp() - (48 * 3600)
    
    # Include ALL batches, not just [1:]
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

    prompt = f"""You are a news feed curator. Rank these {len(all_articles)} articles for a news feed.

Pick best 30 (or all if less than 30). For each article assign:
- relevance_score: 0-100
- is_new: true if fetched in last hour
- is_developing: true if story is still evolving
- age_label: "Just now", "2 hours ago", "Yesterday" etc

Rules:
- New articles rank higher generally
- Developing stories can outrank boring new ones
- Variety — avoid 5 same category in a row
- Keep exact URL, image, and published_at unchanged
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
      "published_at": "string",
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

    print("   Pass 2: Ranking with GPT-4o mini...")
    return call_gpt(client, prompt)


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
        # Preserve original URL and image if missing
        if not article.get("url") and i < len(original_articles):
            article["url"] = original_articles[i]["url"]
        if not article.get("image") and i < len(original_articles):
            article["image"] = original_articles[i]["image"]
        # Preserve published_at
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
    except FileNotFoundError:
        print("   No existing archive found, creating new one")
        archive = {"batches": []}
    except Exception as e:
        print(f"   Error loading archive: {e}, creating new one")
        archive = {"batches": []}
    
    # Add new batch at the beginning
    archive["batches"].insert(0, new_batch)
    print(f"   Added new batch: {new_batch['label']}")
    
    # Clean up old batches (keep 7 days)
    cutoff = datetime.now(timezone.utc).timestamp() - (7 * 24 * 3600)
    original_count = len(archive["batches"])
    archive["batches"] = [
        b for b in archive["batches"]
        if datetime.fromisoformat(b["fetched_at"]).timestamp() >= cutoff
    ]
    removed = original_count - len(archive["batches"])
    if removed > 0:
        print(f"   Removed {removed} old batches (>7 days)")
    
    # Save archive
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Archive saved: {len(archive['batches'])} total batches")
    
    return archive


def fetch_news():
    news_api_key = os.environ.get("NEWS_API_KEY")
    github_token = os.environ.get("GITHUB_TOKEN")
    
    if not news_api_key:
        raise ValueError("NEWS_API_KEY environment variable not set")
    if not github_token:
        raise ValueError("GITHUB_TOKEN environment variable not set")
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now_iso = datetime.now(timezone.utc).isoformat()

    print(f"\n{'='*60}")
    print(f"AI·Brief News Fetch — {today}")
    print(f"{'='*60}\n")

    # Setup GitHub Models client
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=github_token
    )

    # Step 1 — Fetch real news
    print(f"[Step 1] Fetching real news from NewsAPI...")
    try:
        original_articles = fetch_real_news(news_api_key)
    except Exception as e:
        print(f"   ❌ ERROR: Failed to fetch news: {e}")
        raise

    # Step 2 — Simplify with GPT-4o mini
    print(f"\n[Step 2] Processing with GPT-4o mini (Pass 1)...")
    try:
        news_data = process_articles(original_articles, client, today)
        news_data = validate(news_data, original_articles)
        news_data["generated_at"] = now_iso
    except Exception as e:
        print(f"   ❌ ERROR: Failed to process articles: {e}")
        raise

    # Step 3 — Save news.json
    print(f"\n[Step 3] Saving news.json...")
    try:
        with open("news.json", "w", encoding="utf-8") as f:
            json.dump(news_data, f, indent=2, ensure_ascii=False)
        print(f"   ✅ news.json saved with {len(news_data['articles'])} articles")
    except Exception as e:
        print(f"   ❌ ERROR: Failed to save news.json: {e}")
        raise

    # Step 4 — Update archive
    print(f"\n[Step 4] Updating archive...")
    try:
        new_batch = {
            "id": now_iso,
            "fetched_at": now_iso,
            "label": datetime.now(timezone.utc).strftime("%b %d, %Y — %I:%M %p UTC"),
            "articles": news_data["articles"]
        }
        archive = update_archive(new_batch)
    except Exception as e:
        print(f"   ❌ ERROR: Failed to update archive: {e}")
        raise

    # Step 5 — Rank feed
    print(f"\n[Step 5] Ranking feed with GPT-4o mini (Pass 2)...")
    try:
        feed_data = rank_articles(
            news_data["articles"],
            archive["batches"],  # FIXED: Use all batches, not [1:]
            client,
            today
        )
        with open("feed.json", "w", encoding="utf-8") as f:
            json.dump(feed_data, f, indent=2, ensure_ascii=False)
        print(f"   ✅ feed.json saved with {len(feed_data['articles'])} articles")
    except Exception as e:
        print(f"   ⚠️  Warning: Feed ranking failed: {e}")
        print(f"   Continuing without feed update...")

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Fetch Complete!")
    print(f"{'='*60}")
    print(f"Articles processed: {len(news_data['articles'])}")
    print(f"Archive batches: {len(archive['batches'])}")
    print(f"\nArticles:")
    for idx, a in enumerate(news_data["articles"]):
        print(f"   [{idx+1}] [{a['category']}] {a['title'][:60]}...")
    print(f"\n")


if __name__ == "__main__":
    try:
        fetch_news()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
