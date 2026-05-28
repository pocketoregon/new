import os
import json
import requests
import google.generativeai as genai
from datetime import datetime, timezone

def fetch_real_news(news_api_key):
    """Fetch real tech news headlines from NewsAPI."""

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
        raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")

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


def simplify_with_gemini(articles, model, today):
    """Pass 1 — Gemini simplifies the real news into easy English."""

    articles_text = ""
    for i, article in enumerate(articles):
        articles_text += f"""
Article {i+1}:
Title: {article['title']}
Source: {article['source']}
URL: {article['url']}
Image: {article['image']}
Description: {article['description']}
Content: {article['content'][:300] if article['content'] else 'No content available'}
---"""

    prompt = f"""You are a tech news simplifier. I will give you {len(articles)} real tech news articles. Rewrite them in simple easy English.

WRITING RULES:
- Use simple everyday words. No complex English.
- Keep sentences short. Maximum 15 words per sentence.
- Be direct. Say what happened. No fluff.
- Write like texting a smart friend who is not a tech expert.
- Summary must be 2-3 short sentences only.
- Key points must be under 8 words each.
- Impact must be one simple sentence under 20 words.
- Keep the EXACT source name, URL and Image URL as given. Do not change or invent them.
- Category must be ONE of: AI, Hardware, Software, Security, Science, Business, Policy, Startups

Here are the real articles:
{articles_text}

Respond with ONLY a valid JSON object. No markdown, no extra text.

JSON structure:
{{
  "date": "{today}",
  "generated_at": "{datetime.now(timezone.utc).isoformat()}",
  "edition": "Daily Tech Briefing",
  "articles": [
    {{
      "title": "simplified short headline under 12 words",
      "summary": "2-3 short simple sentences in easy English",
      "category": "AI or Hardware or Software or Security or Science or Business or Policy or Startups",
      "source": "exact source name as given",
      "url": "exact URL as given, do not change",
      "image": "exact image URL as given, do not change",
      "key_points": ["under 8 words", "under 8 words", "under 8 words"],
      "impact": "one simple sentence under 20 words"
    }}
  ]
}}

Process all {len(articles)} articles. Most important story first."""

    print("   Pass 1: Simplifying articles...")
    response = model.generate_content(prompt)
    raw = response.text.strip()

    if raw.startswith("```"):
        lines = [l for l in raw.split("\n") if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    return json.loads(raw)


def review_with_gemini(news_data, original_articles, model, today):
    """Pass 2 — Gemini reviews and improves its own output."""

    review_prompt = f"""You are a strict editor reviewing a tech news JSON file.
A junior AI wrote this JSON. Your job is to review and fix any issues.

CHECK AND FIX THESE THINGS:
1. Are summaries truly simple? Max 15 words per sentence. Fix any complex words.
2. Are key points short and clear? Under 8 words each. Fix any that are too long.
3. Is the impact sentence simple and under 20 words? Fix if not.
4. Are URLs real and unchanged? Every article must have a url field with a real link.
5. Are image URLs real and unchanged? Every article must have an image field.
6. Are titles short and simple? Under 12 words. Fix any that are too long.
7. Does every article have all fields: title, summary, category, source, url, image, key_points, impact?

Here is the JSON to review:
{json.dumps(news_data, indent=2)}

Here are the original URLs and images you must preserve exactly:
{json.dumps([{"url": a["url"], "image": a["image"]} for a in original_articles], indent=2)}

Rules for your response:
- Fix all problems you find
- Keep all URLs and image URLs exactly as given above
- Make sure every article has both url and image fields
- Respond with ONLY the fixed valid JSON object. No markdown, no explanation.

The JSON must keep this exact structure:
{{
  "date": "{today}",
  "generated_at": "{datetime.now(timezone.utc).isoformat()}",
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

    print("   Pass 2: Reviewing and improving output...")
    response = model.generate_content(review_prompt)
    raw = response.text.strip()

    if raw.startswith("```"):
        lines = [l for l in raw.split("\n") if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    return json.loads(raw)


def validate(news_data, original_articles):
    """Validate final JSON and restore any missing URLs or images."""

    required_keys = {"date", "generated_at", "edition", "articles"}
    missing = required_keys - set(news_data.keys())
    if missing:
        raise ValueError(f"Missing top-level keys: {missing}")

    if not isinstance(news_data["articles"], list) or len(news_data["articles"]) == 0:
        raise ValueError("articles must be a non-empty list")

    article_required = {"title", "summary", "category", "source", "url", "image", "key_points", "impact"}
    valid_categories = {"AI", "Hardware", "Software", "Security", "Science", "Business", "Policy", "Startups"}

    for i, article in enumerate(news_data["articles"]):
        missing_fields = article_required - set(article.keys())
        if missing_fields:
            raise ValueError(f"Article {i} missing fields: {missing_fields}")

        # Restore URL if Gemini dropped it
        if not article.get("url") and i < len(original_articles):
            article["url"] = original_articles[i]["url"]
            print(f"   ⚠️  Restored missing URL for article {i+1}")

        # Restore image if Gemini dropped it
        if not article.get("image") and i < len(original_articles):
            article["image"] = original_articles[i]["image"]
            print(f"   ⚠️  Restored missing image for article {i+1}")

        # Fix invalid category
        if article.get("category") not in valid_categories:
            article["category"] = "Software"

    return news_data


def rank_with_gemini(new_articles, recent_batches, model, today):
    """Pass 3 — Gemini ranks new + old articles together for the feed."""

    # Build old articles list from recent batches (last 48hrs)
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
Summary: {a['summary']}
Category: {a['category']}
Source: {a['source']}
URL: {a.get('url','')}
Image: {a.get('image','')}
Is New: {'YES' if is_new else 'NO'}
Fetched At: {a.get('fetched_at', now_iso)}
---"""

    prompt = f"""You are a news feed curator AI. You have {len(all_articles)} articles — {len(new_articles)} new and {len(old_articles)} older ones.

Your job: rank all of them for a news feed. Pick the best 30 (or all if less than 30).

For each article decide:
- relevance_score: 0-100 (how important/interesting is this right now)
- is_new: true if fetched in last hour, false otherwise
- is_developing: true if this story is likely still evolving
- age_label: human readable like "Just now", "2 hours ago", "Yesterday"

Rules:
- New articles generally rank higher but not always
- A developing old story can outrank a boring new one
- Variety matters — don't put 5 hardware stories in a row
- Keep exact URL and image as given
- Respond with ONLY valid JSON, no markdown

Here are the articles:
{articles_text}

Current time: {now_iso}

JSON structure:
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

    print("   Pass 3: Ranking articles for feed...")
    response = model.generate_content(prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        lines = [l for l in raw.split("\n") if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    return json.loads(raw)


def update_archive(new_batch, archive_path="archive.json"):
    """Append new batch to archive and clean entries older than 7 days."""

    try:
        with open(archive_path, "r", encoding="utf-8") as f:
            archive = json.load(f)
    except:
        archive = {"batches": []}

    # Add new batch at top
    archive["batches"].insert(0, new_batch)

    # Remove batches older than 7 days
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
    """Main function — fetch, simplify, review, rank, archive, save."""

    news_api_key = os.environ["NEWS_API_KEY"]
    gemini_api_key = os.environ["GEMINI_API_KEY"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now_iso = datetime.now(timezone.utc).isoformat()

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    # Step 1 — Get real news
    print(f"[{today}] Step 1: Fetching real news from NewsAPI...")
    original_articles = fetch_real_news(news_api_key)

    # Step 2 — Simplify (Pass 1)
    print(f"[{today}] Step 2: Simplifying with Gemini (Pass 1)...")
    news_data = simplify_with_gemini(original_articles, model, today)

    import time; time.sleep(30)
    # Step 3 — Review (Pass 2)
    print(f"[{today}] Step 3: Reviewing with Gemini (Pass 2)...")
    news_data = review_with_gemini(news_data, original_articles, model, today)

    # Step 4 — Validate
    print(f"[{today}] Step 4: Validating...")
    news_data = validate(news_data, original_articles)
    news_data["generated_at"] = now_iso

    # Step 5 — Save news.json (NOW section)
    with open("news.json", "w", encoding="utf-8") as f:
        json.dump(news_data, f, indent=2, ensure_ascii=False)
    print(f"   news.json saved with {len(news_data['articles'])} articles")

    # Step 6 — Build batch and update archive.json (ALL section)
    print(f"[{today}] Step 5: Updating archive...")
    new_batch = {
        "id": now_iso,
        "fetched_at": now_iso,
        "label": datetime.now(timezone.utc).strftime("%b %d, %Y — %I:%M %p UTC"),
        "articles": news_data["articles"]
    }
    archive = update_archive(new_batch)

    import time; time.sleep(30)
    # Step 7 — Rank for feed.json (FEED section)
    print(f"[{today}] Step 6: Ranking feed with Gemini (Pass 3)...")
    try:
        feed_data = rank_with_gemini(
            news_data["articles"],
            archive["batches"][1:],  # exclude current batch
            model,
            today
        )
        with open("feed.json", "w", encoding="utf-8") as f:
            json.dump(feed_data, f, indent=2, ensure_ascii=False)
        print(f"   feed.json saved with {len(feed_data['articles'])} articles")
    except Exception as e:
        print(f"   ⚠️ Feed ranking failed: {e} — skipping feed.json update")

    print(f"\n✅ Done!")
    for idx, a in enumerate(news_data["articles"]):
        print(f"   [{idx+1}] [{a['category']}] {a['title']}")


if __name__ == "__main__":
    fetch_news()
