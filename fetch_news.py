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
{json.dumps([{{"url": a['url'], "image": a['image']}} for a in original_articles], indent=2)}

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


def fetch_news():
    """Main function — fetch real news, simplify, review, save."""

    news_api_key = os.environ["NEWS_API_KEY"]
    gemini_api_key = os.environ["GEMINI_API_KEY"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    print(f"[{today}] Step 1: Fetching real news from NewsAPI...")
    original_articles = fetch_real_news(news_api_key)

    print(f"[{today}] Step 2: Simplifying with Gemini (Pass 1)...")
    news_data = simplify_with_gemini(original_articles, model, today)

    print(f"[{today}] Step 3: Reviewing with Gemini (Pass 2)...")
    news_data = review_with_gemini(news_data, original_articles, model, today)

    print(f"[{today}] Step 4: Validating final output...")
    news_data = validate(news_data, original_articles)

    news_data["generated_at"] = datetime.now(timezone.utc).isoformat()

    with open("news.json", "w", encoding="utf-8") as f:
        json.dump(news_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! news.json written with {len(news_data['articles'])} articles.")
    for idx, a in enumerate(news_data["articles"]):
        print(f"   [{idx+1}] [{a['category']}] {a['title']}")
        print(f"         🔗 {a.get('url', 'NO URL')}")
        print(f"         🖼️  {a.get('image', 'NO IMAGE')}")


if __name__ == "__main__":
    fetch_news()
