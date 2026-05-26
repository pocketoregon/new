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
                "published_at": item.get("publishedAt", "")
            })
    
    if not articles:
        raise ValueError("No articles returned from NewsAPI")
    
    print(f"   Fetched {len(articles)} real articles from NewsAPI")
    return articles[:8]


def simplify_with_gemini(articles, gemini_api_key):
    """Use Gemini to simplify the real news into easy English."""
    
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    # Build article list for the prompt
    articles_text = ""
    for i, article in enumerate(articles):
        articles_text += f"""
Article {i+1}:
Title: {article['title']}
Source: {article['source']}
URL: {article['url']}
Description: {article['description']}
Content: {article['content'][:300] if article['content'] else 'No content available'}
---"""

    prompt = f"""You are a tech news simplifier. I will give you {len(articles)} real tech news articles. Your job is to rewrite them in simple, easy English.

WRITING RULES:
- Use simple everyday words. No complex English.
- Keep sentences short. Maximum 15 words per sentence.
- Be direct. Say what happened. No fluff.
- Write like texting a smart friend who is not a tech expert.
- Summary must be 2-3 short sentences only.
- Key points must be under 8 words each.
- Impact must be one simple sentence under 20 words.
- Keep the EXACT source name and URL as given. Do not change them.
- Category must be ONE of: AI, Hardware, Software, Security, Science, Business, Policy, Startups

Here are the real articles:
{articles_text}

You MUST respond with ONLY a valid JSON object. No markdown, no extra text, nothing outside the JSON.

The JSON must match this exact structure:
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
      "key_points": ["under 8 words", "under 8 words", "under 8 words"],
      "impact": "one simple sentence under 20 words"
    }}
  ]
}}

Process all {len(articles)} articles. Keep the most important story first. Use simple English throughout. Never change the URL."""

    print(f"   Sending {len(articles)} real articles to Gemini for simplification...")
    
    response = model.generate_content(prompt)
    raw_response = response.text.strip()
    
    # Strip accidental markdown fences
    if raw_response.startswith("```"):
        lines = raw_response.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_response = "\n".join(lines).strip()
    
    news_data = json.loads(raw_response)
    return news_data


def fetch_news():
    """Main function — fetch real news and simplify it."""
    
    news_api_key = os.environ["NEWS_API_KEY"]
    gemini_api_key = os.environ["GEMINI_API_KEY"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    print(f"[{today}] Step 1: Fetching real news from NewsAPI...")
    real_articles = fetch_real_news(news_api_key)
    
    print(f"[{today}] Step 2: Simplifying with Gemini...")
    news_data = simplify_with_gemini(real_articles, gemini_api_key)
    
    # Validate top-level keys
    required_keys = {"date", "generated_at", "edition", "articles"}
    missing = required_keys - set(news_data.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    
    if not isinstance(news_data["articles"], list) or len(news_data["articles"]) == 0:
        raise ValueError("articles must be a non-empty list")
    
    # Validate each article and ensure URLs are preserved
    article_required = {"title", "summary", "category", "source", "url", "key_points", "impact"}
    for i, article in enumerate(news_data["articles"]):
        missing_fields = article_required - set(article.keys())
        if missing_fields:
            raise ValueError(f"Article {i} missing fields: {missing_fields}")
        
        # If Gemini dropped the URL, restore it from original
        if not article.get("url") and i < len(real_articles):
            article["url"] = real_articles[i]["url"]
    
    # Override generated_at with actual current UTC time
    news_data["generated_at"] = datetime.now(timezone.utc).isoformat()
    
    with open("news.json", "w", encoding="utf-8") as f:
        json.dump(news_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ news.json written with {len(news_data['articles'])} real articles.")
    for idx, a in enumerate(news_data["articles"]):
        print(f"   [{idx+1}] [{a['category']}] {a['title']}")
        print(f"         URL: {a.get('url', 'NO URL')}")


if __name__ == "__main__":
    fetch_news()
