import os
import json
import google.generativeai as genai
from datetime import datetime, timezone

def fetch_news():
    """Fetch AI-generated tech news from Gemini and save to news.json."""

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prompt = f"""Today is {today}. You are a tech news summarizer. Your job is to write short, simple, easy-to-read tech news for everyday people.

WRITING RULES (follow strictly):
- Use simple everyday words. No complex or fancy English.
- Keep sentences short. Maximum 15 words per sentence.
- Be direct. Say what happened. No fluff.
- Write like you are texting a friend who is smart but not a tech expert.
- Each summary must be 2-3 short sentences only. No more.
- Key points must be under 8 words each.
- Impact must be one simple sentence. Max 20 words.

Your task: Write exactly 5 tech news stories from today.

You MUST respond with ONLY a valid JSON object. No markdown, no extra text, nothing outside the JSON.

The JSON must match this exact structure:
{{
  "date": "{today}",
  "generated_at": "{datetime.now(timezone.utc).isoformat()}",
  "edition": "Daily AI Tech Briefing",
  "articles": [
    {{
      "title": "short simple headline under 10 words",
      "summary": "2-3 short simple sentences. Easy words only.",
      "category": "one of: AI, Hardware, Software, Security, Science, Business, Policy, Startups",
      "source": "e.g. The Verge, Wired, TechCrunch",
      "key_points": ["under 8 words", "under 8 words", "under 8 words"],
      "impact": "one simple sentence under 20 words"
    }}
  ]
}}

Generate exactly 5 articles. Put the most important story first. Use simple English throughout."""

    print(f"[{today}] Calling Gemini API...")

    response = model.generate_content(prompt)
    raw_response = response.text.strip()

    print("Gemini responded. Parsing JSON...")

    # Strip accidental markdown fences if the model added them
    if raw_response.startswith("```"):
        lines = raw_response.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_response = "\n".join(lines).strip()

    news_data = json.loads(raw_response)

    # Validate top-level keys
    required_keys = {"date", "generated_at", "edition", "articles"}
    missing = required_keys - set(news_data.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    if not isinstance(news_data["articles"], list) or len(news_data["articles"]) == 0:
        raise ValueError("articles must be a non-empty list")

    # Validate each article
    article_required = {"title", "summary", "category", "source", "key_points", "impact"}
    for i, article in enumerate(news_data["articles"]):
        missing_fields = article_required - set(article.keys())
        if missing_fields:
            raise ValueError(f"Article {i} missing fields: {missing_fields}")

    # Override generated_at with actual current UTC time for accuracy
    news_data["generated_at"] = datetime.now(timezone.utc).isoformat()

    with open("news.json", "w", encoding="utf-8") as f:
        json.dump(news_data, f, indent=2, ensure_ascii=False)

    print(f"✅ news.json written with {len(news_data['articles'])} articles.")
    for idx, a in enumerate(news_data["articles"]):
        print(f"   [{idx+1}] [{a['category']}] {a['title']}")


if __name__ == "__main__":
    fetch_news()
