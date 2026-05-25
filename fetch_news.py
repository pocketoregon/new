import os
import json
import google.generativeai as genai
from datetime import datetime, timezone

def fetch_news():
    """Fetch AI-generated tech news from Gemini and save to news.json."""

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prompt = f"""Today is {today}. You are a senior tech journalist writing a daily AI-generated technology briefing.

Your task: Produce a briefing of exactly 8 top technology news stories for today. These should be plausible, realistic, high-quality summaries of the kinds of major tech stories that would appear in top outlets like The Verge, Wired, TechCrunch, and Ars Technica.

Each article MUST include:
- title: A compelling headline (max 15 words)
- summary: A 2–3 sentence informative summary (80–120 words)
- category: Exactly ONE of these: AI, Hardware, Software, Security, Science, Business, Policy, Startups
- source: A realistic news source name (e.g., "The Verge", "Wired", "Reuters", "TechCrunch")
- key_points: An array of exactly 3 short bullet-point strings (each under 12 words)
- impact: One sentence explaining why this story matters (30–50 words)

Ordering: Put the single most important/impactful story FIRST — it will be displayed as the hero article.

You MUST respond with ONLY a valid JSON object. Do NOT include any markdown fences, preamble, commentary, or text outside the JSON object.

The JSON object must match this exact schema:
{{
  "date": "{today}",
  "generated_at": "{datetime.now(timezone.utc).isoformat()}",
  "edition": "Daily AI Tech Briefing",
  "articles": [
    {{
      "title": "string",
      "summary": "string",
      "category": "string",
      "source": "string",
      "key_points": ["string", "string", "string"],
      "impact": "string"
    }}
  ]
}}

Generate exactly 8 articles covering a diverse range of the categories listed above."""

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

    news_data["generated_at"] = datetime.now(timezone.utc).isoformat()

    with open("news.json", "w", encoding="utf-8") as f:
        json.dump(news_data, f, indent=2, ensure_ascii=False)

    print(f"✅ news.json written with {len(news_data['articles'])} articles.")
    for idx, a in enumerate(news_data["articles"]):
        print(f"   [{idx+1}] [{a['category']}] {a['title']}")


if __name__ == "__main__":
    fetch_news()
