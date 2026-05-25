import os
import json
import anthropic
from datetime import datetime, timezone

def fetch_news():
    """Fetch AI-generated tech news from Claude and save to news.json."""
    
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    prompt = f"""Today is {today}. You are a senior tech journalist writing a daily AI-generated technology briefing.

Your task: Produce a briefing of exactly 8 top technology news stories for today. These should be plausible, realistic, high-quality summaries of the kinds of major tech stories that would appear in top outlets like The Verge, Wired, TechCrunch, and Ars Technica.

Each article MUST include:
- title: A compelling headline (max 15 words)
- summary: A 2–3 sentence informative summary (80–120 words)
- category: Exactly ONE of these: AI, Hardware, Software, Security, Science, Business, Policy, Startups
- source: A realistic news source name (e.g., "The Verge", "Wired", "Reuters", "TechCrunch")
- key_points: An array of exactly 3 short bullet-point strings (each under 12 words)
- impact: One sentence explaining why this story matters to the tech industry or general public (30–50 words)

Ordering: Put the single most important/impactful story FIRST — it will be displayed as the hero article.

You MUST respond with ONLY a valid JSON object. Do NOT include any markdown fences, preamble, commentary, or text outside the JSON object.

The JSON object must match this exact schema:
{{
  "date": "{today}",
  "generated_at": "<ISO 8601 UTC timestamp of now>",
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

Generate exactly 8 articles covering a diverse range of the categories listed above. Make the content realistic, insightful, and genuinely useful to a tech-savvy reader."""

    print(f"[{today}] Calling Claude API...")
    
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    raw_response = message.content[0].text.strip()
    print("Claude responded. Parsing JSON...")
    
    # Strip accidental markdown fences if Claude added them
    if raw_response.startswith("```"):
        lines = raw_response.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_response = "\n".join(lines).strip()
    
    # Parse and validate
    news_data = json.loads(raw_response)
    
    # Validate top-level keys
    required_keys = {"date", "generated_at", "edition", "articles"}
    missing = required_keys - set(news_data.keys())
    if missing:
        raise ValueError(f"Missing required top-level keys in response: {missing}")
    
    if not isinstance(news_data["articles"], list) or len(news_data["articles"]) == 0:
        raise ValueError("articles must be a non-empty list")
    
    # Validate each article
    article_required = {"title", "summary", "category", "source", "key_points", "impact"}
    valid_categories = {"AI", "Hardware", "Software", "Security", "Science", "Business", "Policy", "Startups"}
    
    for i, article in enumerate(news_data["articles"]):
        missing_fields = article_required - set(article.keys())
        if missing_fields:
            raise ValueError(f"Article {i} missing fields: {missing_fields}")
        
        if article["category"] not in valid_categories:
            print(f"  Warning: Article {i} has non-standard category '{article['category']}', keeping as-is.")
        
        if not isinstance(article["key_points"], list):
            raise ValueError(f"Article {i} key_points must be a list")
    
    # Override generated_at with actual current UTC time for accuracy
    news_data["generated_at"] = datetime.now(timezone.utc).isoformat()
    
    # Write to news.json
    output_path = "news.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(news_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ news.json written successfully with {len(news_data['articles'])} articles.")
    print(f"   Edition: {news_data['edition']} — {news_data['date']}")
    for idx, a in enumerate(news_data["articles"]):
        print(f"   [{idx+1}] [{a['category']}] {a['title']}")


if __name__ == "__main__":
    fetch_news()
