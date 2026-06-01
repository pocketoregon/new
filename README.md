# AI·Brief — Daily Tech Intelligence

An automated tech news aggregator that fetches, deduplicates, and rewrites the latest technology headlines using AI — then publishes them as a clean, fast web frontend via GitHub Pages.

## How It Works

1. **Fetch** — Pulls top 40 tech headlines from NewsAPI every hour
2. **Deduplicate** — Filters out already-seen articles using URL-based fingerprinting
3. **Rewrite** — Rewrites unseen articles into crisp executive summaries using GPT-4o mini
4. **Publish** — Commits updated `news.json`, `feed.json`, and `archive.json` to the repo
5. **Display** — `index.html` reads the JSON files and renders the frontend

## Setup

### 1. Fork or clone this repo

### 2. Add these two secrets in your repo Settings → Secrets → Actions:

| Secret | Description |
|--------|-------------|
| `NEWS_API_KEY` | Your API key from [newsapi.org](https://newsapi.org) |
| `MODEL_ACCESS_TOKEN` | Your GitHub Personal Access Token (with repo write permissions) |

### 3. Enable GitHub Pages
Go to **Settings → Pages** and set the source to `main` branch, `/ (root)`.

### 4. Enable GitHub Actions
The workflow runs automatically every hour via `.github/workflows/daily_news.yml`.
You can also trigger it manually from the **Actions** tab.

## Project Structure
├── fetch_news.py          # Core pipeline script
├── requirements.txt       # Python dependencies
├── news.json              # Latest batch snapshot
├── feed.json              # Frontend feed data
├── archive.json           # Rolling 30-batch archive
├── index.html             # Frontend UI
└── .github/
└── workflows/
└── daily_news.yml # GitHub Actions automation

## Tech Stack

- **Python 3.11** — pipeline script
- **NewsAPI** — news source
- **GPT-4o mini** (via GitHub Models / Azure) — AI rewriting
- **Pydantic** — structured output enforcement
- **GitHub Actions** — automation
- **GitHub Pages** — hosting

## License

MIT
