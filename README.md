# OneQuery

[![GitHub License](https://img.shields.io/github/license/addy999/onequery)](https://github.com/addy999/onequery/blob/main/LICENSE)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/addy999/onequery)](https://github.com/addy999/onequery/commits/main)
[![Buy Me a Coffee](https://img.shields.io/badge/buy%20me%20a%20coffee-donate-yellow)](https://buymeacoffee.com/adibhatia)

> 🔨 **Note:** This repository is still in development. Contributions and feedback are welcome!

## Setup

- Requirements: `pip install -r requirements.txt`
- Install browser: `python -m playwright install`

## Usage

### General query with no source to start with

```python
task = "Find 2 recent issues from PyTorch repository."

class IssueModel(BaseModel):
    date: str
    title: str
    author: str
    description: str

class OutputModel(BaseModel):
    issues: list[IssueModel]

scraper = WebScraper(task, None, OutputModel)
scraper.run()
```

### If you know the URL

```python
start_url = "https://in.bookmyshow.com/"
task = "Find 5 events happening in Bangalore this week."

class EventsModel(BaseModel):
    name: str
    date: str
    location: str

class OutputModel(BaseModel):
    events: list[EventsModel]

scraper = WebScraper(task, start_url, OutputModel)
scraper.run()
```

### Serving with a REST API

Server:

```bash
pip install fastapi[all]
```

```python
uvicorn server:app --reload
```

Client:

```python
import requests

url = "http://0.0.0.0:8000/scrape"

payload = {
    "start_url": "http://example.com",
    "task": "Scrape the website for data",
    "schema": {
        "title": (str, ...),
        "description": (str, ...)
    }
}

response = requests.post(url, json=payload)

print(response.status_code)
print(response.json())
```

## Architecture

(needs to be revised)

### Flowchart

```mermaid
graph TD;
    A[Text Query] --> B[WebLLM];
    B --> C[Browser Instructions];
    C --> D[Browser Execution];
    D --> E[OmniParser];
    E --> F[Screenshot & Structured Info];
    F --> G[AI];
    C --> G;
    G --> H[JSON Output];
```

### Stack

- Browser: Puppeteer
- Parser: [OmniParser](https://huggingface.co/spaces/microsoft/OmniParser)
- WebLlama: https://webllama.github.io/
