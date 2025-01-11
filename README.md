# OneQuery

[![GitHub License](https://img.shields.io/github/license/addy999/onequery)](https://github.com/addy999/onequery/blob/main/LICENSE)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/addy999/onequery)](https://github.com/addy999/onequery/commits/main)
[![Buy Me a Coffee](https://img.shields.io/badge/buy%20me%20a%20coffee-donate-yellow)](https://buymeacoffee.com/adibhatia)

> 🔨 **Note:** This repository is still in development. Contributions and feedback are welcome!

## Setup

- Requirements: `pip install -r requirements.txt`
- Install browser: `python -m playwright install`
  - This project uses Playwright to control the browser. You can install the browser of your choice using the command above.
- Write your environment variables in a `.env` file (see `.env.test`)
- Install OmniParser
  - For webpage analysis, we use the [OmniParser](https://huggingface.co/spaces/microsoft/OmniParser) model from Hugging Face. You'll need to host it via an [API](https://github.com/addy999/omniparser-api) locally.

## Examples

- Finding issues on a github repo

[![Video Demo 1](http://img.youtube.com/vi/a_QPDnAosKM/0.jpg)](https://youtu.be/a_QPDnAosKM?si=pXtZgrRlvXzii7FX "Finding issues on a GitHub repo")
  
- Finding live events

[![Video Demo 2](http://img.youtube.com/vi/sp_YuZ1Q4wU/0.jpg)](https://youtu.be/sp_YuZ1Q4wU?feature=shared "Finding live events")

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

> 💡 **Tip:** For a hosted solution with a lightning fast Zig based browser, worldwide proxy support, and job queuing system, check out [onequery.app](https://www.onequery.app).

## Testing

In the works

## Status

- ✅ Basic functionality
- 🛠️ Testing
- 🛠️ Documentation

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
