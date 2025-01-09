from fastapi import FastAPI, Body
from typing import List
from rich.traceback import install
from pydantic import BaseModel
from .browser import WebScraper
from pydantic import create_model

install(show_locals=True)

app = FastAPI()


class ScrapeRequestModel(BaseModel):
    start_url: str
    task: str
    schema: dict


@app.post("/scrape")
async def scrape(request: ScrapeRequestModel = Body(...)):
    start_url = request.start_url
    task = request.task
    schema = request.schema
    model = create_model(
        "ResponseModel", **{key: (value, ...) for key, value in schema.items()}
    )

    class OutputModel(BaseModel):
        results: List[model]

        class Config:
            arbitrary_types_allowed = True

    scraper = WebScraper(task, start_url, OutputModel)
    result = await scraper.run()
    return result
