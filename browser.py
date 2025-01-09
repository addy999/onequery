import ast
import base64
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import requests
from playwright.async_api import Page, async_playwright
from pydantic import BaseModel

from .index import RAGSystem
from .agent import fetch_query_for_rag, get_reply, summarize_text


def call_process_image_api(
    image_path, box_threshold=0.05, iou_threshold=0.1, timeout=30
):
    start = time.time()
    url = os.environ.get("OMNIPARSER_API")
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    files = {"image_file": ("image.png", image_data, "image/png")}
    params = {"box_threshold": box_threshold, "iou_threshold": iou_threshold}

    response = requests.post(url, files=files, params=params, timeout=timeout)

    if response.status_code == 200:
        resp = response.json()
        print(f"Image API call took: {time.time() - start:.2f}s")
        return resp["image"], resp["parsed_content_list"], resp["label_coordinates"]
    else:
        raise Exception(f"Request failed with status code {response.status_code}")


# wake up the server
try:
    call_process_image_api("downloaded_image.png", 0.05, 0.1, timeout=60)
except Exception:
    pass


@dataclass
class WebElement:
    id: int
    text: str
    x: float
    y: float
    width: float
    height: float
    element_type: str  # 'text' or 'icon'

    @property
    def center(self) -> Tuple[float, float]:
        """Returns the center coordinates of the element"""
        return (self.x + (self.width / 2), self.y + (self.height / 2))

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Returns the boundary coordinates (x1, y1, x2, y2)"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class WebPageProcessor:
    def __init__(self):
        self.elements: Dict[int, WebElement] = {}

    def load_elements(self, text_boxes: str, coordinates: str) -> None:
        """
        Load elements from the processed webpage data

        Args:
            text_boxes: String mapping ID to text content
            coordinates: String mapping ID to [x, y, width, height] lists
        """

        self.elements = {}

        def parse_text_boxes(text: str) -> dict:
            # Split into lines and filter empty lines
            lines = [line.strip() for line in text.split("\n") if line.strip()]

            # Dictionary to store results
            boxes = {}

            for line in lines:
                # Split on ":" to separate ID from text
                id_part, text_part = line.split(":", 1)

                # Extract ID number using string operations
                id_str = id_part.split("ID")[1].strip()
                id_num = int(id_str)

                # Store in dictionary with cleaned text
                boxes[id_num] = text_part.strip()

            return boxes

        def parse_coordinates(coords: str) -> dict:
            """
            Example string:
            `{'0': [0.89625, 0.04333332697550456, 0.06125, 0.03], '1': [0.01875, 0.14499998728434244, 0.34875, 0.03833333333333333]}`
            """
            return ast.literal_eval(coords)

        coordinates = parse_coordinates(coordinates)
        for element_id, text in parse_text_boxes(text_boxes).items():
            id_str = str(element_id)
            if id_str in coordinates:
                coords = coordinates[id_str]
                element_type = "icon" if "Icon Box" in text else "text"

                self.elements[element_id] = WebElement(
                    id=element_id,
                    text=text.strip(),
                    x=coords[0],
                    y=coords[1],
                    width=coords[2],
                    height=coords[3],
                    element_type=element_type,
                )

    async def click_element(self, page, element_id: int) -> None:
        """Click an element using its center coordinates"""
        if element_id not in self.elements:
            raise ValueError(f"Element ID {element_id} not found")

        element = self.elements[element_id]
        x, y = element.center

        # Convert normalized coordinates to actual pixels
        viewport_size = await page.viewport()
        actual_x = x * viewport_size["width"]
        actual_y = y * viewport_size["height"]

        await page.mouse.click(actual_x, actual_y)

    def find_elements_by_text(
        self, text: str, partial_match: bool = True
    ) -> List[WebElement]:
        """Find elements containing the specified text"""
        matches = []
        for element in self.elements.values():
            if partial_match and text.lower() in element.text.lower():
                matches.append(element)
            elif not partial_match and text.lower() == element.text.lower():
                matches.append(element)
        return matches

    def get_nearby_elements(
        self, element_id: int, max_distance: float = 0.1
    ) -> List[WebElement]:
        """Find elements within a certain distance of the specified element"""
        if element_id not in self.elements:
            raise ValueError(f"Element ID {element_id} not found")

        source = self.elements[element_id]
        nearby = []

        for element in self.elements.values():
            if element.id == element_id:
                continue

            # Calculate center-to-center distance
            sx, sy = source.center
            ex, ey = element.center
            distance = ((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5

            if distance <= max_distance:
                nearby.append(element)

        return nearby


@dataclass
class Action:
    action_type: str
    params: Optional[Dict[str, Union[str, int]]]


class PlaywrightExecutor:
    def __init__(self, page: Page, web_processor: "WebPageProcessor"):
        self.page = page
        self.processor = web_processor

    async def execute_action(self, action_str: str) -> None:
        """Execute a Playwright action from a string command."""
        print("> Executing action:", action_str)
        action = self.parse_action(action_str)
        element = None
        if "uid" in action.params:
            element = self.processor.elements.get(int(action.params["uid"]))
            if not element:
                raise ValueError(f"Element with uid {action.params['uid']} not found")
        if action.action_type == "click":
            await self._execute_click(element)
        elif action.action_type == "text_input":
            await self._execute_change(element, action.params["text"])
        elif action.action_type == "change":
            await self._execute_change(element, action.params["value"])
        elif action.action_type == "load":
            await self._execute_load(action.params["url"])
        elif action.action_type == "scroll":
            await self._execute_scroll(int(action.params["x"]), int(action.params["y"]))
        elif action.action_type == "submit":
            await self._execute_submit(element)
        elif action.action_type == "back":
            await self.page.go_back()
        elif action.action_type == "enter":
            await self.page.keyboard.press("Enter")
        elif action.action_type == "nothing":
            pass
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")

    def parse_action(self, action_str: str) -> Action:
        """Parse an action string into an Action object."""
        if action_str == "back":
            return Action(action_type="back", params={})
        if action_str == "enter":
            return Action(action_type="enter", params={})
        if action_str == "nothing":
            return Action(action_type="nothing", params={})
        action_type = action_str[: action_str.index("(")]
        params_str = action_str[action_str.index("(") + 1 : action_str.rindex(")")]
        params = {}
        if params_str:
            param_pairs = params_str.split(",")
            for pair in param_pairs:
                key, value = pair.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                params[key] = value
        return Action(action_type=action_type, params=params)

    async def _execute_click(self, element: "WebElement") -> None:
        """Execute a click action."""
        x, y = element.center
        viewport = self.page.viewport_size
        actual_x = x * viewport["width"]
        actual_y = y * viewport["height"]
        await self.page.mouse.move(actual_x, actual_y)
        await self.page.mouse.click(actual_x, actual_y, delay=100)

    async def _execute_text_input(self, element: "WebElement", text: str) -> None:
        """Execute a text input action."""
        x, y = element.center
        viewport = self.page.viewport_size
        actual_x = x * viewport["width"]
        actual_y = y * viewport["height"]
        await self.page.mouse.click(actual_x, actual_y, delay=100)
        await self.page.keyboard.type(text, delay=100)

    async def _execute_change(self, element: "WebElement", value: str) -> None:
        """Execute a change action."""
        x, y = element.center
        viewport = self.page.viewport_size
        actual_x = x * viewport["width"]
        actual_y = y * viewport["height"]
        await self.page.mouse.click(actual_x, actual_y)
        await self.page.keyboard.down("Meta")
        await self.page.keyboard.press("A")
        await self.page.keyboard.up("Meta")
        await self.page.keyboard.type(value, delay=100)

    async def _execute_load(self, url: str) -> None:
        """Execute a load action."""
        await self.page.goto(url)

    async def _execute_scroll(self, x: int, y: int) -> None:
        """Execute a scroll action."""
        await self.page.evaluate(f"window.scrollTo({x}, {y})")
        await self.page.wait_for_timeout(1000)

    async def _execute_submit(self, element: "WebElement") -> None:
        """Execute a submit action."""
        x, y = element.center
        viewport = self.page.viewport_size
        actual_x = x * viewport["width"]
        actual_y = y * viewport["height"]
        await self.page.mouse.click(actual_x, actual_y)


class WebScraper:
    def __init__(self, task, start_url, output_model: BaseModel):
        self.task = task
        self.start_url = start_url
        index_path = "output/index"
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        self.rag = RAGSystem(index_path="output/index")
        self.web_processor = WebPageProcessor()
        self.output_model = output_model
        self.browser = None

    async def main(self, p):
        # locally
        self.browser = await p.chromium.launch(
            headless=True,
        )

        context = await self.browser.new_context(
            record_video_dir="videos/",
            record_video_size={"width": 1920, "height": 1080},
        )
        page = await context.new_page()
        await page.set_viewport_size({"width": 1920, "height": 1080})
        next_task = (
            "Find the website to visit."
            if "google.com" in self.start_url
            else "Figure out what to do on the website."
        )
        next_action = f'load(url="{self.start_url}")'
        second_action = None
        max_iterations = 30
        iteration_count = 0
        state = [
            {
                "role": "user",
                "content": "Overall goal: " + self.task,
            }
        ]
        while next_task and iteration_count < max_iterations:
            executor = PlaywrightExecutor(page, self.web_processor)
            await executor.execute_action(next_action)
            time.sleep(1)
            if second_action:
                await executor.execute_action(second_action)
                time.sleep(1)
            print("> Inspecting the screen...")
            start_time = datetime.now()
            await page.screenshot(path="screenshot.png", scale="css")
            img, parsed, coordinates = call_process_image_api(
                "screenshot.png", 0.2, 0.1
            )

            # Save the base64 image locally as "screenshot.png"
            image_data = base64.b64decode(img)
            with open("screenshot.png", "wb") as f:
                f.write(image_data)

            end_time = datetime.now()
            print(f"Inspection took: {(end_time - start_time).total_seconds()}s")
            self.web_processor.load_elements(parsed, coordinates)
            text_content = " ".join(
                [
                    a.text
                    for a in self.web_processor.elements.values()
                    if a.element_type == "text"
                ]
            )
            self.rag.add_document(
                text_content,
                {"url": page.url, "timestamp": datetime.now().isoformat()},
            )
            state.append(
                {
                    "role": "user",
                    "content": "Elements on screen: " + parsed,
                    # {
                    #     "type": "image",
                    #     "source": {
                    #         "type": "base64",
                    #         "media_type": "image/png",
                    #         "data": img,
                    #     },
                    # },
                    # ],
                }
            )
            print("> Getting reply from AI...")
            start_time = datetime.now()
            reply = get_reply(state)
            print("> AI time taken:", (datetime.now() - start_time).total_seconds())

            next_task, next_action, second_action = (
                reply["next_task"],
                reply["next_action"],
                reply.get("next_action_2"),
            )
            print("> next_task", next_task, next_action, second_action)
            state.append(
                {
                    "role": "assistant",
                    "content": f"Next task: {next_task}. Next action: {next_action}",
                }
            )

            if next_action == "nothing":
                print("> No further action required.")
                iteration_count += 1000
            else:
                iteration_count += 1
        return page, context

    async def run(self):
        async with async_playwright() as p:
            start = time.time()
            page, context = await self.main(p)

            rag_query = fetch_query_for_rag(self.task)
            print("> Querying RAG for task:", rag_query)
            docs = [a["text"] for a in self.rag.query(rag_query)]
            answer = summarize_text(self.task, docs, self.output_model)
            print("> Answer:", answer)
            print("> Total time taken:", time.time() - start)

            try:
                await context.close()
            except Exception as e:
                raise Warning(e)

        return answer
