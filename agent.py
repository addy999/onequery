import anthropic
import re
import json
from pydantic import BaseModel
from litellm import completion

client = anthropic.Anthropic()
SYSTEM_PROMPT = open("./agent_prompt.txt").read()
SYSTEM_JSON_PROMPT = open("./agent_prompt_json.txt").read()


def parse_text(text):
    next_action_pattern = r"<next_action-1>\n(.*?)\n</next_action-1>"
    next_action2_pattern = r"<next_action-2>\n(.*?)\n</next_action-2>"
    explanation_pattern = r"<explanation>\n(.*?)\n</explanation>"
    next_task_pattern = r"<next_task>\n(.*?)\n</next_task>"

    next_action_match = re.search(next_action_pattern, text, re.DOTALL)
    next_action2_match = re.search(next_action2_pattern, text, re.DOTALL)
    explanation_match = re.search(explanation_pattern, text, re.DOTALL)
    next_task_match = re.search(next_task_pattern, text, re.DOTALL)

    result = {
        "next_action": next_action_match.group(1) if next_action_match else None,
        "next_action_2": (next_action2_match.group(1) if next_action2_match else None),
        "explanation": explanation_match.group(1) if explanation_match else None,
        "next_task": next_task_match.group(1) if next_task_match else None,
    }

    return result


def is_valid_json(string: str) -> bool:
    try:
        json.loads(string)
        return True
    except json.JSONDecodeError:
        return False


def clean_up_json(string: str) -> str:
    def extract_json_from_string(string):
        start_index = string.find("{")
        end_index = string.rfind("}")
        if start_index != -1 and end_index != -1:
            return string[start_index : end_index + 1]
        return ""

    cleaned = (
        extract_json_from_string(string)
        .strip()
        .replace("\n", "")
        .replace('\\"', "'")
        .replace("```", "")
        .replace("json", "")
    )
    if not is_valid_json(cleaned):
        raise ValueError("String not valid", cleaned)
    return cleaned


def get_reply(state, mode="deepseek") -> str:

    if mode == "ollama":
        reply = (
            completion(
                model="ollama/llama3.3",
                max_tokens=256,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + state,
                temperature=0.3,
            )
            .choices[0]
            .message.content
        )

    elif mode == "deepseek":
        reply = (
            completion(
                model="deepseek/deepseek-chat",
                max_tokens=256,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + state,
                temperature=0.3,
            )
            .choices[0]
            .message.content
        )

    elif mode == "anthropic":
        reply = (
            client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=256,
                system=SYSTEM_PROMPT,
                temperature=0.3,
                messages=state,
            )
            .content[0]
            .text
        )

    return parse_text(reply)


def summarize_text(prompt: str, documents: list, schema: BaseModel) -> str:
    return json.loads(
        clean_up_json(
            (
                "{"
                + client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    system=f"""Summarize the following documents for this prompt in JSON format.
                
                Prompt: {prompt}
                
                Return using this schema: {schema.schema()}""",
                    temperature=0.3,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": text,
                                }
                                for text in documents
                            ],
                        },
                        {"role": "assistant", "content": "{"},
                    ],
                )
                .content[0]
                .text
            )
        )
    )


def fetch_query_for_rag(task: str) -> str:
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=256,
        system="Generate a simple keyword/phrase query for a RAG system based on the following task. Return the query as JSON with 'query' key. The query should help fetch documents relevant to the task: "
        + task,
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": task,
            }
        ],
    )
    return json.loads(response.content[0].text)["query"]
