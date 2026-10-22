# llm_clients.py
import base64
import json
from pathlib import Path
from typing import Optional
import os.path as osp
import requests
from openai import OpenAI
import logging
logger = logging.getLogger("uvicorn.error")


#LLM_HOST = "http://127.0.0.1:18081"
LLM_HOST = "http://192.168.0.52:18081"
#LLM_MODEL = "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
LLM_MODEL = "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
REMOTE_LLM_HOST = "https://api.deepseek.com"
REMOTE_LLM_MODEL = "deepseek-v4-flash"
API_KEY = "sk-ba92822a2eec4d5d8c54219e3d499eb6"

def _image_to_base64(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")

def call_remote_llm_fact_sementic(
        text: str,
        model: str = REMOTE_LLM_MODEL,
        host: str = REMOTE_LLM_HOST,
        timeout: int = 120, ) -> str:
    messages = [{"role": "user",
                 "content": text,
                 }]
    text_answer = None
    client = OpenAI(api_key=API_KEY,
                    base_url=host)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        timeout=timeout
    )
    try:
        # OpenAI v1.0+ 对象格式
        text_answer = response.choices[0].message.content
    except (TypeError, AttributeError):
        try:
            # 字典格式
            text_answer = response['choices'][0]['message']['content']
        except (KeyError, TypeError):
            # 原始格式
            text_answer = response.choices[0].get("message", {}).get("content", "")
    return text_answer

def call_remote_llm(
        content: list,
        model: str = REMOTE_LLM_MODEL,
        host: str = REMOTE_LLM_HOST,
        timeout: int = 120,) -> str:
    text_answer = None
    img_answer = None
    client = OpenAI(api_key=API_KEY,
                    base_url=host)
    for c in content[0]["content"]:
        if c["type"] == "text":
            prompt = c["text"]
            messages = [{"role": "user",
                         "content": prompt,
                         }]
            response = client.chat.completions.create(
                model = model,
                messages = messages,
                stream = False,
                reasoning_effort="high",
                extra_body = {"thinking": {"type": "enabled"}},
                timeout=timeout
            )
            #response.raise_for_status()
            logger.info("Response: %s", response) 
            try:
                # OpenAI v1.0+ 对象格式
                text_answer = response.choices[0].message.content
            except (TypeError, AttributeError):
                try:
                    # 字典格式
                    text_answer = response['choices'][0]['message']['content']
                except (KeyError, TypeError):
                    # 原始格式
                    text_answer = response.choices[0].get("message", {}).get("content", "")
        if c["type"] == "image_url":
            image_url = c["image_url"]["url"]
            if osp.exists(image_url):
                image_b64 = _image_to_base64(image_url)
                messages = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    },
                }
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    reasoning_effort="high",
                    extra_body={"thinking": {"type": "enabled"},},
                    timeout=timeout
                )
                #response.raise_for_status()
                try:
                    # OpenAI v1.0+ 对象格式
                    img_answer = response.choices[0].message.content
                except (TypeError, AttributeError):
                    try:
                        # 字典格式
                        img_answer = response['choices'][0]['message']['content']
                    except (KeyError, TypeError):
                        # 原始格式
                        img_answer = response.choices[0].get("message", {}).get("content", "")

    if img_answer:
        return text_answer + "\n" + img_answer + "\n"
    return text_answer

def call_local_llm(
    content: list,
    model: str = LLM_MODEL,
    host: str = LLM_HOST,
    timeout: int = 360,
) -> str:
    text_answer = None
    img_answer = None
    #print("content.............:", content)
    for c in content[0]["content"]:
        if c["type"] == "text":
            prompt = c["text"]
            url = f"{host.rstrip('/')}/v1/chat/completions"
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                    ],
                "temperature": 0.1,
                "max_tokens": 9162,
                "stream": False,
            }
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            text_answer = r.json()["choices"][0]["message"]["content"]
        if c["type"] == "image_url":
            image_url = c["image_url"]["url"]
            if osp.exists(image_url):
                img_answer = call_local_vlm_for_image(image_url, text_answer)
    if img_answer:
        return text_answer + "\n" + img_answer + "\n"
    return text_answer


def call_local_vlm_for_image(
    image_path: str,
    prompt: str,
    model: str = LLM_MODEL,
    host: str = LLM_HOST,
    timeout: int = 240,
) -> str:
    """
    适配 OpenAI-compatible 多模态接口。
    如果你的服务不是这种格式，需要按你的本地接口改。
    """
    image_b64 = _image_to_base64(image_path)
    url = f"{host.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]


def clean_dirty_text_with_llm(raw_text: str) -> str:
    prompt = f"""
You are a document cleanup assistant.

Please clean the following OCR or extracted text.

Rules:
1. Remove meaningless repeated characters like sssssssss, ::::::, ........ when they are clearly OCR noise.
2. Repair split letters like "i t e m 1" -> "item 1" when obvious.
3. Preserve technical terms, model names, numbers, table values, units, and product names.
4. If a line is unreadable garbage, remove it.
5. Keep the output concise, readable, and faithful to source meaning.
6. Do NOT invent facts.
7. Return only cleaned text.

RAW TEXT:
{raw_text}
"""
    return call_local_llm(prompt).strip()


def summarize_image_with_vlm(image_path: str, nearby_text: Optional[str] = None) -> str:
    extra = f"\nNearby document text:\n{nearby_text}\n" if nearby_text else ""
    prompt = f"""
Describe this technical document image for retrieval.

Requirements:
1. Identify whether it is a diagram, architecture figure, rack layout, flow chart, screenshot, or table image.
2. Extract the main topic and important entities.
3. Mention visible labels, product names, model names, ports, network, components, and relationships if legible.
4. If there is a watermark or rotated overlay text, ignore it unless it changes document meaning.
5. Output one short paragraph suitable for semantic retrieval.
{extra}
"""
    return call_local_vlm_for_image(image_path=image_path, prompt=prompt).strip()