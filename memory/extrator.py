import requests
import json
from config import OLLAMA_URL, MODEL
from llm_clients import *
import logging
logger = logging.getLogger("uvicorn.error")
class MemoryExtractor:

    def extract(self, text):

        prompt = f"""
        Extract structured memory.

        Return STRICT JSON:

        {{
          "facts": {{
            "key": "value"
          }},
          "semantic": ["concept1", "concept2"]
        }}

        Rules:
        - Only JSON
        - No markdown
        - No explanation
        - English keys only

        TEXT:
        {text}
        """
        logger.info("============llama response==============")
        #fact = text.split("\n")[0]
        #semantic = text.split("\n")[1:]
        #logger.info(prompt)
        """
        r = requests.post(
            f"{OLLAMA_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                    ],
                "temperature": 0.1,
                "max_tokens": 9162,
                "stream": False
            }
        )
        """
        ### Arguments: ('{\n  "facts": {\n    "nvlink_details_not_provided": "Current context has no detailed NVLink introduction,
        # only cross-references to other manuals.",\n    "nvlink_connections_verify": "Verify that all the NVLink connections are active.",\n
        # "references": "NVLink Management Software, IMEX Service for NVLink Networks, NMX-T documentation, NMX-C documentation, GB200 NVL72 Rack System Specification and Integration Guide"\n  },\n
        # "semantic": ["NVLink", "NVLink Management Software", "IMEX Service", "NMX-T", "NMX-C", "GB200 NVL72", "cross-reference", "verification"]\n}',)
        r = call_remote_llm_fact_sementic(prompt)
        # 添加调试信息
        logger.info(f"Input type: {type(r)}")
        logger.info(f"Input content (first 200 chars): {str(r)[:200]}")

        # 然后根据类型处理
        if isinstance(r, str):
            logger.info("Input is string, attempting to parse as JSON...")
            r = json.loads(r)
            # ... 解析逻辑
        elif isinstance(r, dict):
            logger.info("Input is dictionary, using directly")
            # ... 直接使用
        else:
            logger.error(f"Unexpected input type: {type(r)}")
            return None

        logger.info(r["facts"])
        logger.info(r["semantic"])

        if r["facts"] and r["semantic"]:
            return {"facts": r["facts"], "semantic": r["semantic"]}
        else:
            return {"facts": {}, "semantic": []}
