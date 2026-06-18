import logging
import time

import requests

logger = logging.getLogger("vllm")


class VLLMClient:

    def __init__(self, endpoint, model):
        self.endpoint = endpoint
        self.model = model
        self.call_count = 0

    def ask(self, prompt, image_b64=None):
        self.call_count += 1
        content = []

        img_count = 0
        if image_b64:
            images = image_b64 if isinstance(image_b64, list) else [image_b64]
            img_count = len(images)
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}"
                    }
                })

        content.append({
            "type": "text",
            "text": prompt
        })

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
        }

        t0 = time.time()
        response = requests.post(
            self.endpoint, json=payload, timeout=300,
        )
        elapsed = time.time() - t0
        response.raise_for_status()

        data = response.json()
        result = data["choices"][0]["message"]["content"]
        prompt_preview = prompt[:80].replace("\n", " ") + ("..." if len(prompt) > 80 else "")
        logger.info("vllm #%d | %.1fs | %d imgs | %d chars out | %s",
                     self.call_count, elapsed, img_count, len(result), prompt_preview)
        return result
