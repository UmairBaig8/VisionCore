import requests


class VLLMClient:

    def __init__(self, endpoint, model):
        self.endpoint = endpoint
        self.model = model

    def ask(self, prompt, image_b64=None):
        content = []

        if image_b64:
            images = image_b64 if isinstance(image_b64, list) else [image_b64]
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

        response = requests.post(
            self.endpoint, json=payload, timeout=300,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]
