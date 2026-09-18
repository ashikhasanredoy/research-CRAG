import json
import logging
from typing import Dict, Any, List, Optional, Generator
import requests
from app.config import settings

logger = logging.getLogger(__name__)

class OllamaClient:
    """Wrapper for Ollama API providing chat, completion, streaming, and JSON structured extraction."""

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model_name: str = settings.DEFAULT_LLM_MODEL,
        temperature: float = settings.LLM_TEMPERATURE
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        format_json: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Sends prompt to Ollama and returns completion text."""
        url = f"{self.base_url}/api/generate"
        options: Dict[str, Any] = {
            "temperature": temperature if temperature is not None else self.temperature
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": options
        }
        if system_prompt:
            payload["system"] = system_prompt
        if format_json:
            payload["format"] = "json"

        try:
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            else:
                logger.error(f"Ollama generate failed (status {resp.status_code}): {resp.text}")
                return f"[Error: Ollama returned {resp.status_code}]"
        except Exception as e:
            logger.error(f"Error contacting Ollama at {self.base_url}: {e}")
            return f"[Error: Could not connect to Ollama ({e})]"

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = 450
    ) -> Generator[str, None, None]:
        """Streams tokens from Ollama generate endpoint."""
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "10m",
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens or 450
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with requests.post(url, json=payload, stream=True, timeout=120) as resp:
                if resp.status_code == 200:
                    for line in resp.iter_lines(decode_unicode=True):
                        if line:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            yield chunk
                else:
                    yield f"[Error: Ollama returned status {resp.status_code}]"
        except Exception as e:
            yield f"[Error: Streaming connection failed ({e})]"

    def list_local_models(self) -> List[str]:
        """Lists available models currently installed in Ollama."""
        try:
            url = f"{self.base_url}/api/tags"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                models_data = resp.json().get("models", [])
                return [m["name"] for m in models_data]
            return []
        except Exception:
            return []

    def check_health(self) -> bool:
        """Checks if Ollama instance is accessible."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False
