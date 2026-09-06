from ai.base_provider import BaseAIProvider
from config.settings import get_secret, GEMINI_API_KEY_NAME, GEMINI_MODEL


class GeminiProvider(BaseAIProvider):
    name = "gemini"

    def __init__(self):
        # ============================================================
        # 🔑 API KEY LOADED HERE — actual value comes from Colab Secrets /
        # Streamlit Secrets, whichever environment this runs in.
        # ============================================================
        self.api_key = get_secret(GEMINI_API_KEY_NAME)
        self._client = None
        self._client_ready = False

        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                self._client_ready = True
            except Exception:
                self._client_ready = False

    def is_configured(self) -> bool:
        return self._client_ready

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024) -> str:
        if not self._client_ready:
            raise RuntimeError("Gemini not configured (missing/invalid API key)")

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            max_output_tokens=max_tokens,
        )
        response = self._client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )
        if not getattr(response, "text", None):
            raise RuntimeError("Gemini returned an empty response")
        return response.text
