from __future__ import annotations

import os
from typing import Literal, Protocol, runtime_checkable

from rag.config import LLM_PROVIDER

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from yandex_cloud_ml_sdk import YCloudML

LLMProvider = Literal["yandex", "openai"]


@runtime_checkable
class ChatLLM(Protocol):
    def invoke(self, input_: str | list[SystemMessage | HumanMessage]) -> str: ...


class YandexGPT:
    def __init__(
        self,
        folder_id: str,
        api_key: str,
        model_name: str = "yandexgpt",
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ):
        self.sdk = YCloudML(folder_id=folder_id, auth=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _convert_messages(messages: list[SystemMessage | HumanMessage]) -> list[dict[str, str]]:
        converted = []
        for message in messages:
            if isinstance(message, SystemMessage):
                converted.append({"role": "system", "text": str(message.content)})
            elif isinstance(message, HumanMessage):
                converted.append({"role": "user", "text": str(message.content)})
        return converted

    def invoke(self, input_: str | list[SystemMessage | HumanMessage]) -> str:
        if isinstance(input_, str):
            messages = [{"role": "user", "text": input_}]
        else:
            messages = self._convert_messages(input_)

        result = (
            self.sdk.models.completions(self.model_name)
            .configure(temperature=self.temperature, max_tokens=self.max_tokens)
            .run(messages)
        )
        if result:
            return result[0].text
        return "Нет ответа"


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ):
        self.model_name = model_name
        self._client = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def invoke(self, input_: str | list[SystemMessage | HumanMessage]) -> str:
        response = self._client.invoke(input_)
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [block.get("text", "") for block in content if isinstance(block, dict)]
            return "".join(parts).strip() or "Нет ответа"
        return str(content) if content is not None else "Нет ответа"


def build_llm() -> ChatLLM:
    if LLM_PROVIDER == "yandex":
        folder_id = os.getenv("YANDEX_FOLDER_ID")
        api_key = os.getenv("YANDEX_API_KEY")
        if not folder_id or not api_key:
            raise ValueError("Добавьте YANDEX_FOLDER_ID и YANDEX_API_KEY в .env")
        model_name = os.getenv("YANDEX_MODEL_NAME", "yandexgpt")
        return YandexGPT(folder_id, api_key, model_name=model_name)

    if LLM_PROVIDER == "openai":
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL_NAME")
        if not base_url or not api_key or not model_name:
            raise ValueError(
                "Добавьте OPENAI_BASE_URL, OPENAI_API_KEY и OPENAI_MODEL_NAME в .env"
            )
        return OpenAICompatibleLLM(base_url, api_key, model_name)

    raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}. Use 'yandex' or 'openai'.")
