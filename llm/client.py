from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional, Protocol

from utils.logger import get_logger

logger = get_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    EXTRACTIVE = "extractive"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class AnswerProvider(Protocol):
    """Minimal provider contract used by the query pipeline."""

    def generate(self, context: str, query: str, max_tokens: int = 500) -> str:
        ...


ABSTENTION = "I don't have enough evidence in the indexed sources to answer that."


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class ExtractiveProvider:
    """No-key provider that quotes the highest-overlap source sentences."""

    def generate(self, context: str, query: str, max_tokens: int = 500) -> str:
        query_terms = _tokens(query)
        if not query_terms:
            return ABSTENTION
        candidates: list[tuple[int, int, str]] = []
        for index, sentence in enumerate(re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", context)):
            sentence = sentence.strip()
            sentence = re.sub(r"^\[[^\]]+\]\s*", "", sentence)
            if not sentence:
                continue
            overlap = len(query_terms & _tokens(sentence))
            if overlap:
                candidates.append((overlap, -index, sentence))
        if not candidates:
            return ABSTENTION
        candidates.sort(key=lambda item: (-item[0], -item[1]))
        answer = " ".join(item[2] for item in candidates[:2])
        words = answer.split()
        return " ".join(words[:max_tokens])


class LLMClient:
    """Provider abstraction with a deterministic extractive default.

    Network-backed providers are intentionally not configured in v0.1.0.  A
    caller can inject an object implementing :class:`AnswerProvider`; without one,
    the service always remains useful and key-free through extraction.
    """
    
    def __init__(
        self,
        provider: LLMProvider | str = LLMProvider.EXTRACTIVE,
        model_name: str = "local-extractive",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        provider_client: AnswerProvider | None = None,
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider to use
            model_name: Model identifier
            temperature: Sampling temperature
            api_key: API key (if required)
        """
        self.provider = LLMProvider(provider)
        self.model_name = model_name
        self.temperature = temperature
        # Keep only a boolean: credentials must never be logged or returned.
        self._has_api_key = bool(api_key)
        self.client = provider_client or ExtractiveProvider()
        self.effective_provider = (
            self.provider if provider_client is not None else LLMProvider.EXTRACTIVE
        )
        logger.info(
            "answer provider initialized",
            extra={
                "provider": self.effective_provider.value,
                "api_key_configured": self._has_api_key,
            },
        )

    def generate(
        self,
        context: str,
        query: str,
        max_tokens: int = 500
    ) -> str:
        """Generate from supplied context via the configured provider."""

        return self.client.generate(context, query, max_tokens=max_tokens)

    def generate_with_citations(
        self,
        chunks: list[dict[str, Any]],
        query: str,
    ) -> dict[str, Any]:
        """Generate a grounded answer and citations tied to retrieved chunks."""

        if not chunks:
            return {
                "answer": ABSTENTION,
                "citations": [],
                "provider": self.effective_provider.value,
                "abstained": True,
            }
        context = "\n".join(
            f"[{chunk.get('id', '')}] {chunk.get('text', '')}" for chunk in chunks
        )
        answer = self.generate(context, query)
        if answer == ABSTENTION:
            return {
                "answer": answer,
                "citations": [],
                "provider": self.effective_provider.value,
                "abstained": True,
            }
        query_terms = _tokens(query)
        used = [
            chunk
            for chunk in chunks
            if query_terms & _tokens(str(chunk.get("text", "")))
        ] or chunks[:1]
        citations = [self._citation(chunk) for chunk in used[:3]]
        return {
            "answer": answer,
            "citations": citations,
            "provider": self.effective_provider.value,
            "abstained": False,
        }

    @staticmethod
    def _citation(chunk: dict[str, Any]) -> dict[str, Any]:
        metadata = chunk.get("metadata") or {}
        text = str(chunk.get("text", ""))
        return {
            "document": metadata.get("document", ""),
            "source": metadata.get("source", ""),
            "chunk": chunk.get("id") or metadata.get("chunk", ""),
            "excerpt": text[:400],
        }
