"""
LLM Service using OpenRouter.

This module provides async LLM completions using the OpenRouter API,
which provides access to various models including Claude.
"""

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMService:
    """
    OpenRouter-based LLM service for async completions.

    This service handles:
    - Async API calls with rate limiting
    - Model selection
    - Error handling and retries
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3-haiku",
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        """
        Initialize the LLM service.

        Args:
            api_key: OpenRouter API key.
            model: Default model to use.
            base_url: OpenRouter API base URL.
            max_retries: Maximum number of retries for failed requests.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout

        # Rate limiting
        self._semaphore = asyncio.Semaphore(10)  # Max concurrent requests
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # Minimum seconds between requests

    async def _get_client(self) -> httpx.AsyncClient:
        """Get an HTTP client configured for OpenRouter."""
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/joonspk-research/generative_agents",
                "X-Title": "Generative Agents",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The user prompt.
            system: Optional system message.
            model: Model to use (defaults to instance model).
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in response.
            stop: Stop sequences.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            The generated text.

        Raises:
            Exception: If the API call fails after retries.
        """
        model = model or self.model

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        if stop:
            payload["stop"] = stop

        async with self._semaphore:
            for attempt in range(self.max_retries):
                try:
                    # Rate limiting
                    now = asyncio.get_event_loop().time()
                    time_since_last = now - self._last_request_time
                    if time_since_last < self._min_request_interval:
                        await asyncio.sleep(self._min_request_interval - time_since_last)

                    async with await self._get_client() as client:
                        response = await client.post("/chat/completions", json=payload)
                        self._last_request_time = asyncio.get_event_loop().time()

                        if response.status_code == 429:
                            # Rate limited, wait and retry
                            retry_after = float(response.headers.get("Retry-After", 1))
                            logger.warning(f"Rate limited, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue

                        response.raise_for_status()
                        data = response.json()

                        if "choices" in data and len(data["choices"]) > 0:
                            return data["choices"][0]["message"]["content"]

                        logger.error(f"Unexpected response format: {data}")
                        raise ValueError("Invalid response format from API")

                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise

                except httpx.RequestError as e:
                    logger.error(f"Request error: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise

        raise RuntimeError("Failed to get completion after all retries")

    async def complete_structured(
        self,
        prompt: str,
        output_format: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> str:
        """
        Generate a structured completion with format guidance.

        This adds format instructions to encourage structured output.

        Args:
            prompt: The user prompt.
            output_format: Description of expected output format.
            system: Optional system message.
            model: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            **kwargs: Additional parameters.

        Returns:
            The generated text.
        """
        format_prompt = f"{prompt}\n\nOutput format: {output_format}"
        return await self.complete(
            prompt=format_prompt,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def generate_importance(
        self,
        description: str,
        agent_identity: str,
    ) -> float:
        """
        Generate an importance/poignancy score for an event.

        Args:
            description: Description of the event.
            agent_identity: Summary of the agent's identity.

        Returns:
            Importance score from 1-10.
        """
        prompt = f"""On a scale of 1 to 10, where 1 is purely mundane
(e.g., brushing teeth, making bed) and 10 is extremely poignant
(e.g., a break up, college acceptance), rate the likely poignancy
of the following event for the agent.

Agent:
{agent_identity}

Event: {description}

Rate the poignancy of this event (respond with just a number 1-10):"""

        response = await self.complete(
            prompt=prompt,
            temperature=0.3,
            max_tokens=10,
        )

        try:
            # Extract the first number from the response
            import re
            numbers = re.findall(r"\d+", response)
            if numbers:
                score = min(10, max(1, int(numbers[0])))
                return float(score)
        except (ValueError, IndexError):
            pass

        return 5.0  # Default to medium importance

    async def generate_keywords(
        self,
        description: str,
        agent_name: str,
    ) -> set[str]:
        """
        Generate keywords for a memory description.

        Args:
            description: The event/thought description.
            agent_name: Name of the agent.

        Returns:
            Set of keywords.
        """
        prompt = f"""Extract the key nouns and proper nouns from this description.
Return them as a comma-separated list.

Description: {description}

Keywords:"""

        response = await self.complete(
            prompt=prompt,
            temperature=0.3,
            max_tokens=100,
        )

        keywords = {kw.strip().lower() for kw in response.split(",")}
        keywords.add(agent_name.lower())
        return keywords


class MockLLMService:
    """
    Mock LLM service for testing without an API key.

    Returns simple predefined responses to allow simulation testing.
    """

    def __init__(self) -> None:
        logger.warning("Using MockLLMService - set OPENROUTER_API_KEY for real LLM")

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stop: list[str] | None = None,
        **kwargs,
    ) -> str:
        """Return mock completions based on prompt content."""
        prompt_lower = prompt.lower()

        # Daily schedule
        if "schedule" in prompt_lower:
            return """
- Wake up and get ready (60)
- Have breakfast at home (30)
- Walk to work (15)
- Work at office (180)
- Lunch break (60)
- Continue working (180)
- Walk home (15)
- Relaxation time (90)
- Have dinner (45)
- Evening activities (120)
- Get ready for bed (30)
"""

        # Location selection - vary based on activity
        if "location" in prompt_lower or "which" in prompt_lower:
            import re
            # Extract activity from prompt
            activity = ""
            wants_match = re.search(r"wants to:\s*([^\n]+)", prompt, re.IGNORECASE)
            if wants_match:
                activity = wants_match.group(1).lower()

            # Map activities to locations
            if any(word in activity for word in ["sleep", "wake", "bed", "rest"]):
                # Return home
                match = re.search(r"home is:\s*([^\n]+)", prompt, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            elif any(word in activity for word in ["work", "office", "study", "research"]):
                return "the Ville:Oak Hill College:library"
            elif any(word in activity for word in ["breakfast", "lunch", "dinner", "eat", "food", "cafe"]):
                return "the Ville:Hobbs Cafe:cafe"
            elif any(word in activity for word in ["walk", "exercise", "jog", "park"]):
                return "the Ville:Johnson Park:park"
            elif any(word in activity for word in ["shop", "buy", "groceries"]):
                return "the Ville:The Willows Market and Pharmacy:store"
            elif any(word in activity for word in ["relax", "evening", "hobby"]):
                return "the Ville:artist's co-living space:common room"

            # Default to a public location
            return "the Ville:Johnson Park:park"

        # Emoji - return activity-appropriate emoji
        if "emoji" in prompt_lower:
            if any(word in prompt_lower for word in ["sleep", "bed", "wake", "rest"]):
                return "😴"
            elif any(word in prompt_lower for word in ["eat", "breakfast", "lunch", "dinner", "food", "cafe"]):
                return "🍽️"
            elif any(word in prompt_lower for word in ["work", "office", "study", "research", "library"]):
                return "💼"
            elif any(word in prompt_lower for word in ["walk", "exercise", "jog", "park"]):
                return "👣"
            elif any(word in prompt_lower for word in ["talk", "chat", "conversation"]):
                return "💬"
            elif any(word in prompt_lower for word in ["shop", "buy", "store"]):
                return "🛒"
            elif any(word in prompt_lower for word in ["relax", "hobby", "evening"]):
                return "😌"
            elif any(word in prompt_lower for word in ["ready", "routine", "prepare"]):
                return "🧹"
            return "💭"

        # Default
        return "doing something interesting"

    async def generate_importance(
        self, description: str, agent_identity: str
    ) -> float:
        """Return a mock importance score."""
        # Simple heuristic: longer descriptions are more important
        return min(10.0, max(1.0, len(description) / 20.0))

    async def generate_keywords(
        self, description: str, agent_name: str
    ) -> set[str]:
        """Return mock keywords from the description."""
        words = description.lower().split()
        keywords = {w for w in words if len(w) > 3}
        keywords.add(agent_name.lower())
        return keywords


# Global singleton instance
_llm_service: LLMService | MockLLMService | None = None


def get_llm_service() -> LLMService | MockLLMService:
    """
    Get the global LLM service instance.

    Returns a MockLLMService if no API key is configured.
    This should be called after Django settings are loaded.
    """
    global _llm_service

    if _llm_service is None:
        from django.conf import settings

        if settings.OPENROUTER_API_KEY:
            _llm_service = LLMService(
                api_key=settings.OPENROUTER_API_KEY,
                model=settings.OPENROUTER_MODEL,
                base_url=settings.OPENROUTER_BASE_URL,
            )
        else:
            _llm_service = MockLLMService()

    return _llm_service
