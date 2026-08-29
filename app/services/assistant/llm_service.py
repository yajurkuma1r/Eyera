import os

from openai import OpenAI


class LLMService:
    """
    Handles communication with the LLM.
    """

    def __init__(self, model: str = "gpt-5.6-luna"):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_response(
        self,
        user_query: str,
        visual_context: str = ""
    ) -> str:

        system_prompt = """
You are Eyera, a voice-based AI assistant for smart glasses.

Your job is to help the user understand their surroundings.

Rules:
- Keep responses concise because they will be spoken through an earpiece.
- Use natural spoken language.
- Do not use markdown.
- Never invent visual information.
- Only use objects, text, distances, or other visual information provided to you.
- If visual information is unavailable, say so clearly.
- For safety-related situations, state the important warning first.
"""

        if visual_context:
            user_input = f"""
User request:
{user_query}

Visual information from Eyera's vision system:
{visual_context}

Answer the user's request using only relevant visual information.
"""
        else:
            user_input = f"""
User request:
{user_query}

No visual information is currently available.
Answer accordingly.
"""

        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_input
        )

        return response.output_text.strip()
