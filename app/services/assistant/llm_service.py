import os
import re
from typing import Optional


class LLMService:
    """
    Handles communication with OpenAI LLM with intelligent offline fallback.
    """

    def __init__(self, model: str = "gpt-4o-mini", mock: bool = False):
        self.model = model
        self.mock = mock or os.getenv("MOCK_LLM", "").lower() in ("true", "1", "yes")
        self.client = None

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and not self.mock:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except Exception as e:
                print(f"[LLMService] Failed to initialize OpenAI client ({e}). Using intelligent fallback.")
                self.client = None
        else:
            print("[LLMService] Running in intelligent fallback mode (No active API key or MOCK_LLM enabled).")

    def generate_response(
        self,
        user_query: str,
        visual_context: str = ""
    ) -> str:
        """
        Generates a concise spoken response for Eyera smart glasses.
        """
        system_prompt = (
            "You are Eyera, a voice-based AI assistant for smart glasses.\n"
            "Your job is to help the user understand their surroundings.\n"
            "Rules:\n"
            "- Keep responses concise because they will be spoken through an earpiece.\n"
            "- Use natural spoken language.\n"
            "- Do not use markdown.\n"
            "- Never invent visual information.\n"
            "- Only use visual information provided by the vision system.\n"
            "- If visual information is unavailable, say so clearly.\n"
            "- For safety-related situations, state the important warning first."
        )

        if visual_context:
            user_input = (
                f"User request:\n{user_query}\n\n"
                f"Visual information from Eyera's vision system:\n{visual_context}\n\n"
                f"Answer the user's request using only the relevant visual information."
            )
        else:
            user_input = (
                f"User request:\n{user_query}\n\n"
                f"No visual information is currently available. Answer accordingly."
            )

        if self.client and not self.mock:
            try:
                # Try standard chat completions first
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=150
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[LLMService] OpenAI API error ({e}). Using intelligent fallback reasoning.")
                return self._generate_fallback_response(user_query, visual_context)

        return self._generate_fallback_response(user_query, visual_context)

    def _generate_fallback_response(self, user_query: str, visual_context: str) -> str:
        """
        Intelligent context-aware rule-based response generator when offline or quota exhausted.
        """
        query_lower = user_query.lower()

        if not visual_context or "no visual information" in visual_context.lower():
            return "I don't have any visual information from the camera right now. How else can I help?"

        # 1. Safety & Warnings
        if "warning" in visual_context.lower() or "approaching" in visual_context.lower() or "close" in visual_context.lower():
            if "approaching" in visual_context.lower():
                match = re.search(r"-?\s*(.+approaching[^\n]*)", visual_context, re.IGNORECASE)
                warning_text = match.group(1) if match else "Caution, an object is approaching."
                return f"Caution! {warning_text.strip('- ')}."

        # 2. Reading Intent (Menu / Sign / Text)
        if any(k in query_lower for k in ["read", "menu", "sign", "text", "what does it say", "price", "words"]):
            match = re.search(r'\[(?:Detected Text in Scene|Visible Text)\]:\s*"?([^"\n\[]+(?:\n[^"\n\[]+)*)"?', visual_context)
            if match:
                raw_text = match.group(1).strip()
                # If it's a menu or list with newlines
                lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                if len(lines) > 1:
                    first_few = ", ".join(lines[:4])
                    return f"The sign shows: {first_few}."
                return f"It reads: {raw_text}."
            return "I don't see any clear text to read in this view."

        # 3. Object / Obstacle Search (Where / Is there / What is in front)
        if any(k in query_lower for k in ["where", "find", "is there", "what is", "chair", "door", "table", "see", "front"]):
            match = re.search(r'\[(?:Detected Objects & Positions|Detected Objects)\]:\s*([^\n\[]+(?:\n[^\n\[]+)*)', visual_context)
            if match:
                objs = match.group(1).strip()
                cleaned_objs = re.sub(r'-\s*', '', objs).replace('\n', ', ')
                return f"In front of you, I see: {cleaned_objs}."
            return "I don't detect any specific objects directly in front of you."

        # 4. General query fallback
        return f"Based on what I see: {visual_context.splitlines()[0]}"
