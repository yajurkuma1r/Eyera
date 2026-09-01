import os
import re
from typing import Optional


class LLMService:
    """
    LLM reasoning service for Eyera Smart Glasses.
    Interprets live visual perception data and generates concise, factual,
    natural spoken responses.
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
                print(f"[LLM] OpenAI init notice ({e}). Using live local response synthesizer.")
                self.client = None
        else:
            print("[LLM] Active in live response synthesizer mode.")

    def generate_response(
        self,
        user_query: str,
        visual_context: str = ""
    ) -> str:
        """
        Generates a concise spoken response based strictly on actual visual facts.
        """
        system_prompt = (
            "You are Eyera, a live voice-based AI assistant for smart glasses.\n"
            "Your job is to help the user understand what the camera ACTUALLY sees.\n\n"
            "Rules:\n"
            "- Answer concisely because your response will be spoken through an earpiece.\n"
            "- Use natural, conversational spoken language.\n"
            "- Do not use markdown, bullet points, or special characters.\n"
            "- NEVER invent, assume, or hallucinate visual information.\n"
            "- ONLY state visual facts provided in the visual perception input.\n"
            "- If the visual perception states that no text or no objects were detected, state that clearly.\n"
            "- If safety warnings exist, state the warning first.\n"
        )

        if visual_context:
            user_input = (
                f"User command: {user_query}\n\n"
                f"Current visual observation from camera:\n{visual_context}\n\n"
                f"Answer the user's command using only the above factual visual observations."
            )
        else:
            user_input = (
                f"User command: {user_query}\n\n"
                f"No visual information is available from the camera. Answer accordingly."
            )

        if self.client and not self.mock:
            try:
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
                print(f"[LLM] API call notice ({e}). Synthesizing from live visual facts.")
                return self._synthesize_live_response(user_query, visual_context)

        return self._synthesize_live_response(user_query, visual_context)

    def _synthesize_live_response(self, user_query: str, visual_context: str) -> str:
        """
        Synthesizes natural spoken response directly from the factual visual observations.
        Does NOT use hardcoded content; reads directly from whatever the camera provided.
        """
        if not visual_context or "no visual perception" in visual_context.lower() or "camera frame unavailable" in visual_context.lower():
            return "I am unable to access the camera right now."

        query_lower = user_query.lower()

        # 1. Critical Safety Warnings
        if "[CRITICAL SAFETY WARNINGS]" in visual_context:
            match = re.search(r"-\s*([^\n]+)", visual_context)
            if match:
                return f"Caution! {match.group(1).strip()}."

        # 2. Text / OCR Reading
        if "[Detected Text in Camera View]" in visual_context or "[Visible Text]" in visual_context:
            match = re.search(r'\[(?:Detected Text in Camera View|Visible Text)\]:\s*(?:No readable text|"([^"]+)")', visual_context)
            if match and match.group(1):
                actual_text = match.group(1).strip()
                if "menu" in query_lower:
                    return f"The menu has {actual_text}."
                if "sign" in query_lower or "board" in query_lower:
                    return f"The sign says {actual_text}."
                return f"The text reads: {actual_text}."
            elif "no readable text" in visual_context.lower():
                return "I couldn't detect any readable text in the camera view."

        # 3. Specific Object query (e.g. "Where is the chair?", "Where is the door?")
        target_obj = None
        for word in ["chair", "door", "table", "person", "bottle", "cup", "car", "laptop", "cell phone"]:
            if word in query_lower:
                target_obj = word
                break

        if target_obj and "[Detected Objects & Spatial Positions]" in visual_context:
            match = re.search(rf"-\s*({target_obj}[^\n]*)", visual_context, re.IGNORECASE)
            if match:
                return f"I see a {match.group(1).strip()}."
            else:
                return f"I don't see any {target_obj} in front of you."

        # 4. General Object / Spatial Detection
        if "[Detected Objects & Spatial Positions]" in visual_context or "[Detected Objects]" in visual_context:
            if "no notable objects" in visual_context.lower() or "no specific objects" in visual_context.lower():
                return "I don't see any notable objects in front of you."

            items = re.findall(r"-\s*([^\n]+)", visual_context)
            if items:
                items_str = ", ".join(items[:4])
                return f"In front of you, I see {items_str}."

        cleaned_lines = [l.strip() for l in visual_context.splitlines() if l.strip() and not l.startswith("[")]
        if cleaned_lines:
            return f"I see {cleaned_lines[0]}."

        return "I analyzed the camera view, but didn't find any notable text or objects."
