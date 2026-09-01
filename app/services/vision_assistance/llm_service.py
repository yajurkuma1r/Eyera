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
        if not visual_context:
            system_prompt = (
                "You are Eyera, a helpful voice-based AI assistant for smart glasses.\n"
                "Answer the user's question accurately and concisely.\n"
                "The question may be about general knowledge, daily information, "
                "science, geography, history, technology, or any other topic.\n"
                "Use your general knowledge when answering general questions.\n"
                "Do not assume that the question is about the camera.\n"
                "Keep the response natural and suitable for spoken audio.\n"
                "Do not use markdown, bullet points, or special characters.\n"
            )
        else:
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
                f"User question: {user_query}\n\n"
                f"Please answer the question concisely and accurately in a natural conversational tone."
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
                print(f"[LLM] API call notice ({e}). Using local response synthesizer.")
                if visual_context:
                    return self._synthesize_vision_response(user_query, visual_context)
                else:
                    return self._synthesize_general_response(user_query)

        if visual_context:
            return self._synthesize_vision_response(user_query, visual_context)
        else:
            return self._synthesize_general_response(user_query)

    def _synthesize_general_response(self, user_query: str) -> str:
        """
        Synthesizes a general knowledge response when no visual perception is needed.
        Used when LLM API is unavailable or in local synthesizer / mock mode.
        """
        query = user_query.strip().lower()
        cleaned_query = re.sub(r"[^\w\s]", "", query)

        # Common knowledge domains for voice assistant
        knowledge_base = {
            "capital of france": "The capital of France is Paris.",
            "capital of japan": "The capital of Japan is Tokyo.",
            "capital of germany": "The capital of Germany is Berlin.",
            "capital of italy": "The capital of Italy is Rome.",
            "capital of india": "The capital of India is New Delhi.",
            "capital of the united states": "The capital of the United States is Washington, D.C.",
            "capital of usa": "The capital of the United States is Washington, D.C.",
            "capital of uk": "The capital of the United Kingdom is London.",
            "capital of canada": "The capital of Canada is Ottawa.",
            "capital of australia": "The capital of Australia is Canberra.",
            "who invented the telephone": "Alexander Graham Bell is widely credited with inventing the telephone in 1876.",
            "who invented telephone": "Alexander Graham Bell is credited with inventing the telephone.",
            "who invented the light bulb": "Thomas Edison is commonly recognized for creating the first practical incandescent light bulb.",
            "who invented the airplane": "The Wright brothers, Orville and Wilbur Wright, invented and flew the first successful motor-operated airplane.",
            "what is machine learning": "Machine learning is a branch of artificial intelligence where algorithms learn patterns from data to make predictions or decisions without being explicitly programmed.",
            "what is artificial intelligence": "Artificial intelligence refers to computer systems designed to perform tasks that typically require human intelligence, such as visual perception, speech recognition, and decision making.",
            "why is the sky blue": "The sky appears blue because gases in Earth's atmosphere scatter shorter blue wavelengths of sunlight more than other colors, a phenomenon known as Rayleigh scattering.",
            "why is ocean blue": "The ocean is blue because water absorbs longer red wavelengths of light and reflects shorter blue light.",
            "how do airplanes fly": "Airplanes fly by generating lift through their wings as air moves faster over the curved top surface than under the bottom, following Bernoulli's principle and Newton's laws of motion.",
            "speed of light": "The speed of light in a vacuum is approximately 299,792 kilometers per second, or about 186,282 miles per second.",
            "largest planet": "Jupiter is the largest planet in our solar system.",
        }

        # Exact or substring match in knowledge base
        for key, answer in knowledge_base.items():
            if key in cleaned_query:
                return answer

        # Structured Q&A heuristics for general questions
        if "capital of" in cleaned_query:
            match = re.search(r"capital of\s+([a-zA-Z\s]+)", cleaned_query)
            if match:
                country = match.group(1).strip().title()
                return f"I can look up the capital of {country} for you."

        if cleaned_query.startswith("who invented") or cleaned_query.startswith("who created"):
            match = re.search(r"who (?:invented|created)\s+(?:the\s+)?([a-zA-Z\s]+)", cleaned_query)
            if match:
                item = match.group(1).strip()
                return f"The invention of the {item} is an important historical milestone."

        if cleaned_query.startswith("what is") or cleaned_query.startswith("whats"):
            match = re.search(r"(?:what is|whats)\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z\s]+)", cleaned_query)
            if match:
                topic = match.group(1).strip()
                return f"{topic.capitalize()} is a broad topic with many facets across science and technology."

        if cleaned_query.startswith("why is") or cleaned_query.startswith("why are"):
            return "That is governed by natural scientific laws and physical principles."

        if cleaned_query.startswith("how does") or cleaned_query.startswith("how do"):
            return "It works through a coordinated combination of physical and mechanical principles."

        return f"Regarding your question about '{user_query}', I am ready to provide helpful information."

    def _synthesize_vision_response(self, user_query: str, visual_context: str) -> str:
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
