import re
from typing import Any, Dict, List, Optional, Union


class FusionService:
    """
    Fusion Service for Eyera AI Assistant.

    Fuses user intent from voice queries with multimodal visual perception data
    (OCR text, detected objects, depth, approach warnings).
    Intelligently determines relevance to keep LLM context concise and accurate.
    """

    # Keyword sets for intent classification
    READING_KEYWORDS = {
        "read", "text", "menu", "sign", "written", "words", "board",
        "label", "book", "page", "price", "say", "saying", "letter",
        "letters", "document", "ingredient", "ingredients", "nutrition"
    }

    OBJECT_KEYWORDS = {
        "where", "find", "is there", "what is", "look like", "see",
        "chair", "table", "door", "bottle", "cup", "person", "car",
        "obstacle", "around", "front", "left", "right", "behind",
        "describe", "surroundings", "room", "item", "items"
    }

    SAFETY_KEYWORDS = {
        "safe", "cross", "stop", "danger", "warning", "approaching",
        "close", "hit", "walk", "clear", "traffic", "vehicle"
    }

    def __init__(self):
        pass

    def classify_intent(self, user_query: str) -> str:
        """
        Classifies the primary intent of the user's query.
        Returns one of: 'READING', 'OBJECT_SEARCH', 'SAFETY', or 'GENERAL'.
        """
        query_lower = user_query.lower()
        words = set(re.findall(r"\b\w+\b", query_lower))

        # Check safety first
        if words.intersection(self.SAFETY_KEYWORDS) or "is it safe" in query_lower:
            return "SAFETY"

        # Check reading / OCR intent
        if words.intersection(self.READING_KEYWORDS) or "what does it say" in query_lower or "read this" in query_lower:
            return "READING"

        # Check object / spatial intent
        if words.intersection(self.OBJECT_KEYWORDS) or "what do you see" in query_lower:
            return "OBJECT_SEARCH"

        return "GENERAL"

    def fuse(
        self,
        user_query: str,
        visual_data: Optional[Union[Dict[str, Any], str]] = None
    ) -> str:
        """
        Main fusion entry point.
        Takes user query and raw visual data, applies relevance filtering,
        and returns a structured visual context string for the LLM.
        """
        if not visual_data:
            return ""

        # If visual_data is already a pre-formatted string, pass it cleanly
        if isinstance(visual_data, str):
            return visual_data.strip()

        intent = self.classify_intent(user_query)

        if intent == "READING":
            return self._format_reading_context(visual_data)
        elif intent == "OBJECT_SEARCH":
            return self._format_object_context(visual_data)
        elif intent == "SAFETY":
            return self._format_safety_context(visual_data)
        else:
            return self._format_general_context(visual_data)

    def _format_reading_context(self, data: Dict[str, Any]) -> str:
        """
        Prioritizes OCR text detected in the visual frame.
        """
        ocr_text = data.get("ocr_text", "")
        if isinstance(ocr_text, list):
            ocr_text = "\n".join(str(t) for t in ocr_text if t)
        ocr_text = ocr_text.strip()

        context_parts = []
        if ocr_text:
            context_parts.append(f"[Detected Text in Scene]:\n\"{ocr_text}\"")
        else:
            context_parts.append("[Detected Text in Scene]: No legible text was detected in the camera view.")

        # Optionally append prominent objects for context
        objects = data.get("detected_objects", [])
        if objects:
            obj_summary = self._summarize_objects(objects[:3])
            context_parts.append(f"[Nearby Objects]: {obj_summary}")

        return "\n\n".join(context_parts)

    def _format_object_context(self, data: Dict[str, Any]) -> str:
        """
        Prioritizes detected objects, spatial positions, and distances.
        """
        context_parts = []
        objects = data.get("detected_objects", [])

        if objects:
            obj_lines = []
            for obj in objects:
                label = obj.get("label", obj.get("name", "Object"))
                pos = obj.get("position", "ahead")
                dist = obj.get("distance", obj.get("depth", ""))
                dist_str = f" at {dist}" if dist else ""
                obj_lines.append(f"- {label} ({pos}{dist_str})")
            context_parts.append("[Detected Objects & Positions]:\n" + "\n".join(obj_lines))
        else:
            context_parts.append("[Detected Objects]: No notable objects detected directly in view.")

        warnings = data.get("warnings", [])
        if warnings:
            warn_lines = [f"- {w.get('message', 'Warning alert')}" for w in warnings]
            context_parts.append("[Active Warnings]:\n" + "\n".join(warn_lines))

        return "\n\n".join(context_parts)

    def _format_safety_context(self, data: Dict[str, Any]) -> str:
        """
        Prioritizes safety warnings, approaching objects, and path obstacles.
        """
        context_parts = []
        warnings = data.get("warnings", [])
        objects = data.get("detected_objects", [])

        if warnings:
            warn_lines = [f"- {w.get('message', 'Warning')} (Depth/Urgency: {w.get('depth', 'High')})" for w in warnings]
            context_parts.append("[CRITICAL SAFETY WARNINGS]:\n" + "\n".join(warn_lines))

        if objects:
            obj_lines = []
            for obj in objects:
                label = obj.get("label", obj.get("name", "Object"))
                pos = obj.get("position", "ahead")
                dist = obj.get("distance", "")
                obj_lines.append(f"- {label} located at {pos} ({dist})")
            context_parts.append("[Surrounding Objects in Path]:\n" + "\n".join(obj_lines))

        if not warnings and not objects:
            context_parts.append("[Safety Assessment]: Path appears clear of immediate detected obstacles.")

        return "\n\n".join(context_parts)

    def _format_general_context(self, data: Dict[str, Any]) -> str:
        """
        Provides a balanced multimodal overview (objects + text + warnings).
        """
        context_parts = []

        warnings = data.get("warnings", [])
        if warnings:
            warn_lines = [f"- {w.get('message', 'Warning')}" for w in warnings]
            context_parts.append("[Active Warnings]:\n" + "\n".join(warn_lines))

        objects = data.get("detected_objects", [])
        if objects:
            obj_summary = self._summarize_objects(objects)
            context_parts.append(f"[Detected Objects]: {obj_summary}")

        ocr_text = data.get("ocr_text", "")
        if isinstance(ocr_text, list):
            ocr_text = "\n".join(str(t) for t in ocr_text if t)
        if ocr_text:
            cleaned_snippet = ocr_text.strip()[:200]
            context_parts.append(f"[Visible Text]: \"{cleaned_snippet}\"")

        return "\n\n".join(context_parts) if context_parts else "No notable visual features in view."

    def _summarize_objects(self, objects: List[Dict[str, Any]]) -> str:
        items = []
        for obj in objects:
            label = obj.get("label", obj.get("name", "object"))
            pos = obj.get("position", "")
            dist = obj.get("distance", "")
            details = []
            if pos:
                details.append(pos)
            if dist:
                details.append(str(dist))
            detail_str = f" ({', '.join(details)})" if details else ""
            items.append(f"{label}{detail_str}")
        return ", ".join(items)
