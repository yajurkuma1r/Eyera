import re
from typing import Any, Dict, List, Optional, Union


class FusionService:
    """
    Multimodal Fusion & Relevance Engine for Eyera Smart Glasses.

    1. Determines required visual capabilities based on user voice query and command.
    2. Fuses live factual vision and OCR detections into concise, factual context for the LLM.
    """

    READING_COMMANDS = {
        "READ_MENU", "READ_SIGN", "READ_TEXT", "READ_LABEL", "READ_MEDICINE", "CONTINUE_READING"
    }

    READING_KEYWORDS = {
        "read", "text", "menu", "sign", "written", "words", "board",
        "label", "book", "page", "price", "say", "saying", "letter",
        "letters", "document", "ingredient", "ingredients", "nutrition"
    }

    OBJECT_COMMANDS = {
        "DESCRIBE_OBJECT", "DESCRIBE_SCENE", "GET_COLOR", "COUNT_PEOPLE",
        "COUNT_OBJECTS", "CONFIRM_OBJECT", "WHERE_AM_I"
    }

    OBJECT_KEYWORDS = {
        "where", "find", "is there", "what is", "look like", "see",
        "chair", "table", "door", "bottle", "cup", "person", "car",
        "obstacle", "around", "front", "left", "right", "behind",
        "describe", "surroundings", "room", "item", "items"
    }

    SAFETY_COMMANDS = {
        "CHECK_OBSTACLE"
    }

    SAFETY_KEYWORDS = {
        "safe", "cross", "stop", "danger", "warning", "approaching",
        "close", "hit", "walk", "clear", "traffic", "vehicle"
    }

    def __init__(self):
        pass

    def determine_requirements(self, command: str = "", user_query: str = "") -> Dict[str, Any]:
        """
        Determines which live vision models must be executed for this query.
        Returns a dict: {'need_ocr': bool, 'need_objects': bool, 'need_depth': bool, 'capability': str}
        """
        cmd = command.upper() if command else ""
        query_lower = user_query.lower()
        words = set(re.findall(r"\b\w+\b", query_lower))

        # 1. Reading request
        if cmd in self.READING_COMMANDS or words.intersection(self.READING_KEYWORDS) or "what does" in query_lower:
            return {
                "need_ocr": True,
                "need_objects": False,
                "need_depth": False,
                "capability": "OCR"
            }

        # 2. Safety / Obstacle check
        if cmd in self.SAFETY_COMMANDS or words.intersection(self.SAFETY_KEYWORDS) or "is it safe" in query_lower:
            return {
                "need_ocr": False,
                "need_objects": True,
                "need_depth": True,
                "capability": "OBJECT_DETECTION, DEPTH, SAFETY"
            }

        # 3. Object / Spatial / Room query
        if cmd in self.OBJECT_COMMANDS or words.intersection(self.OBJECT_KEYWORDS) or "what is" in query_lower or "where" in query_lower:
            return {
                "need_ocr": False,
                "need_objects": True,
                "need_depth": True,
                "capability": "OBJECT_DETECTION, DEPTH"
            }

        # 4. General query - run full understanding
        return {
            "need_ocr": True,
            "need_objects": True,
            "need_depth": True,
            "capability": "OCR, OBJECT_DETECTION, DEPTH"
        }

    def classify_intent(self, user_query: str, command: str = "") -> str:
        """
        Classifies the primary intent of the user's query.
        """
        cmd = command.upper() if command else ""
        if cmd in self.READING_COMMANDS:
            return "READING"
        if cmd in self.SAFETY_COMMANDS:
            return "SAFETY"
        if cmd in self.OBJECT_COMMANDS:
            return "OBJECT_SEARCH"

        query_lower = user_query.lower()
        words = set(re.findall(r"\b\w+\b", query_lower))

        if words.intersection(self.SAFETY_KEYWORDS) or "is it safe" in query_lower:
            return "SAFETY"
        if words.intersection(self.READING_KEYWORDS) or "what does it say" in query_lower or "what does this" in query_lower:
            return "READING"
        if words.intersection(self.OBJECT_KEYWORDS) or "what do you see" in query_lower or "where" in query_lower:
            return "OBJECT_SEARCH"

        return "GENERAL"

    def fuse(
        self,
        user_query: str,
        visual_data: Optional[Union[Dict[str, Any], str]] = None,
        command: str = ""
    ) -> str:
        """
        Fuses user query and live visual observations into a strictly factual context string.
        """
        if not visual_data:
            return "No visual perception data available from the camera."

        if isinstance(visual_data, str):
            return visual_data.strip()

        intent = self.classify_intent(user_query, command=command)

        if intent == "READING":
            return self._format_reading_context(visual_data)
        elif intent == "OBJECT_SEARCH":
            return self._format_object_context(visual_data)
        elif intent == "SAFETY":
            return self._format_safety_context(visual_data)
        else:
            return self._format_general_context(visual_data)

    def _get_ocr_text(self, data: Dict[str, Any]) -> str:
        text = data.get("ocr_text", data.get("text", ""))
        if isinstance(text, list):
            text = ", ".join(str(t) for t in text if t)
        return str(text).strip() if text else ""

    def _get_objects(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return data.get("detected_objects", data.get("objects", []))

    def _format_reading_context(self, data: Dict[str, Any]) -> str:
        ocr_text = self._get_ocr_text(data)

        if ocr_text:
            return f"[Detected Text in Camera View]:\n\"{ocr_text}\""
        else:
            return "[Detected Text in Camera View]: No readable text was detected in the current camera frame."

    def _format_object_context(self, data: Dict[str, Any]) -> str:
        context_parts = []
        objects = self._get_objects(data)

        if objects:
            obj_lines = []
            for obj in objects:
                label = obj.get("label", obj.get("name", "Object"))
                pos = obj.get("position", "ahead")
                dist = obj.get("distance", obj.get("depth", ""))
                dist_str = f" at {dist}" if dist else ""
                obj_lines.append(f"- {label} ({pos}{dist_str})")
            context_parts.append("[Detected Objects & Spatial Positions]:\n" + "\n".join(obj_lines))
        else:
            context_parts.append("[Detected Objects]: No notable objects detected in the current camera view.")

        warnings = data.get("warnings", [])
        if warnings:
            warn_lines = [f"- {w.get('message', 'Warning alert')}" for w in warnings if isinstance(w, dict)]
            if warn_lines:
                context_parts.append("[Active Proximity Warnings]:\n" + "\n".join(warn_lines))

        return "\n\n".join(context_parts)

    def _format_safety_context(self, data: Dict[str, Any]) -> str:
        context_parts = []
        warnings = data.get("warnings", [])
        objects = self._get_objects(data)

        if warnings:
            warn_lines = [f"- {w.get('message', 'Warning')}" for w in warnings if isinstance(w, dict)]
            if warn_lines:
                context_parts.append("[CRITICAL SAFETY WARNINGS]:\n" + "\n".join(warn_lines))

        if objects:
            obj_lines = []
            for obj in objects:
                label = obj.get("label", obj.get("name", "Object"))
                pos = obj.get("position", "ahead")
                dist = obj.get("distance", obj.get("depth", ""))
                dist_str = f" ({dist})" if dist else ""
                obj_lines.append(f"- {label} located at {pos}{dist_str}")
            context_parts.append("[Surrounding Objects in Path]:\n" + "\n".join(obj_lines))

        if not warnings and not objects:
            context_parts.append("[Safety Assessment]: Path appears clear of detected obstacles.")

        return "\n\n".join(context_parts)

    def _format_general_context(self, data: Dict[str, Any]) -> str:
        context_parts = []

        ocr_text = self._get_ocr_text(data)
        if ocr_text:
            context_parts.append(f"[Visible Text]: \"{ocr_text}\"")

        objects = self._get_objects(data)
        if objects:
            obj_summary = ", ".join([f"{o.get('label', o.get('name'))} ({o.get('position')})" for o in objects])
            context_parts.append(f"[Detected Objects]: {obj_summary}")

        if not context_parts:
            return "Camera frame analyzed: No notable objects or text detected in current view."

        return "\n\n".join(context_parts)
