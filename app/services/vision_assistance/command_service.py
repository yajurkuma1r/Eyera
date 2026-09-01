class CommandService:
    """
    Converts natural-language voice input into standardized commands.

    This service only detects USER INTENT.
    It does not perform vision, OCR, navigation, or LLM processing.
    """

    def process_command(self, text):

        if not text:
            return "UNKNOWN"

        text = text.lower().strip()

        # ============================================================
        # 1. MENU / FOOD
        # ============================================================

        if (
            "read the menu" in text
            or "read this menu" in text
            or "read menu" in text
            or "what is on the menu" in text
            or "what's on the menu" in text
            or "today's menu" in text
            or "todays menu" in text
        ):
            return "READ_MENU"

        # ============================================================
        # 2. SIGNS
        # IMPORTANT: These must come BEFORE generic "read this"
        # ============================================================

        if (
            "read this sign" in text
            or "read that sign" in text
            or "what does this sign say" in text
            or "what does that sign say" in text
        ):
            return "READ_SIGN"

        # ============================================================
        # 3. LABELS / PRODUCT INFORMATION
        # ============================================================

        if (
            "read this label" in text
            or "read that label" in text
            or "what does this label say" in text
            or "what does that label say" in text
            or "read the product label" in text
        ):
            return "READ_LABEL"

        # ============================================================
        # 4. MEDICINE
        # ============================================================

        if (
            "read this medicine" in text
            or "read that medicine" in text
            or "what does this medicine say" in text
            or "medicine instructions" in text
            or "read the medicine instructions" in text
        ):
            return "READ_MEDICINE"

        # ============================================================
        # 5. BOOK / DOCUMENT / TEXT
        # ============================================================

        if "stop reading" in text:
            return "STOP_READING"

        if (
            "continue reading" in text
            or "continue the reading" in text
            or "read from where you stopped" in text
        ):
            return "CONTINUE_READING"

        if (
            "read this book" in text
            or "read this page" in text
            or "read this document" in text
            or "read this text" in text
            or text == "read this"
        ):
            return "READ_TEXT"

        # ============================================================
        # 6. OBJECT IDENTIFICATION
        # ============================================================

        if (
            "what is this" in text
            or "what is it" in text
            or "identify this" in text
            or "identify that" in text
            or "describe this object" in text
            or "what am i looking at" in text
            or "what am i seeing" in text
        ):
            return "DESCRIBE_OBJECT"

        # ============================================================
        # 7. SCENE / SURROUNDINGS
        # ============================================================

        if (
            "what is around me" in text
            or "what's around me" in text
            or "what is in front of me" in text
            or "what's in front of me" in text
            or "describe my surroundings" in text
            or "what do you see" in text
            or "tell me what's around me" in text
            or "tell me what's ahead" in text
            or "describe the scene" in text
        ):
            return "DESCRIBE_SCENE"

        # ============================================================
        # 8. NAVIGATION
        # ============================================================

        if "stop navigation" in text:
            return "STOP_NAVIGATION"

        if (
            "start navigation" in text
            or "start navigating" in text
            or "guide me" in text
            or "help me navigate" in text
        ):
            return "START_NAVIGATION"

        if (
            "where am i" in text
            or "what is my location" in text
            or "where are we" in text
        ):
            return "WHERE_AM_I"

        # ============================================================
        # 9. SAFETY / OBSTACLE CHECK
        # ============================================================

        if (
            "is there an obstacle" in text
            or "is there anything in front of me" in text
            or "is there something in front of me" in text
            or "is it safe to move" in text
            or "is it safe to walk" in text
            or "is it safe to cross" in text
            or "is it safe ahead" in text
            or "is there a car nearby" in text
            or "is there anything ahead" in text
        ):
            return "CHECK_OBSTACLE"

        # ============================================================
        # 10. SPECIFIC VISUAL INFORMATION
        # ============================================================

        if (
            "what color is this" in text
            or "what colour is this" in text
            or "what color is it" in text
            or "what colour is it" in text
        ):
            return "GET_COLOR"

        if (
            "how many people" in text
            or "how many persons" in text
            or "count the people" in text
        ):
            return "COUNT_PEOPLE"

        if (
            "how many objects" in text
            or "count the objects" in text
            or "how many things are around me" in text
        ):
            return "COUNT_OBJECTS"

        if (
            "is this a chair" in text
            or "is this a car" in text
            or "is this a person" in text
            or "is this a table" in text
        ):
            return "CONFIRM_OBJECT"

        # ============================================================
        # 11. DAILY LIFE / GENERAL INFORMATION (no camera required)
        # ============================================================

        if (
            "what time is it" in text
            or "what's the time" in text
            or "what is the time" in text
            or "current time" in text
            or "tell me the time" in text
        ):
            return "GET_TIME"

        if (
            "what is the date" in text
            or "what's the date" in text
            or "today's date" in text
            or "todays date" in text
            or "what is today's date" in text
            or "what's today's date" in text
        ):
            return "GET_DATE"

        if (
            "what day is it" in text
            or "what day is today" in text
            or "what's the day today" in text
            or "which day is it" in text
            or "which day is today" in text
        ):
            return "GET_DAY"

        if (
            "festival today" in text
            or "any festival" in text
            or "is today a festival" in text
            or "is it a festival" in text
            or "is today a holiday" in text
            or "is it a holiday today" in text
            or "what festival is today" in text
            or "any holiday today" in text
        ):
            return "GET_FESTIVAL"

        # ============================================================
        # 12. ASSISTANT INTERACTION
        # ============================================================

        if (
            "hey eyera" in text
            or "hello eyera" in text
            or "hi eyera" in text
        ):
            return "WAKE_WORD"

        if (
            "are you there" in text
            or "are you listening" in text
        ):
            return "CHECK_PRESENCE"

        if (
            "repeat that" in text
            or "say that again" in text
            or "what did you just say" in text
        ):
            return "REPEAT_LAST"

        return "UNKNOWN"
