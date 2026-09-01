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
        # ============================================================

        if (
            "read this sign" in text
            or "read that sign" in text
            or "what does this sign say" in text
            or "what does that sign say" in text
            or "read the sign" in text
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
            or "read the label" in text
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
            or "read this tablet" in text
            or "read the prescription" in text
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
            or "read this" in text
            or text == "read"
        ):
            return "READ_TEXT"

        # ============================================================
        # 6. PRICE / PRODUCT DETAILS
        # ============================================================

        if (
            "read the price" in text
            or "what is the price" in text
            or "what's the price" in text
            or "how much does this cost" in text
            or "how much is this" in text
            or "read the cost" in text
        ):
            return "READ_PRICE"

        # ============================================================
        # 7. EXPIRY / DATE ON PRODUCT
        # ============================================================

        if (
            "read the expiry" in text
            or "what is the expiry" in text
            or "what's the expiry" in text
            or "when does this expire" in text
            or "expiry date" in text
            or "expiration date" in text
            or "read the expiration" in text
        ):
            return "READ_EXPIRY"

        # ============================================================
        # 8. INGREDIENTS / NUTRITION
        # ============================================================

        if (
            "read the ingredients" in text
            or "what are the ingredients" in text
            or "what ingredients are in this" in text
            or "read the nutrition" in text
            or "nutrition information" in text
            or "read nutritional information" in text
        ):
            return "READ_INGREDIENTS"

        # ============================================================
        # 9. OBJECT IDENTIFICATION
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
        # 10. FIND OBJECT
        # ============================================================

        if (
            "where is my" in text
            or "where is the" in text
            or "where is a" in text
            or "where is an" in text
            or "where are my" in text
            or "where are the" in text
            or "where's my" in text
            or "where's the" in text
            or "where can i find" in text
            or "find my" in text
            or "find the" in text
            or "find a" in text
            or "find an" in text
            or "locate my" in text
            or "locate the" in text
            or "locate a" in text
            or "locate an" in text
            or "where can i see" in text
        ):
            return "FIND_OBJECT"

        # ============================================================
        # 11. SCENE / SURROUNDINGS
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
        # 12. NAVIGATION
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
        # 13. SAFETY / OBSTACLE CHECK
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
            or "is there something blocking me" in text
            or "is anything blocking me" in text
        ):
            return "CHECK_OBSTACLE"

        # ============================================================
        # 14. COLOR
        # ============================================================

        if (
            "what color is this" in text
            or "what colour is this" in text
            or "what color is it" in text
            or "what colour is it" in text
            or "what is the color" in text
            or "what is the colour" in text
        ):
            return "GET_COLOR"

        # ============================================================
        # 15. COUNT PEOPLE
        # ============================================================

        if (
            "how many people" in text
            or "how many persons" in text
            or "count the people" in text
            or "count people" in text
        ):
            return "COUNT_PEOPLE"

        # ============================================================
        # 16. COUNT OBJECTS
        # ============================================================

        if (
            "how many objects" in text
            or "count the objects" in text
            or "how many things are around me" in text
            or "count the objects around me" in text
        ):
            return "COUNT_OBJECTS"

        # ============================================================
        # 17. CONFIRM OBJECT
        # ============================================================

        if (
            "is this a chair" in text
            or "is this a car" in text
            or "is this a person" in text
            or "is this a table" in text
            or "is this a bottle" in text
            or "is this a door" in text
        ):
            return "CONFIRM_OBJECT"

        # ============================================================
        # 18. DAILY LIFE / GENERAL INFORMATION
        # These commands do NOT require the camera.
        # ============================================================

        # TIME
        if (
            "what time is it" in text
            or "what's the time" in text
            or "what is the time" in text
            or "current time" in text
            or "tell me the time" in text
            or "tell me what time it is" in text
            or "time right now" in text
            or "what time" in text
        ):
            return "GET_TIME"

        # DATE
        if (
            "what is the date" in text
            or "what's the date" in text
            or "what is the date today" in text
            or "what's the date today" in text
            or "today's date" in text
            or "todays date" in text
            or "tell me today's date" in text
            or "tell me todays date" in text
            or "tell me the date" in text
            or "what date is it" in text
            or "what date is today" in text
        ):
            return "GET_DATE"

        # DAY
        if (
            "what day is it" in text
            or "what day is today" in text
            or "what's the day today" in text
            or "which day is it" in text
            or "which day is today" in text
            or "tell me the day" in text
            or "tell me what day it is" in text
        ):
            return "GET_DAY"

        # FESTIVAL / HOLIDAY
        if (
            "festival today" in text
            or "any festival" in text
            or "is today a festival" in text
            or "is it a festival" in text
            or "is today a holiday" in text
            or "is it a holiday today" in text
            or "what festival is today" in text
            or "any holiday today" in text
            or "is there a holiday today" in text
        ):
            return "GET_FESTIVAL"
        # ============================================================
        # 19. ASSISTANT INTERACTION
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

        if (
            "what can you do" in text
            or "what can you help me with" in text
            or "help me" in text
            or text == "help"
        ):
            return "HELP"

        if (
            text == "stop"
            or text == "stop eyera"
            or text == "please stop"
        ):
            return "STOP"

        if (
            text == "cancel"
            or text == "cancel that"
            or text == "never mind"
            or text == "nevermind"
        ):
            return "CANCEL"

        # ============================================================
        # 20. GENERAL KNOWLEDGE / NON-VISION QUESTIONS
        # ============================================================

        # Questions that do not require the camera are sent directly
        # to the LLM for a general factual answer.

        if (
            text.startswith("who ")
            or text.startswith("who's ")
            or text.startswith("what ")
            or text.startswith("what's ")
            or text.startswith("when ")
            or text.startswith("why ")
            or text.startswith("how ")
            or text.startswith("where ")
            or text.startswith("which ")
            or text.startswith("whose ")
            or text.startswith("whom ")
            or text.startswith("can you explain")
            or text.startswith("can you tell me")
            or text.startswith("could you explain")
            or text.startswith("could you tell me")
            or text.startswith("explain ")
            or text.startswith("tell me about")
            or text.startswith("define ")
            or text.startswith("meaning of ")
            or text.endswith("?")
        ):
            return "GENERAL_QUERY"

        return "UNKNOWN"
