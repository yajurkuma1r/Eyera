class CommandService:

    def process_command(self, text):
        """
        Convert recognized speech into a predefined command.
        """

        if not text:
            return "UNKNOWN"

        text = text.lower().strip()

        # Menu commands
        if "read the menu" in text or "read menu" in text:
            return "READ_MENU"

        # Object description commands
        if "what is this" in text or "describe this" in text:
            return "DESCRIBE_OBJECT"

        # Scene description commands
        if "what is in front of me" in text:
            return "DESCRIBE_SCENE"

        # Navigation commands
        if "start navigation" in text:
            return "START_NAVIGATION"

        if "stop navigation" in text:
            return "STOP_NAVIGATION"

        return "UNKNOWN"