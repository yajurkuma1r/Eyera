from app.services.audio.speech_service import SpeechService
from app.services.audio.command_service import CommandService
from app.services.audio.tts_service import TTSService


class VoiceController:

    def __init__(self):
        self.speech_service = SpeechService()
        self.command_service = CommandService()
        self.tts_service = TTSService()

    def listen_for_command(self):
        # 1. Listen to the user's voice
        text = self.speech_service.listen()

        # 2. Convert speech into a standardized command
        command = self.command_service.process_command(text)

        print(f"[Command] {command}")

        return command

    def speak(self, message):
        # Reuse the existing TTS service
        self.tts_service.speak(message)

    def handle_command(self, command):
        """
        Temporary command routing.

        Later, each command will connect to the appropriate
        Vision, OCR, Navigation or LLM service.
        """

        if command == "READ_MENU":
            self.speak("I will read the menu.")

        elif command == "READ_TEXT":
            self.speak("I will read the text.")

        elif command == "CONTINUE_READING":
            self.speak("Continuing reading.")

        elif command == "STOP_READING":
            self.speak("Stopping reading.")

        elif command == "READ_SIGN":
            self.speak("I will read the sign.")

        elif command == "READ_LABEL":
            self.speak("I will read the label.")

        elif command == "READ_MEDICINE":
            self.speak("I will read the medicine information.")

        elif command == "DESCRIBE_OBJECT":
            self.speak("I will describe the object.")

        elif command == "DESCRIBE_SCENE":
            self.speak("I will describe the scene.")

        elif command == "START_NAVIGATION":
            self.speak("Starting navigation.")

        elif command == "STOP_NAVIGATION":
            self.speak("Stopping navigation.")

        elif command == "WHERE_AM_I":
            self.speak("I will determine your location.")

        elif command == "CHECK_OBSTACLE":
            self.speak("I will check for obstacles.")

        elif command == "GET_COLOR":
            self.speak("I will identify the color.")

        elif command == "COUNT_PEOPLE":
            self.speak("I will count the people.")

        elif command == "COUNT_OBJECTS":
            self.speak("I will count the objects.")

        elif command == "CONFIRM_OBJECT":
            self.speak("I will check the object.")

        elif command == "WAKE_WORD":
            self.speak("Yes, I am listening.")

        elif command == "CHECK_PRESENCE":
            self.speak("Yes, I am here.")

        elif command == "REPEAT_LAST":
            self.speak("I will repeat the last response.")

        else:
            self.speak("Sorry, I did not understand the command.")
    def get_action(self, command):
        """
        Return the service/action that should handle the command.

        These are routing hooks only.
        Actual Vision, OCR, Navigation and LLM services
        will be connected later.
        """

        action_map = {

            # OCR / text reading
            "READ_MENU": "OCR_MENU",
            "READ_TEXT": "OCR_TEXT",
            "CONTINUE_READING": "OCR_CONTINUE",
            "STOP_READING": "STOP_READING",

            # Signs and labels
            "READ_SIGN": "OCR_SIGN",
            "READ_LABEL": "OCR_LABEL",
            "READ_MEDICINE": "OCR_MEDICINE",

            # Vision + LLM
            "DESCRIBE_OBJECT": "VISION_LLM_OBJECT",
            "DESCRIBE_SCENE": "VISION_LLM_SCENE",

            # Navigation
            "START_NAVIGATION": "NAVIGATION_START",
            "STOP_NAVIGATION": "NAVIGATION_STOP",
            "WHERE_AM_I": "NAVIGATION_LOCATION",

            # Safety / vision
            "CHECK_OBSTACLE": "VISION_OBSTACLE",
            "GET_COLOR": "VISION_COLOR",
            "COUNT_PEOPLE": "VISION_COUNT_PEOPLE",
            "COUNT_OBJECTS": "VISION_COUNT_OBJECTS",
            "CONFIRM_OBJECT": "VISION_CONFIRM_OBJECT",

            # Assistant
            "WAKE_WORD": "ASSISTANT_WAKE",
            "CHECK_PRESENCE": "ASSISTANT_STATUS",
            "REPEAT_LAST": "ASSISTANT_REPEAT",

            # Unknown
            "UNKNOWN": "UNKNOWN"
        }

        return action_map.get(command, "UNKNOWN")
    def run_once(self):
        """
        Complete voice interaction flow.

        Speech
        -> Command
        -> Action
        -> Temporary response through TTS

        Later, the temporary response can be replaced
        by the actual Vision/OCR/LLM service result.
        """

        # Step 1: Listen and detect command
        command = self.listen_for_command()

        # Step 2: Convert command into an action
        action = self.get_action(command)

        print(f"[Action] {action}")

        # Step 3: Temporary responses
        # These will later be replaced by actual service calls.

        responses = {
            "OCR_MENU": "I will read the menu.",
            "OCR_TEXT": "I will read the text.",
            "OCR_CONTINUE": "Continuing reading.",
            "STOP_READING": "Stopping reading.",
            "OCR_SIGN": "I will read the sign.",
            "OCR_LABEL": "I will read the label.",
            "OCR_MEDICINE": "I will read the medicine information.",

            "VISION_LLM_OBJECT": "I will describe the object.",
            "VISION_LLM_SCENE": "I will describe your surroundings.",

            "NAVIGATION_START": "Starting navigation.",
            "NAVIGATION_STOP": "Stopping navigation.",
            "NAVIGATION_LOCATION": "I will determine your location.",

            "VISION_OBSTACLE": "I will check for obstacles.",
            "VISION_COLOR": "I will identify the color.",
            "VISION_COUNT_PEOPLE": "I will count the people.",
            "VISION_COUNT_OBJECTS": "I will count the objects.",
            "VISION_CONFIRM_OBJECT": "I will check the object.",

            "ASSISTANT_WAKE": "Yes, I am listening.",
            "ASSISTANT_STATUS": "Yes, I am here.",
            "ASSISTANT_REPEAT": "I will repeat the last response.",

            "UNKNOWN": "Sorry, I did not understand the command."
        }

        response = responses.get(
            action,
            "Sorry, I do not know how to handle that command."
        )

        self.speak(response)

        return command, action