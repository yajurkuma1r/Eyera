from app.services.audio.voice_controller import VoiceController


controller = VoiceController()

test_commands = [
    "READ_MENU",
    "READ_TEXT",
    "READ_SIGN",
    "READ_MEDICINE",
    "DESCRIBE_OBJECT",
    "DESCRIBE_SCENE",
    "CHECK_OBSTACLE",
    "GET_COLOR",
    "COUNT_PEOPLE",
    "START_NAVIGATION",
    "UNKNOWN"
]

print("==============================================")
print("EYERA VOICE ROUTING TEST")
print("==============================================")

for command in test_commands:
    action = controller.get_action(command)
    print(f"{command:25} -> {action}")