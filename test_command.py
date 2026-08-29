from app.services.audio.command_service import CommandService


command_service = CommandService()

commands = [
    "read the menu",
    "what is this",
    "describe this",
    "what is in front of me",
    "start navigation",
    "stop navigation",
    "hello eyera"
]

for text in commands:
    result = command_service.process_command(text)
    print(f"{text} -> {result}")