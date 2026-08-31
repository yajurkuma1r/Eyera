from app.services.audio.voice_controller import VoiceController


controller = VoiceController()

print("==============================================")
print("EYERA VOICE CONTROLLER TEST")
print("==============================================")

command, action = controller.run_once()

print(f"[Final Command] {command}")
print(f"[Final Action]  {action}")