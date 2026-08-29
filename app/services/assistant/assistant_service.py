from app.models.assistant_response import AssistantResponse
from app.services.assistant.llm_service import LLMService
from app.services.audio.tts_service import TTSService


class AssistantService:
    """
    Main AI Assistant service.

    Takes a user request and optional visual context,
    sends relevant information to the LLM,
    and optionally speaks the response.
    """

    def __init__(self, model: str = "gpt-5.6-luna"):
        self.llm = LLMService(model=model)
        self.tts = TTSService()

    def process(
        self,
        user_query: str,
        visual_context: str = "",
        speak: bool = True
    ) -> AssistantResponse:

        response_text = self.llm.generate_response(
            user_query=user_query,
            visual_context=visual_context
        )

        response = AssistantResponse(
            text=response_text,
            priority="NORMAL",
            should_speak=speak
        )

        if response.should_speak:
            self.tts.speak(response.text)

        return response


def main():

    assistant = AssistantService()

    print("================================")
    print("       EYERA AI ASSISTANT")
    print("================================")
    print("Type 'exit' to quit.\n")

    while True:

        user_query = input("You: ").strip()

        if user_query.lower() == "exit":
            break

        if not user_query:
            continue

        try:

            response = assistant.process(
                user_query=user_query
            )

            print(f"Eyera: {response.text}")

        except Exception as error:

            print(f"[ERROR] {error}")


if __name__ == "__main__":
    main()
