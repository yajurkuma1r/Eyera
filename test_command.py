from app.services.audio.command_service import CommandService


service = CommandService()


test_cases = [

    # Menu
    ("read the menu", "READ_MENU"),
    ("read this menu", "READ_MENU"),
    ("what is on the menu", "READ_MENU"),
    ("read today's menu", "READ_MENU"),

    # Book / Document
    ("read this book", "READ_TEXT"),
    ("read this page", "READ_TEXT"),
    ("read this document", "READ_TEXT"),
    ("read this", "READ_TEXT"),
    ("continue reading", "CONTINUE_READING"),
    ("stop reading", "STOP_READING"),

    # Signs
    ("read this sign", "READ_SIGN"),
    ("what does this sign say", "READ_SIGN"),

    # Labels
    ("read this label", "READ_LABEL"),
    ("what does this label say", "READ_LABEL"),
    ("read the product label", "READ_LABEL"),

    # Medicine
    ("read this medicine", "READ_MEDICINE"),
    ("what does this medicine say", "READ_MEDICINE"),
    ("read the medicine instructions", "READ_MEDICINE"),

    # Objects
    ("what is this", "DESCRIBE_OBJECT"),
    ("what is it", "DESCRIBE_OBJECT"),
    ("identify this", "DESCRIBE_OBJECT"),
    ("describe this object", "DESCRIBE_OBJECT"),
    ("what am i looking at", "DESCRIBE_OBJECT"),

    # Scene
    ("what is around me", "DESCRIBE_SCENE"),
    ("what's around me", "DESCRIBE_SCENE"),
    ("describe my surroundings", "DESCRIBE_SCENE"),
    ("what do you see", "DESCRIBE_SCENE"),

    # Navigation
    ("start navigation", "START_NAVIGATION"),
    ("start navigating", "START_NAVIGATION"),
    ("stop navigation", "STOP_NAVIGATION"),
    ("guide me", "START_NAVIGATION"),
    ("where am i", "WHERE_AM_I"),

    # Safety
    ("is there an obstacle", "CHECK_OBSTACLE"),
    ("is there anything in front of me", "CHECK_OBSTACLE"),
    ("is it safe to move", "CHECK_OBSTACLE"),
    ("is there a car nearby", "CHECK_OBSTACLE"),

    # Specific information
    ("what color is this", "GET_COLOR"),
    ("what colour is this", "GET_COLOR"),
    ("how many people are there", "COUNT_PEOPLE"),
    ("how many objects are around me", "COUNT_OBJECTS"),
    ("is this a chair", "CONFIRM_OBJECT"),

    # Assistant
    ("hey eyera", "WAKE_WORD"),
    ("hello eyera", "WAKE_WORD"),
    ("are you there", "CHECK_PRESENCE"),
    ("repeat that", "REPEAT_LAST"),
    ("say that again", "REPEAT_LAST"),

    # Unknown
    ("tell me a joke", "UNKNOWN"),
    ("play some music", "UNKNOWN"),
    ("what is the weather", "UNKNOWN"),
]


print("==============================================")
print("EYERA COMMAND SERVICE TEST")
print("==============================================")

passed = 0
failed = 0

for phrase, expected in test_cases:

    result = service.process_command(phrase)

    if result == expected:
        print(f"[PASS] {phrase} -> {result}")
        passed += 1
    else:
        print(
            f"[FAIL] {phrase} -> "
            f"Expected: {expected}, Got: {result}"
        )
        failed += 1


print("\n==============================================")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print("==============================================")