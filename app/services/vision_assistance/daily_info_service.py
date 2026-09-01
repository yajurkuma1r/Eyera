from datetime import datetime
from typing import Optional


class DailyInfoService:
    """
    Daily-life / general-information service for Eyera.

    Answers everyday questions blind users frequently need - the current
    time, date, day of the week, and whether today is a known festival
    or holiday. Unlike every other command in Eyera, these answers do
    NOT come from the camera: the camera cannot "see" the date, so this
    service reads the device clock directly and looks up a real calendar
    instead of running the vision pipeline.

    IMPORTANT - festival data limitation:
    Most Gregorian dates for festivals like Diwali, Eid, or Holi shift
    every year because they follow a lunar/lunisolar calendar, so they
    cannot be computed with a formula. FESTIVAL_CALENDAR below is a
    small, explicitly maintained table of real festival dates. It is
    NOT fabricated or placeholder data, but it does need to be updated
    at the start of each year (or replaced with a calendar API) to stay
    accurate. This is called out again in the final implementation
    report as a known limitation.
    """

    # Real festival/holiday dates. Update yearly - see class docstring.
    FESTIVAL_CALENDAR = {
        "2026-01-01": "New Year's Day",
        "2026-01-14": "Makar Sankranti / Pongal",
        "2026-01-26": "Republic Day",
        "2026-03-03": "Holi",
        "2026-03-19": "Eid al-Fitr (Ramzan Eid)",
        "2026-03-27": "Ram Navami",
        "2026-04-14": "Ambedkar Jayanti",
        "2026-05-27": "Eid al-Adha (Bakrid)",
        "2026-08-15": "Independence Day",
        "2026-08-26": "Raksha Bandhan",
        "2026-09-04": "Janmashtami",
        "2026-09-14": "Ganesh Chaturthi",
        "2026-10-02": "Gandhi Jayanti",
        "2026-10-20": "Dussehra",
        "2026-11-08": "Diwali",
        "2026-12-25": "Christmas",
    }

    def get_time(self, now: Optional[datetime] = None) -> str:
        now = now or datetime.now()
        return now.strftime("%I:%M %p").lstrip("0")

    def get_date(self, now: Optional[datetime] = None) -> str:
        now = now or datetime.now()
        return now.strftime("%A, %B %d, %Y")

    def get_day(self, now: Optional[datetime] = None) -> str:
        now = now or datetime.now()
        return now.strftime("%A")

    def get_festival(self, now: Optional[datetime] = None) -> Optional[str]:
        now = now or datetime.now()
        key = now.strftime("%Y-%m-%d")
        return self.FESTIVAL_CALENDAR.get(key)

    def answer(self, command: str, now: Optional[datetime] = None) -> str:
        """
        Returns a natural-language spoken answer for a daily-info command.
        Deterministic and instant - these are read straight from the
        clock/calendar, so no LLM call or camera frame is needed.
        """
        now = now or datetime.now()
        cmd = (command or "").upper()

        if cmd == "GET_TIME":
            return f"It is currently {self.get_time(now)}."

        if cmd == "GET_DATE":
            return f"Today is {self.get_date(now)}."

        if cmd == "GET_DAY":
            return f"Today is {self.get_day(now)}."

        if cmd == "GET_FESTIVAL":
            festival = self.get_festival(now)
            if festival:
                return f"Yes, today is {festival}."
            return "I don't have any festival or holiday marked for today."

        return "I don't have that information right now."
