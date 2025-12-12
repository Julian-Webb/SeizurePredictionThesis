from dataclasses import dataclass

@dataclass(frozen=True)
class PatientTimezone:
    """
    Represents the timezone for a patients.
    For example:
    location: 'Europe/Berlin'
    main_timezone: 'UTC+01:00'
    dst_timezone: 'UTC+02:00'
    """
    location: str
    main_timezone: str
    dst_timezone: str

    @classmethod
    def from_competition(cls, is_competition_ptnt: bool) -> "PatientTimezone":
        if is_competition_ptnt:
            return cls(location="Europe/London", main_timezone="UTC+00:00", dst_timezone="UTC+01:00")
        return cls(location="Europe/Berlin", main_timezone="UTC+01:00", dst_timezone="UTC+02:00")
