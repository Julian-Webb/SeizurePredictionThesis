from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from pandas import Timestamp, Timedelta, Series
from pyedflib import EdfReader
from pytz import timezone


@dataclass(frozen=True)
class PatientTimezone:
    """
    Represents the timezone for a patient.
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


def timezone_from_edf_annotation(annotation: list) -> str:
    # Read the right annotation and convert from bytes to str (decode)
    tz = annotation[0][2].decode()
    # It now has the format "LOCAL TIME = UTC+02h" for 2 hour offset.
    # To make it work with pandas.Timestamp, we want 'UTC+02:00'.
    tz = tz.removeprefix('LOCAL TIME = ').removesuffix('h') + ':00'
    return tz


def read_edf_time_info(edf_path: Path, is_competition_ptnt: bool) -> Tuple[Timestamp, Timestamp, Timedelta]:
    """
    :raises: AmbiguousTimeError, if the edf start can't be unambiguously localized
    :raises OSError, if the edf can't be read
    :param edf_path: The str path to the edf file
    :param is_competition_ptnt: whether this patient is in the competition dataset
    :return: start, end, duration
    """
    # Read raw info
    with EdfReader(str(edf_path)) as edf:
        start = edf.getStartdatetime()
        duration = edf.getFileDuration()
        annotation = edf.read_annotation()

    # Convert info
    tz_info = PatientTimezone.from_competition(is_competition_ptnt)
    # Competition patient's EDF headers don't contain a timezone
    tz_location = tz_info.location if is_competition_ptnt else timezone_from_edf_annotation(annotation)
    # To make the start timezone-aware, convert to patient's main timezone and remove explicit timezone info
    start = Timestamp(start, tz=tz_location)
    start = start.tz_convert(tz_info.main_timezone).tz_localize(None)
    duration = Timedelta(seconds=duration)
    end = start + duration
    # noinspection PyTypeChecker
    return start, end, duration
