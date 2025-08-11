from pathlib import Path

# Data Folders

def get_data_folders(base_path: Path) -> tuple[Path, Path, Path]:
    """:return: for_mayo, uneeg_extended, competition_dir folders. """
    for_mayo = base_path / '20240201_UNEEG_ForMayo'
    uneeg_extended = base_path / '20250217_UNEEG_Extended'
    competition_dir = base_path / '20250501_SUBQ_SeizurePredictionCompetition_2025final'

    return for_mayo, uneeg_extended, competition_dir
