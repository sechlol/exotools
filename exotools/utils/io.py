from pathlib import Path
from typing import Optional


def get_file_paths_in_subfolder(
    parent_path: Path,
    file_extension: Optional[str] = None,
    match_name: Optional[str] = None,
) -> dict[int, list[Path]]:
    subfolder_dict = {}
    if not file_extension and not match_name:
        raise ValueError("At least one between file_extension and match_name should be given")
    pattern = match_name if match_name else f"*.{file_extension}"

    # Iterate over each subfolder
    for subfolder in parent_path.iterdir():
        if subfolder.is_dir():
            fits_files = list(subfolder.glob(pattern))
            if fits_files:
                subfolder_dict[int(subfolder.name)] = [Path(file) for file in fits_files]

    return subfolder_dict
