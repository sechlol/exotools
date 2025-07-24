from dataclasses import dataclass


@dataclass
class DownloadParams:
    url: str
    download_path: str
