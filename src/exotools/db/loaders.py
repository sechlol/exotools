from .lightcurve_db import LightcurveDB


def load_lightcurve_db() -> LightcurveDB:
    downloaded_lc = get_file_paths_in_subfolder(configs.MAST_FOLDER, file_extension="fits")
    dataset = LightcurveDB.path_map_to_qtable(downloaded_lc)
    return LightcurveDB(dataset)
