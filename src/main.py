import json

from utils import GDriveDownloader
# from utils import SpatialProjection
# from utils import preposses_data


# ... config
config_file = open("config/config.json")
config = json.load(config_file)
config_file.close()

downloader = GDriveDownloader()
downloader.download(
    fid=config["dataset_id"],
    destination=config["data_dir"]
)
