from utils import GDriveDownloader


# ... config
data_dir = "../data/asl/raw/"
dataset_id = "1p0CSRb9gax0sKqdyzOYVt-BXvZ4GtrBv"

downloader = GDriveDownloader()
downloader.download(
    fid=dataset_id,
    destination=data_dir
)
