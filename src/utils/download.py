import os
import tarfile
import requests

from rich.status import Status
from .dir_utils import clean_dir


class GDriveDownloader:
    def __init__(self, chunk_size: int = 32768):
        self.chunk_size = chunk_size
        self.url = "https://docs.google.com/uc?export=download"
        self.status = Status("processing ... ", spinner="dots9")

    @staticmethod
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(
        self,
        response: requests.Response,
        destination: os.PathLike
    ):
        with open(destination, "wb") as f:
            content = response.iter_content(self.chunk_size)
            for chunk in content:
                if chunk:
                    f.write(chunk)

    def download_file(
        self,
        fid: str,
        destination: os.PathLike
    ):
        session = requests.Session()

        response = session.get(
            url=self.url,
            params={'id': fid},
            stream=True
        )
        token = GDriveDownloader.get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(
                url=self.url,
                params=params,
                stream=True
            )

        self.save_response_content(response, destination)

    def download(
        self,
        fid: str,
        destination: os.PathLike
    ):
        self.status.start()
        self.status.update(" cleaning already existing files ... ")
        clean_dir(destination)

        self.status.update(" downloading data ... ")
        filename = os.path.join(destination, 'dataset.tar.xz')
        try:
            self.download_file(fid, filename)
        except:
            raise OSError("download failed!")

        self.status.update(" extracting the dataset ... ")
        try:
            tar = tarfile.open(filename)
            tar.extractall(destination)
            tar.close()
        except:
            raise OSError("something went wrong!")

        self.status.stop()
