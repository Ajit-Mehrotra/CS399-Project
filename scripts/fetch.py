import os
import requests


def fetch(file_path: list[str]) -> None:
    ''' Downloads the dataset from private server '''

    print("Downloading original dataset ")
    r = requests.get('http://170.39.187.47:8101/glassdoor_reviews.csv', allow_redirects=True)
    open(os.path.join(*file_path), 'wb').write(r.content)
    print("Download complete")
