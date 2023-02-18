import urllib.request

from config import PathConfig
from tqdm import tqdm

if __name__ == '__main__':
    with open(PathConfig.VIDEOS_LINKS_PATH) as f:
        links = f.readlines()

    PathConfig.mkdir(PathConfig.VIDEOS_PATH)
    for link in tqdm(links):
        filename = link.split('/')[-1].strip()
        urllib.request.urlretrieve(link, f"{PathConfig.VIDEOS_PATH}/{filename}")
