import csv
import hashlib
import io
import json
import os
import time
import urllib.parse
import urllib.request

from PIL import Image, ImageOps

BASE_DIR = "."
LIMIT = 100
MAX_SIZE = (1024, 1024)

HEADERS = {
    "User-Agent": "AppleDatasetDownloader/1.0 (educational project)"
}

os.makedirs("apple", exist_ok=True)
os.makedirs("other", exist_ok=True)

used_hashes = set()

def get_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.loads(response.read())

def download_image(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read(20 * 1024 * 1024 + 1)

    if len(data) > 20 * 1024 * 1024:
        return None

    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    image.thumbnail(MAX_SIZE)
    image = image.convert("RGB")

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()

def wikimedia_search(query):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": 50,
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": 1024
    }

    url = "https://commons.wikimedia.org/w/api.php?" + \
          urllib.parse.urlencode(params)

    try:
        data = get_json(url)
        pages = data.get("query", {}).get("pages", {})

        for page in pages.values():
            info = page.get("imageinfo", [{}])[0]
            image_url = info.get("thumburl") or info.get("url")

            if image_url:
                yield image_url, "Wikimedia Commons"

    except Exception as e:
        print("Wikimedia error:", e)

def openverse_search(query):
    for page in range(1, 6):
        params = {
            "q": query,
            "page": page,
            "page_size": 20
        }

        url = "https://api.openverse.org/v1/images/?" + \
              urllib.parse.urlencode(params)

        try:
            data = get_json(url)

            for item in data.get("results", []):
                image_url = item.get("url")

                if image_url:
                    yield (
                        image_url,
                        "Openverse: " + str(item.get("license", "unknown"))
                    )

            if not data.get("results"):
                break

        except Exception as e:
            print("Openverse error:", e)
            break

def collect(folder, queries):
    count = len(os.listdir(folder))

    metadata_path = os.path.join(BASE_DIR, "metadata.csv")

    with open(metadata_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        for query in queries:
            if count >= LIMIT:
                break

            print("\nSearching:", query)

            sources = [
                wikimedia_search(query),
                openverse_search(query)
            ]

            for source in sources:
                for url, source_name in source:
                    if count >= LIMIT:
                        break

                    try:
                        data = download_image(url)

                        if not data:
                            continue

                        image_hash = hashlib.sha256(data).hexdigest()

                        if image_hash in used_hashes:
                            continue

                        used_hashes.add(image_hash)

                        filename = f"{folder}_{count + 1:03d}.jpg"
                        filepath = os.path.join(folder, filename)

                        with open(filepath, "wb") as f:
                            f.write(data)

                        writer.writerow([
                            filename,
                            folder,
                            source_name,
                            url
                        ])
                        csvfile.flush()

                        count += 1
                        print(f"[{count}/{LIMIT}] {filepath}")

                        time.sleep(0.2)

                    except Exception as e:
                        print("Image skipped:", str(e)[:100])

            if count >= LIMIT:
                break

    print(f"\nFinished {folder}: {count} images")

if __name__ == "__main__":

    apple_queries = [
        "apple fruit",
        "red apple",
        "green apple",
        "yellow apple",
        "apple on tree"
    ]

    other_queries = [
        "pear fruit",
        "orange fruit",
        "banana fruit",
        "peach fruit",
        "tomato",
        "strawberry",
        "dog",
        "car",
        "chair"
    ]

    collect("apple", apple_queries)
    collect("other", other_queries)

    print("\nDataset download completed!")
