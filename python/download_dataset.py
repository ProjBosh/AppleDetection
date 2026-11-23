import csv
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image, ImageOps

# Скрипт лежит в python/, данные — в ../data (apple/, other/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
SOURCES_PATH = os.path.join(DATA_DIR, "sources.csv")
LEGACY_SOURCES_PATH = os.path.join(DATA_DIR, "sources_legacy.csv")

LIMIT = 224                  # целевое число изображений в каждой папке
MAX_SIZE = (1024, 1024)       # ограничение C0: размер входа до 1024x1024
MIN_SIDE = 100                # слишком маленькие изображения пропускаются
MAX_BYTES = 20 * 1024 * 1024
WIKIMEDIA_MAX_PAGES = 10      # по 50 результатов на страницу
OPENVERSE_MAX_PAGES = 12      # по 20 результатов на страницу
THUMB_WIDTH = 960             # стандартный размер превью Wikimedia (не 1024!)

# Ограничение частоты запросов (секунды между запросами к одному хосту)
WIKIMEDIA_INTERVAL = 1.5
DEFAULT_INTERVAL = 0.3
MAX_INTERVAL = 10.0
MAX_RETRIES = 5
COOLDOWN_AFTER_429 = 60       # пауза, если повторы не помогли

# ВАЖНО: укажите свой реальный e-mail (требование политики Wikimedia)
HEADERS = {
    "User-Agent": "AppleDatasetDownloader/1.1 "
                  "(educational project; contact: your_email@example.com)"
}

APPLE_QUERIES = [
    "apple fruit",
    "red apple",
    "green apple",
    "yellow apple",
    "apple on tree",
    "apple orchard",
    "Granny Smith apple",
    "Fuji apple",
    "Gala apple",
    "Golden Delicious apple",
    "Red Delicious apple",
    "Honeycrisp apple",
    "Pink Lady apple",
    "apples basket",
    "apple harvest",
    "apple fruit white background",
    "apple fruit macro",
    "sliced apple",
    "ripe apple",
    "apples market",
]

OTHER_QUERIES = [
    "pear fruit",
    "orange fruit",
    "banana fruit",
    "peach fruit",
    "tomato",
    "strawberry",
    "red ball",
    "tennis ball",
    "basketball",
    "balloon",
    "pomegranate",
    "plum fruit",
    "cherry fruit",
    "grapes",
    "lemon",
    "mango fruit",
    "kiwi fruit",
    "watermelon",
    "pumpkin",
    "red bell pepper",
    "onion",
    "potato",
    "dog",
    "cat",
    "bird",
    "car",
    "bicycle",
    "chair",
    "table",
    "bottle",
    "cup",
    "book",
    "shoes",
    "laptop",
    "flower",
    "house",
    "mountain",
    "street",
    "lamp",
    "toy",
]

used_hashes = set()
used_urls = set()

_host_interval = {}
_host_last = {}


def host_key(url):
    """Все хосты Wikimedia делят один лимит."""
    host = urllib.parse.urlparse(url).netloc.lower()

    if host.endswith("wikimedia.org") or host.endswith("wikipedia.org"):
        return "wikimedia"

    return host


def base_interval(key):
    return WIKIMEDIA_INTERVAL if key == "wikimedia" else DEFAULT_INTERVAL


def throttle(key):
    """Выдерживает минимальный интервал между запросами к хосту."""
    interval = _host_interval.get(key, base_interval(key))
    wait = _host_last.get(key, 0.0) + interval - time.monotonic()

    if wait > 0:
        time.sleep(wait)

    _host_last[key] = time.monotonic()


def parse_retry_after(error):
    value = error.headers.get("Retry-After") if error.headers else None

    if value and value.strip().isdigit():
        return min(int(value.strip()), 300)

    return None


def fetch(url, timeout=30, max_bytes=None):
    """
    Загружает URL с ограничением частоты и повторами.
    При 429/503 учитывает Retry-After, увеличивает интервал для хоста
    и делает экспоненциальную паузу.
    """
    key = host_key(url)

    for attempt in range(MAX_RETRIES + 1):
        throttle(key)
        req = urllib.request.Request(url, headers=HEADERS)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if max_bytes:
                    return response.read(max_bytes + 1)

                return response.read()

        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < MAX_RETRIES:
                delay = parse_retry_after(e) or min(300, 15 * 2 ** attempt)
                new_interval = min(
                    MAX_INTERVAL,
                    _host_interval.get(key, base_interval(key)) * 1.5
                )
                _host_interval[key] = new_interval
                print(f"HTTP {e.code} ({key}): пауза {delay} с, "
                      f"интервал запросов {new_interval:.1f} с")
                time.sleep(delay)
                continue

            raise

        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < MAX_RETRIES:
                time.sleep(5 * (attempt + 1))
                continue

            raise


def get_json(url):
    return json.loads(fetch(url, timeout=20))


def download_image(url):
    """Скачивает изображение, приводит к RGB и JPEG не больше 1024x1024."""
    data = fetch(url, timeout=30, max_bytes=MAX_BYTES)

    if len(data) > MAX_BYTES:
        return None

    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)

    if min(image.size) < MIN_SIDE:
        return None

    image.thumbnail(MAX_SIZE)

    # Прозрачные области заливаются белым (как при загрузке в Java)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    else:
        image = image.convert("RGB")

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def wikimedia_search(query):
    """Поиск изображений в Wikimedia Commons с постраничной выдачей."""
    offset = 0

    for _ in range(WIKIMEDIA_MAX_PAGES):
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": 50,
            "gsroffset": offset,
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": THUMB_WIDTH
        }
        url = "https://commons.wikimedia.org/w/api.php?" + \
              urllib.parse.urlencode(params)

        try:
            data = get_json(url)
        except Exception as e:
            print("Wikimedia error:", e)
            return

        pages = data.get("query", {}).get("pages", {})

        for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime", "")

            if not mime.startswith("image/") or "svg" in mime:
                continue

            image_url = info.get("thumburl") or info.get("url")

            if image_url:
                yield image_url, "Wikimedia Commons"

        next_offset = (data.get("continue") or {}).get("gsroffset")

        if next_offset is None:
            return

        offset = next_offset


def openverse_search(query):
    """Поиск изображений в Openverse."""
    for page in range(1, OPENVERSE_MAX_PAGES + 1):
        params = {
            "q": query,
            "page": page,
            "page_size": 20
        }
        url = "https://api.openverse.org/v1/images/?" + \
              urllib.parse.urlencode(params)

        try:
            data = get_json(url)
        except Exception as e:
            print("Openverse error:", e)
            return

        results = data.get("results", [])

        if not results:
            return

        for item in results:
            image_url = item.get("url")

            if image_url:
                yield (
                    image_url,
                    "Openverse: " + str(item.get("license", "unknown"))
                )


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def migrate_legacy_metadata():
    """
    Старый metadata.csv (имя, класс, источник, url) несовместим с форматом,
    который читает Trainer (путь, класс, split). Такой файл переносится в
    data/sources_legacy.csv, чтобы не мешать обучению.
    """
    candidates = [
        os.path.join(SCRIPT_DIR, "metadata.csv"),
        os.path.join(DATA_DIR, "metadata.csv"),
    ]

    for path in candidates:
        if not os.path.isfile(path):
            continue

        with open(path, encoding="utf-8") as f:
            first_line = f.readline().strip().lower()

        if first_line.startswith("path,class,split"):
            continue

        os.replace(path, LEGACY_SOURCES_PATH)
        print("Старый файл перенесён:", path, "->", LEGACY_SOURCES_PATH)


def load_known():
    """Загружает хеши уже скачанных файлов и их URL, чтобы избежать дубликатов."""
    for folder in ("apple", "other"):
        folder_path = os.path.join(DATA_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        for name in os.listdir(folder_path):
            if name.lower().endswith(".jpg"):
                used_hashes.add(file_sha256(os.path.join(folder_path, name)))

    for path in (SOURCES_PATH, LEGACY_SOURCES_PATH):
        if not os.path.isfile(path):
            continue

        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 4:
                    used_urls.add(row[3])


def rename_to_four_digits(folder_path, folder):
    """apple_001.jpg -> apple_0001.jpg (для корректной сортировки до 1000)."""
    pattern = re.compile(rf"{re.escape(folder)}_(\d+)\.jpg")

    for name in sorted(os.listdir(folder_path)):
        match = pattern.fullmatch(name)

        if not match or len(match.group(1)) >= 4:
            continue

        new_name = f"{folder}_{int(match.group(1)):04d}.jpg"
        new_path = os.path.join(folder_path, new_name)

        if not os.path.exists(new_path):
            os.rename(os.path.join(folder_path, name), new_path)


def count_images(folder_path):
    return sum(1 for n in os.listdir(folder_path) if n.lower().endswith(".jpg"))


def next_free_index(folder_path, folder):
    pattern = re.compile(rf"{re.escape(folder)}_(\d+)\.jpg")
    max_index = 0

    for name in os.listdir(folder_path):
        match = pattern.fullmatch(name)

        if match:
            max_index = max(max_index, int(match.group(1)))

    return max_index + 1


def limit_reached(count, added, quota):
    return count >= LIMIT or (quota is not None and added >= quota)


def collect(folder, queries, per_query):
    """
    Докачивает изображения до LIMIT. Первый проход ограничивает число
    изображений на один запрос (разнообразие выборки), второй — добирает
    недостающее без ограничения.
    """
    folder_path = os.path.join(DATA_DIR, folder)
    os.makedirs(folder_path, exist_ok=True)

    rename_to_four_digits(folder_path, folder)
    count = count_images(folder_path)
    index = next_free_index(folder_path, folder)

    print(f"\n=== {folder}: уже есть {count}, цель {LIMIT} ===")

    is_new = not os.path.exists(SOURCES_PATH)

    with open(SOURCES_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        if is_new:
            writer.writerow(["filename", "class", "source", "url"])

        for quota in (per_query, None):
            for query in queries:
                if count >= LIMIT:
                    break

                print("\nSearching:", query)
                added = 0

                for source in (wikimedia_search(query), openverse_search(query)):
                    if limit_reached(count, added, quota):
                        break

                    for url, source_name in source:
                        if limit_reached(count, added, quota):
                            break

                        if url in used_urls:
                            continue

                        try:
                            data = download_image(url)
                            used_urls.add(url)

                            if not data:
                                continue

                            image_hash = hashlib.sha256(data).hexdigest()

                            if image_hash in used_hashes:
                                continue

                            used_hashes.add(image_hash)

                            filename = f"{folder}_{index:04d}.jpg"

                            with open(os.path.join(folder_path, filename), "wb") as f:
                                f.write(data)

                            writer.writerow([filename, folder, source_name, url])
                            csvfile.flush()

                            count += 1
                            index += 1
                            added += 1
                            print(f"[{count}/{LIMIT}] {folder}/{filename}")

                        except urllib.error.HTTPError as e:
                            if e.code in (429, 503):
                                # ссылка не помечается как использованная:
                                # при следующем запуске её можно скачать снова
                                print(f"Лимит запросов ({e.code}), "
                                      f"пауза {COOLDOWN_AFTER_429} с")
                                time.sleep(COOLDOWN_AFTER_429)
                            else:
                                used_urls.add(url)
                                print("Image skipped:", str(e)[:100])

                        except Exception as e:
                            used_urls.add(url)
                            print("Image skipped:", str(e)[:100])

            if count >= LIMIT:
                break

    print(f"\nFinished {folder}: {count} images")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    migrate_legacy_metadata()
    load_known()

    collect("apple", APPLE_QUERIES, per_query=60)
    collect("other", OTHER_QUERIES, per_query=25)

    print("\nDataset download completed!")
