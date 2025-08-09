import os
import re
import time
import json
import unicodedata
from typing import Dict, Any, List, Optional

import requests

# ================================
# HARD-CODED SETTINGS
# ================================
OUTPUT_FOLDER = r"C:\Kishan\corpus"  # Save files here
LANGUAGE = "en"                      # 'en' for English; set None for all languages
ALL_LANGUAGES = False                # True = ignore LANGUAGE and download all languages
MAX_FILES = 0                        # 0 = no limit
MIN_WORDS = 2000                     # Skip tiny texts (< this many words)
RATE_LIMIT = 1.0                     # Requests per second

STORY_SUBJECT_TOKENS = [
    # Core Fiction
    "Fiction", "Short stories", "Stories", "Tales", "Novels", "Novelettes",
    "Romance", "Love stories", "Adventure stories", "Detective and mystery stories",
    "Ghost stories", "Horror tales", "Gothic fiction", "Fantasy fiction",
    "Science fiction", "Speculative fiction", "Weird fiction", "Magical realism",
    "Slipstream fiction", "Supernatural fiction", "Metafiction", "Pulp fiction",

    # Folklore & Myth
    "Fairy tales", "Folklore", "Folk tales", "Urban legends", "Legends",
    "Mythology", "Myths", "Sagas", "Epics", "Heroic tales", "Creation myths",
    "Arthurian romances", "Chivalry tales",

    # Historical & Period
    "Historical fiction", "Victorian fiction", "Edwardian fiction",
    "Medieval fiction", "Renaissance fiction", "Regency fiction",
    "Ancient world fiction", "Classical literature", "Modernist fiction",
    "Postmodern fiction", "Romanticism literature", "Enlightenment literature",

    # Drama & Performance
    "Drama", "Tragedies", "Comedies", "Farce", "One-act plays", "Stage plays",
    "Theatre scripts", "Play scripts", "Satire", "Parody", "Burlesque",
    "Opera libretti", "Screenplays", "Radio plays",

    # Youth & Children
    "Juvenile fiction", "Children's stories", "Young adult fiction",
    "Animal stories", "School stories", "Boys' stories", "Girls' stories",
    "Bedtime stories", "Moral tales", "Didactic fiction",

    # Crime & Suspense
    "Crime stories", "Thrillers", "Spy stories", "Suspense fiction",
    "Legal fiction", "Hardboiled fiction", "Noir fiction", "Police procedurals",

    # War & Action
    "War stories", "Military fiction", "Naval stories", "Western stories",
    "Martial fiction", "Revolutionary fiction", "Post-apocalyptic fiction",

    # Romance Subtypes
    "Regency romance", "Historical romance", "Paranormal romance",
    "Gothic romance", "Romantic suspense", "Erotic romance", "Sensational fiction",

    # Spiritual & Moral
    "Allegories", "Parables", "Religious fiction", "Biblical fiction",
    "Inspirational fiction", "Mystical fiction", "Occult fiction",
    "Esoteric fiction", "Prophetic fiction", "Visionary fiction",

    # Humorous & Light
    "Humorous stories", "Comic stories", "Anecdotes", "Satirical fiction",
    "Wit and humor", "Irony", "Absurdist fiction",

    # Travel & Exploration
    "Travel fiction", "Exploration stories", "Sea stories", "Shipwreck stories",
    "Survival fiction", "Jungle stories", "Polar exploration fiction",

    # Narrative Forms
    "Epistolary fiction", "Frame stories", "Autobiographical fiction",
    "Biographical fiction", "Alternate history fiction", "Experimental fiction",
    "Utopian fiction", "Dystopian fiction", "Political fiction",
    "Campus fiction", "Family sagas", "Generational fiction",

    # Taboo / Adult Themes (public-domain only)
    "Erotic fiction", "Decadent fiction", "Banned books", "Controversial fiction",
    "Forbidden fiction", "Scandal fiction",

    # Special Interest & Misc
    "Sports fiction", "Music fiction", "Art fiction", "Courtroom fiction",
    "Medical fiction", "Pirate stories", "Bandit stories", "Detective pulp",
    "Science adventure", "Invasion fiction", "Lost world fiction",
    "Planetary romance", "Sword and sorcery", "Steampunk fiction", "Weird tales"
]


GUTENDEX_BASE = "https://gutendex.com/books"

# ================================
# HELPERS
# ================================
def slugify(text: str, allow_unicode=False, max_len: int = 120) -> str:
    text = str(text)
    if allow_unicode:
        text = unicodedata.normalize('NFKC', text)
    else:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r"[^\w\s-]", "", text).strip()
    text = re.sub(r"[-\s]+", "-", text)
    return text[:max_len] if max_len else text

def pick_txt_url(formats: Dict[str, str]) -> Optional[str]:
    for k in ["text/plain; charset=utf-8", "text/plain; charset=UTF-8"]:
        if k in formats and formats[k].startswith("http"):
            return formats[k]
    if "text/plain" in formats and formats["text/plain"].startswith("http"):
        return formats["text/plain"]
    return None

def is_story_book(book: Dict[str, Any], subject_tokens: List[str]) -> bool:
    subjects = [s.lower() for s in book.get("subjects", [])]
    shelves = [s.lower() for s in book.get("bookshelves", [])]
    tokens = [t.lower() for t in subject_tokens]
    return any(any(tok in s for s in subjects) for tok in tokens) or \
           any(any(tok in sh for tok in tokens) for sh in shelves)

def fetch_books(language: Optional[str], all_languages: bool, subject_tokens: List[str]):
    page = 1
    while True:
        params = {"mime_type": "text/plain", "page": page}
        if not all_languages and language:
            params["languages"] = language
        r = requests.get(GUTENDEX_BASE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if not results:
            break
        for book in results:
            if is_story_book(book, subject_tokens):
                yield book
        page += 1
        if not data.get("next"):
            break
        time.sleep(0.5)

def download_txt(url: str, dest_path: str, max_retries: int = 5):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    attempt = 0
    while True:
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                raw = r.content
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")
            text = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\x00")
            with open(dest_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            return
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = min(2 ** attempt, 30)
            print(f"[warn] Retry {attempt}/{max_retries} after error: {e} (sleep {sleep_s}s)")
            time.sleep(sleep_s)

# ================================
# MAIN
# ================================
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    seen_ids = set()
    downloaded = 0
    interval = 1.0 / max(RATE_LIMIT, 0.1)
    last_time = 0.0

    for book in fetch_books(language=LANGUAGE, all_languages=ALL_LANGUAGES, subject_tokens=STORY_SUBJECT_TOKENS):
        book_id = book.get("id")
        if book_id in seen_ids:
            continue
        seen_ids.add(book_id)

        title = book.get("title") or f"book-{book_id}"
        authors = ", ".join(a.get("name", "").strip() for a in book.get("authors", [])) or "unknown"
        formats = book.get("formats", {}) or {}

        txt_url = pick_txt_url(formats)
        if not txt_url:
            continue

        safe_title = slugify(title, allow_unicode=False)
        safe_auth = slugify(authors, allow_unicode=False, max_len=60)
        filename = f"{safe_title}__{safe_auth}__gutenberg_{book_id}.txt"
        dest_path = os.path.join(OUTPUT_FOLDER, filename)

        if os.path.exists(dest_path):
            continue

        now = time.time()
        if now - last_time < interval:
            time.sleep(max(0.0, interval - (now - last_time)))
        last_time = time.time()

        try:
            download_txt(txt_url, dest_path)
            with open(dest_path, "r", encoding="utf-8") as f:
                words = f.read().split()
            if len(words) < MIN_WORDS:
                os.remove(dest_path)
                continue

            downloaded += 1
            print(f"[downloaded] {filename}")

            if MAX_FILES and downloaded >= MAX_FILES:
                print(f"[done] Reached MAX_FILES={MAX_FILES}. Files saved in: {os.path.abspath(OUTPUT_FOLDER)}")
                return
        except Exception as e:
            print(f"[error] Skipping id={book_id} ({title!r}) due to: {e}")

    print(f"[done] Finished crawl. Files saved in: {os.path.abspath(OUTPUT_FOLDER)} | total files: {downloaded}")

if __name__ == "__main__":
    main()
