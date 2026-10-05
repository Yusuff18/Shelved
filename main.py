"""Shelved: find books similar to one you loved.

Run from this folder:
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/
"""
import ast
import os
import random
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "book_dataset.csv"
STATIC_DIR = BASE / "static"
COVER_DIR = BASE / "cover_cache"
COVER_BASE = os.environ.get(
    "Shelved_COVER_BASE", "https://covers.openlibrary.org/b/isbn"
).rstrip("/")

# Tags that describe format or are too broad to be useful on screen.
# They still count towards similarity, they are just not shown as genres.
HIDDEN_GENRES = {"Audiobook", "Audiobooks", "Fiction"}

INVISIBLE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
BOILERPLATE = re.compile(r"^[\s\u25b6]*alternative cover edition\s*#?\d*\s*", re.I)
# "Tamlin.As her marriage": a full stop glued to the next capital means the
# source had a line break that got stripped. Turn it back into a paragraph.
GLUED = re.compile(r"([a-z0-9][.!?\u2026]+[\"\u201d\u2019')\]]?)(?=[A-Z])")
# "Thorns and Roses .Feyre": a stray space before the full stop.
SPACE_BEFORE_STOP = re.compile(r"\s+([.!?])(?=[A-Z])")
ISBN_RE = re.compile(r"(?<!\d)(\d{13}|\d{9}[\dXx])(?![\dXx])")
SERIES_RE = re.compile(r"\(([^()]*#\s*[^()]+)\)\s*$")


def tidy(value) -> str:
    if not isinstance(value, str):
        return ""
    value = INVISIBLE.sub("", value).replace("\xa0", " ")
    return re.sub(r"[ \t]+", " ", value).strip()


def parse_list(value) -> list:
    """Turn "['A', 'B']" (how the CSV stores lists) into ["A", "B"]."""
    if isinstance(value, (list, tuple)):
        items = list(value)
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            parsed = ast.literal_eval(text)
            items = list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
        except (ValueError, SyntaxError):
            items = text.strip("[]").split(",")
    else:
        return []
    out = []
    for item in items:
        s = tidy(str(item)).strip(" '\"")
        if s and s not in out:
            out.append(s)
    return out


def clean_description(value) -> list:
    if not isinstance(value, str):
        return []
    text = INVISIBLE.sub("", value).replace("\xa0", " ").replace("\t", " ")
    text = BOILERPLATE.sub("", text)
    text = SPACE_BEFORE_STOP.sub(r"\1", text)
    text = GLUED.sub(r"\1\n\n", text)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n+", text)]
    return [p for p in paragraphs if p]


def parse_series(title_complete: str) -> str:
    match = SERIES_RE.search(title_complete or "")
    if not match:
        return ""
    inner = match.group(1).strip()
    name, _, number = inner.rpartition("#")
    name = name.strip(" ,")
    number = number.strip()
    if not name:
        return ""
    if re.fullmatch(r"\d+(\.\d+)?", number):
        return f"Book {number} of {name}"
    return f"{name} #{number}"


def pick_isbn(value) -> str:
    if not isinstance(value, str):
        return ""
    candidates = [m.upper() for m in ISBN_RE.findall(value.replace("-", ""))]
    for c in candidates:
        if len(c) == 13:
            return c
    return candidates[0] if candidates else ""


def norm(text: str) -> str:
    """Lowercase, strip accents and punctuation so curly and straight apostrophes match."""
    text = unicodedata.normalize("NFKD", INVISIBLE.sub("", text))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"['\u2018\u2019`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def to_int(value, low=0, high=10**9):
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if low < number <= high else None


# --------------------------------------------------------------------------
# Load and prepare the data once at startup
# --------------------------------------------------------------------------
raw = pd.read_csv(DATA_FILE, dtype={"isbn": str})
raw = raw.drop_duplicates(subset=["title", "author"]).reset_index(drop=True)
N = len(raw)


def column(name):
    return raw[name].tolist() if name in raw.columns else [None] * N


_titles = [tidy(t) for t in column("title")]
_complete = [tidy(t) for t in column("titleComplete")]
_authors = [parse_list(a) for a in column("author")]
_genres = [parse_list(g) for g in column("genres")]
_desc = [clean_description(d) for d in column("description")]
_isbn = [pick_isbn(i) for i in column("isbn")]
_publisher = [tidy(p) for p in column("publisher")]
_language = [tidy(p) for p in column("language")]
_pages = [to_int(p, 0, 5000) for p in column("numPages")]
_ratings = [to_int(r, -1, 10**9) or 0 for r in column("ratingsCount")]
_characters = [parse_list(c)[:10] for c in column("characters")]

POP = np.array(_ratings, dtype=float)
KEYS = [norm(t) for t in _titles]
AUTHOR_KEYS = [norm(", ".join(a)) for a in _authors]
DEDUPE = [f"{KEYS[i]}|{norm(_authors[i][0]) if _authors[i] else ''}" for i in range(N)]

BOOKS = []
for i in range(N):
    BOOKS.append(
        {
            "id": i,
            "title": _titles[i],
            "series": parse_series(_complete[i]),
            "authors": _authors[i] or ["Unknown author"],
            "genres": [g for g in _genres[i] if g not in HIDDEN_GENRES],
            "description": _desc[i],
            "publisher": _publisher[i] or None,
            "pages": _pages[i],
            "language": _language[i] or None,
            "ratings": _ratings[i],
            "characters": _characters[i],
            "isbn": _isbn[i] or None,
        }
    )

# Genre vectors. Each book is a unit-length row, so the dot product of two
# rows is their cosine similarity. We only ever compare one book against the
# rest, which keeps memory tiny (no N x N matrix).
_vectorizer = CountVectorizer(analyzer=lambda genres: genres, binary=True, lowercase=False)
MATRIX = normalize(_vectorizer.fit_transform(_genres).astype(np.float32)).tocsr()

COVER_DIR.mkdir(exist_ok=True)
print(f"Shelved loaded {N:,} books, {sum(1 for b in BOOKS if b['isbn']):,} with an ISBN for covers.")


# --------------------------------------------------------------------------
# Search and similarity
# --------------------------------------------------------------------------
def brief(i: int) -> dict:
    b = BOOKS[i]
    return {"id": i, "title": b["title"], "authors": b["authors"], "series": b["series"]}


def suggest(query: str, limit: int = 8) -> list:
    q = norm(query)
    if not q:
        return []
    tokens = q.split()
    hits = []
    for i in range(N):
        key = KEYS[i]
        hay = key + " " + AUTHOR_KEYS[i]
        if not all(t in hay for t in tokens):
            continue
        if key.startswith(q):
            rank = 0
        elif all(t in key for t in tokens):
            rank = 1
        else:
            rank = 2
        hits.append((rank, -POP[i], i))
    hits.sort()
    out, seen = [], set()
    for _, _, i in hits:
        if DEDUPE[i] in seen:
            continue
        seen.add(DEDUPE[i])
        out.append(brief(i))
        if len(out) >= limit:
            break
    return out


def similar_to(idx: int, count: int) -> list:
    sims = (MATRIX @ MATRIX[idx].T).toarray().ravel()
    sims[idx] = -1.0
    candidates = np.flatnonzero(sims > 0)
    # Most shared genres first; when tied, the book more people have rated.
    order = candidates[np.lexsort((-POP[candidates], -np.round(sims[candidates], 5)))]
    source_genres = set(BOOKS[idx]["genres"])
    seen = {DEDUPE[idx]}
    results = []
    for j in order:
        if DEDUPE[j] in seen:
            continue
        seen.add(DEDUPE[j])
        record = dict(BOOKS[j])
        record["shared"] = [g for g in BOOKS[j]["genres"] if g in source_genres]
        record["n_shared"] = len(record["shared"])
        results.append(record)
        if len(results) >= count:
            break
    return results


# --------------------------------------------------------------------------
# Cover images: fetched from Open Library once, then served from disk
# --------------------------------------------------------------------------
_cover_gate = threading.BoundedSemaphore(4)
_VALID_ISBN = re.compile(r"^(?:\d{13}|\d{9}[\dX])$")
_MISS_TTL = 14 * 24 * 3600
_FOREVER = {"Cache-Control": "public, max-age=2592000"}


def fetch_cover(isbn: str, size: str):
    """Returns (status, bytes). status is 'ok', 'missing' or 'retry'."""
    url = f"{COVER_BASE}/{isbn}-{size}.jpg?default=false"
    request = urllib.request.Request(
        url, headers={"User-Agent": "Shelved/1.0 (personal book recommender)"}
    )
    with _cover_gate:
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                content_type = response.headers.get_content_type()
                body = response.read(3_000_000)
        except urllib.error.HTTPError as err:
            return ("missing", b"") if err.code == 404 else ("retry", b"")
        except (urllib.error.URLError, TimeoutError, OSError):
            return ("retry", b"")
    if not content_type.startswith("image/") or len(body) < 1500:
        return "missing", b""
    return "ok", body


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
app = FastAPI(title="Shelved")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/popular")
def popular(n: int = Query(12, ge=1, le=30)):
    top = np.argsort(-POP)[:90]
    seen, pool = set(), []
    for i in top:
        if DEDUPE[i] not in seen:
            seen.add(DEDUPE[i])
            pool.append(int(i))
    picks = random.sample(pool, min(n, len(pool)))
    return {"total": N, "books": [BOOKS[i] for i in picks]}


@app.get("/api/suggest")
def api_suggest(q: str = "", limit: int = Query(8, ge=1, le=20)):
    return suggest(q, limit)


@app.get("/api/book/{book_id}")
def api_book(book_id: int, n: int = Query(36, ge=1, le=60)):
    if not 0 <= book_id < N:
        raise HTTPException(status_code=404, detail="Book not found")
    return {"book": BOOKS[book_id], "similar": similar_to(book_id, n)}


@app.get("/cover/{isbn}")
def cover(isbn: str, size: str = "M"):
    isbn = isbn.upper()
    size = size.upper()
    if not _VALID_ISBN.match(isbn) or size not in {"S", "M", "L"}:
        raise HTTPException(status_code=404)
    cached = COVER_DIR / f"{isbn}-{size}.jpg"
    missing = COVER_DIR / f"{isbn}-{size}.none"
    if cached.exists():
        return FileResponse(cached, media_type="image/jpeg", headers=_FOREVER)
    if missing.exists() and time.time() - missing.stat().st_mtime < _MISS_TTL:
        return Response(status_code=404, headers={"Cache-Control": "public, max-age=86400"})
    status, body = fetch_cover(isbn, size)
    if status == "ok":
        tmp = cached.with_suffix(".tmp")
        tmp.write_bytes(body)
        os.replace(tmp, cached)
        return Response(body, media_type="image/jpeg", headers=_FOREVER)
    if status == "missing":
        missing.touch()
        return Response(status_code=404, headers={"Cache-Control": "public, max-age=86400"})
    # Rate limited or offline: do not remember it, try again next time.
    return Response(status_code=503, headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")