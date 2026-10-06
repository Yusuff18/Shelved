# Shelved

Type in a book you loved and Shelved finds others like it. Each result shows its cover, genres, rating and description, and you can open any book to read more or jump to its Goodreads page.

There's no content filtering. It recommends whatever is genuinely similar, spicy books included.

## How it works

Every book has a list of genres. I turn those into vectors and rank other books by cosine similarity to the one you searched. Ties go to the book more people have rated, so popular matches come first.

It covers about 12,500 books. The similarity runs per query against a sparse matrix, so there's no huge precomputed table and it stays light on memory.

## What's in it

- Search with autocomplete that ignores accents, apostrophes and punctuation
- Cover images from Open Library, fetched through a small caching proxy so it doesn't hammer their API
- Details panel with the full description, shared genres and a Goodreads link
- Genre filters, a saved reading list, and light/dark themes
- Shareable links to a book's results

## Stack

Python, FastAPI, pandas, scikit-learn on the back end. Plain HTML, CSS and JavaScript on the front end, with no framework.

## Run it

    python -m venv venv
    venv\Scripts\activate
    pip install fastapi uvicorn pandas numpy scikit-learn
    uvicorn main:app

Then open http://127.0.0.1:8000.

## Data

Book data comes from a Goodreads-based dataset, cleaned and deduplicated. Descriptions had stray formatting and "alternative cover" boilerplate that I stripped out.

## Next

- Add a much larger book dataset
- Use description text as a second similarity signal
- Measure recommendation quality properly
- Deploy it
