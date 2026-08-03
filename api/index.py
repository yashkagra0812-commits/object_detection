import os

os.environ.setdefault("VERCEL", "1")

from app import app

if __name__ == "__main__":
    app.run()
