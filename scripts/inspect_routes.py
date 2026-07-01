import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app

for r in app.routes:
    print(type(r).__name__, getattr(r, 'path', None), getattr(r, 'methods', None))
