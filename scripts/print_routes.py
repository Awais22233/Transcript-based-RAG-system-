import sys
from pathlib import Path
from fastapi.routing import APIRoute

# Ensure project root is on sys.path so `app` can be imported when running this script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app

print('Registered API routes:')
for r in app.routes:
    if isinstance(r, APIRoute):
        print(f"{r.path:40}  methods={sorted(r.methods)}  name={r.name}")
