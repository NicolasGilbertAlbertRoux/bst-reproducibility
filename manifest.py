"""Write the file checksums after preparing a new benchmark release."""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parent
def main():
    records = {}
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if (not path.is_file() or relative.as_posix() == 'checksums.json'
                or any(part in {'.git', '.venv', '__pycache__', 'node_modules'} for part in relative.parts)):
            continue
        with path.open('rb') as stream:
            records[relative.as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (ROOT / 'checksums.json').write_text(json.dumps(records, indent=2) + '\n')
if __name__ == '__main__':
    main()
