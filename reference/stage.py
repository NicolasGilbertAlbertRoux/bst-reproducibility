"""Restore the recorded benchmark's file layout inside a user-selected working directory.

Scientific input and output records carry their original schema identifiers. Public
source names are resolved through the layout table. Each source and each input is
verified independently before any command is executed.
"""
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import shutil
import io
import tokenize

BASE = Path(__file__).resolve().parent
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def inside(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unexpected relative path: {name}')
    return root.joinpath(*relative.parts)
def reference_spelling(source, mapping):
    """Resolve archived API and data-field spellings without changing numerical expressions."""
    if 'lines' in mapping:
        lines=source.decode().splitlines(keepends=True)
        for index, published, archived in mapping['lines']:
            if lines[index]!=published:
                raise ValueError('Published source and line map disagree')
            lines[index]=archived
        return ''.join(lines).encode()
    if not mapping['tokens']:
        return source
    text=source.decode()
    tokens=list(tokenize.generate_tokens(io.StringIO(text).readline))
    offsets=[0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1]+len(line))
    for index, published, archived in reversed(mapping['tokens']):
        token=tokens[index]
        if token.string != published:
            raise ValueError('Published code and field map disagree')
        start=offsets[token.start[0]-1]+token.start[1]
        stop=offsets[token.end[0]-1]+token.end[1]
        text=text[:start]+archived+text[stop:]
    return text.encode()
def stage(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(f'The staging directory must be empty: {destination}')
    layout = json.loads((BASE / 'layout.json').read_text())
    with TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        merged = workspace / 'inputs.zip'
        with merged.open('wb') as out:
            for part in sorted((BASE/'inputs').glob('benchmark-inputs.part*')):
                with part.open('rb') as source:
                    shutil.copyfileobj(source, out)
        if digest(merged) != layout['input_archive_sha256']:
            raise ValueError('Input volume checksum mismatch')
        scopes = {'root': destination}
        scopes.update({k:workspace/k for k in layout['containers']})
        aliases=json.loads((BASE/'schema-aliases.json').read_text())
        with ZipFile(merged) as archive:
            for scope, alias, origin, public_name, expected in layout['entries']:
                target = inside(scopes[scope], alias)
                target.parent.mkdir(parents=True, exist_ok=True)
                if origin == 'code':
                    shutil.copyfile(inside(BASE/'sources', public_name),target)
                else:
                    with archive.open(public_name) as source, target.open('wb') as out:
                        shutil.copyfileobj(source, out)
                if digest(target) != expected:
                    raise ValueError(f'Checksum mismatch: {public_name}')
                if origin=='code' and public_name in aliases:
                    mapping=aliases[public_name]
                    target.write_bytes(reference_spelling(target.read_bytes(),mapping))
                    if digest(target)!=mapping['staged_sha256']:
                        raise ValueError(f'Archived schema reconstruction differs: {public_name}')
        for scope, (parent, name) in layout['containers'].items():
            target=inside(scopes[parent], name)
            target.parent.mkdir(parents=True,exist_ok=True)
            with ZipFile(target,'w',ZIP_DEFLATED,compresslevel=1) as archive:
                for source in sorted(scopes[scope].rglob('*')):
                    if source.is_file():archive.write(source,source.relative_to(scopes[scope]))
        clock_record=destination/'redaction/donnees/Horloges_reference_R411.json'
        if clock_record.exists():
            record=json.loads(clock_record.read_text())
            for n in record['fichiers_originaux']:
                record['fichiers_originaux'][n]=digest(inside(destination/'redaction/heritage_R411',n))
            clock_record.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        manifest={str(p.relative_to(destination)):digest(p) for p in destination.rglob('*') if p.is_file()}
        (destination/'EMPREINTES.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return destination
