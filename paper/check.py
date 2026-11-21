"""Check bilingual parity, reading-path destinations and figure assets in the manuscripts."""
from collections import Counter
from pathlib import Path
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parent

def read(lang):
    source = ROOT / f'manuscript-{lang}.md'
    tree = json.loads(subprocess.check_output(['pandoc', '-f', 'markdown-smart', '-t', 'json', str(source)]))
    equations, links, figures, headings = [], [], [], []
    def visit(node):
        if isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            kind, content = node.get('t'), node.get('c')
            if kind == 'Math':
                equations.append((content[0]['t'], re.sub(r'\s+', '', content[1].replace('{,}', '.'))))
            elif kind == 'Header':
                headings.append((content[0], content[1][0]))
            elif kind == 'Link':
                links.append(content[-1][0])
            elif kind == 'Image':
                path = ROOT / content[-1][0]
                if not path.is_file():
                    raise FileNotFoundError(path)
                figures.append(path.name)
            for child in node.values():
                visit(child)
    visit(tree)
    ids = [identifier for _, identifier in headings]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate heading anchor in {lang}')
    if ids[:3] != ['abstract', 'introduction', 'reading-paths']:
        raise ValueError(f'Missing front matter in {lang}')
    missing = {url[1:] for url in links if url.startswith('#')} - set(ids)
    if missing:
        raise ValueError(f'Missing destination in {lang}: {missing}')
    return equations, links, figures, [level for level, _ in headings], [key for level, key in headings if level == 1]

def main():
    en, fr = read('en'), read('fr')
    for name, a, b in zip(('equations', 'link targets', 'figures', 'heading hierarchy', 'chapter anchors'), en, fr):
        if a != b:
            raise ValueError(f'Edition mismatch in {name}')
    equations, urls, figures, headings, chapters = en
    external = [url for url in urls if not url.startswith('#')]
    print(json.dumps({'equations': dict(Counter(t for t, _ in equations)),
                      'external_link_occurrences': len(external), 'distinct_external_targets': len(set(external)),
                      'internal_link_occurrences': len(urls)-len(external), 'figures': len(figures),
                      'headings': len(headings), 'top_level_headings': len(chapters)}, indent=2))

if __name__ == '__main__':
    main()
