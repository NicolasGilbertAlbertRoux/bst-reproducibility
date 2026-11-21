"""Build the manuscript PDFs from their Markdown sources with Pandoc and XeLaTeX."""
import argparse
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent

def latex(markdown):
    return subprocess.check_output(
        ['pandoc', '-f', 'markdown-smart', '-t', 'latex', '--no-highlight', '--wrap=none'],
        input=markdown, text=True, cwd=ROOT
    )

def build(lang, destination):
    destination.mkdir(parents=True, exist_ok=True)
    source = (ROOT / f'manuscript-{lang}.md').read_text()
    layout = (ROOT / f'layout-{lang}.tex').read_text()
    intro = '# Introduction {#introduction}'
    if source.count('\n' + intro + '\n') != 1:
        raise ValueError('The abstract must precede the introduction.')
    abstract, introduction = source.split('\n' + intro + '\n', 1)
    tex = layout.replace('@@ABSTRACT@@', latex(abstract)).replace('@@BODY@@', latex(intro + '\n' + introduction))
    with TemporaryDirectory() as temp:
        workspace = Path(temp)
        target = workspace / f'manuscript-{lang}.tex'
        target.write_text(tex)
        for _ in range(2):
            result = subprocess.run(
                ['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                 '-output-directory=' + str(workspace), str(target)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            if result.returncode:
                raise RuntimeError(result.stdout.decode(errors='replace')[-2200:])
        shutil.copyfile(target.with_suffix('.pdf'), destination / f'manuscript-{lang}.pdf')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=['en', 'fr', 'both'], default='both', nargs='?')
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    for language in ('en', 'fr') if args.language == 'both' else [args.language]:
        build(language, args.output.resolve())
