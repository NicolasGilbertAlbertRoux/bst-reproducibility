"""Export a preparation with the same selection rules and renderer as the offline console."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OPTIONS = (
    'view', 'mode', 'band', 'angle', 'elevation', 'colour', 'path-length', 'isotope', 'nuclear-state',
    'charge', 'language', 'stiffness', 'inertia', 'restoring', 'sign', 'rigidity',
    'field-inertia', 'quantity', 'fit', 'direction', 'wave-number', 'radius1', 'radius2',
    'radius3', 'body1', 'body2', 'body3', 'density1', 'density2', 'density3',
    'separation', 'speed',
)


def browser_export(args, flags):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.chromium)
        page = browser.new_page(accept_downloads=True, reduced_motion='reduce')
        page.goto((ROOT / 'index.html').as_uri())
        page.evaluate("""([domain, item, flags]) => {
            const selected = projectionSettings(P, domain, item, flags);
            pause();
            Object.assign(state, selected);
            frame = 0;
            ui('language').value = selected.lang;
            configure();
        }""", [args.domain, args.object, flags])
        with page.expect_download(timeout=240000) as download:
            page.click('#save')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        download.value.save_as(str(args.output))
        browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('domain')
    parser.add_argument('object')
    parser.add_argument('output', type=Path)
    for key in OPTIONS:
        parser.add_argument('--' + key)
    parser.add_argument('--engine', choices=['canvas', 'browser', 'comparison'], default='canvas')
    parser.add_argument('--chromium')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    flags = []
    for key in OPTIONS:
        val = getattr(args, key.replace('-', '_'))
        if val is not None:
            flags.extend(['--' + key, val])
    positional = [args.domain, args.object, str(args.output.resolve())]
    legacy = (args.domain in ('spectrum', 'nuclear')
              or (args.domain == 'field' and args.object.isdigit())
              or (args.domain == 'thermal' and (args.object == 'Xe' or args.mode is not None))
              or (args.domain == 'sky' and args.band is not None))
    browser = args.engine == 'browser' or bool(args.chromium)
    if (legacy or args.engine == 'comparison') and any(
            '--' + k in flags for k in ('elevation', 'colour', 'path-length', 'isotope', 'nuclear-state', 'charge')):
        parser.error('These options belong to the Simulations panel')
    if browser and (legacy or args.engine == 'comparison'):
        extra = ['--chromium', args.chromium] if args.chromium else []
        subprocess.run([sys.executable, str(ROOT/'comparison_export.py'), *positional,
                        *flags, '--engine', 'browser', *extra], check=True)
    elif browser:
        browser_export(args, flags)
    else:
        entry = 'comparison-export.cjs' if args.engine == 'comparison' else 'render.cjs'
        subprocess.run(['node', str(ROOT/entry), *positional, *flags], check=True)


if __name__ == '__main__':
    main()
