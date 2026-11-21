"""Export the offline viewer's current preparation as an animated GIF."""
import argparse
from pathlib import Path
import shutil
import subprocess
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('domain',choices=['beat','atom','molecule','spectrum','field','thermal','sky','nuclear']);p.add_argument('object');p.add_argument('output',type=Path)
    p.add_argument('--mode',type=int);p.add_argument('--band');p.add_argument('--angle',type=int,default=24);p.add_argument('--language',choices=['en','fr'],default='en');p.add_argument('--chromium',help='Path to an existing Chrome or Chromium executable')
    p.add_argument('--engine',choices=['canvas','browser'],default=None,
                   help='Default: native canvas with Node, or browser when --chromium is provided')
    args=p.parse_args()
    if args.output.exists():p.error('Choose a new output file')
    if not 0<=args.angle<=180:p.error('View angle must be between 0 and 180 degrees')
    engine=args.engine or ('browser' if args.chromium else 'canvas')
    if engine=='canvas':
        command=['node',str(Path(__file__).with_name('comparison-export.cjs')),args.domain,args.object,
                 str(args.output),'--angle',str(args.angle),'--language',args.language]
        if args.mode is not None:command+=['--mode',str(args.mode)]
        if args.band:command+=['--band',args.band]
        subprocess.run(command,check=True)
        return
    from playwright.sync_api import sync_playwright
    with sync_playwright() as play:
        browser=play.chromium.launch(executable_path=args.chromium or shutil.which('chromium') or shutil.which('google-chrome'),args=['--disable-dev-shm-usage'])
        try:
            page=browser.new_page(accept_downloads=True);page.goto(Path(__file__).with_name('comparisons.html').as_uri())
            page.select_option('#language',args.language);page.select_option('#domain',args.domain)
            values=page.locator('#item option').evaluate_all('(opts)=>opts.map(x=>x.value)')
            if args.object not in values:p.error('Select one of '+', '.join(values))
            page.select_option('#item',args.object)
            if args.domain=='atom' and not page.evaluate('Boolean(D.atoms[state.item] || D.modalScenes[state.item]?.mode)'):
                p.error('This element has a catalogue record but no spatial movie')
            if args.mode is not None or args.band:
                options=page.locator('#options select')
                choice=args.band or str(args.mode)
                if not options.count() or choice not in options.locator('option').evaluate_all('(opts)=>opts.map(x=>x.value)'):p.error('Unsupported mode or band for this preparation')
                options.select_option(choice)
            angle=page.locator('#view input')
            if angle.count():angle.evaluate("(node, value) => { node.value = value; node.dispatchEvent(new Event('input', {bubbles:true})); }", str(args.angle))
            with page.expect_download() as event:page.locator('#export').click()
            args.output.parent.mkdir(parents=True,exist_ok=True);event.value.save_as(str(args.output))
        finally: browser.close()
if __name__=='__main__':main()
