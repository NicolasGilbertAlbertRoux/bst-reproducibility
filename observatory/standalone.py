"""Bundle both tabs, the rendering code and their data into one offline HTML file."""
from pathlib import Path
import argparse, json, re
ROOT=Path(__file__).resolve().parent

def inline(page):
    html=(ROOT/page).read_text();html=re.sub(r'<link rel="stylesheet" href="([^"]+)">', lambda m:'<style>'+(ROOT/m[1]).read_text()+'</style>',html)
    return re.sub(r'<script src="([^"]+)"></script>',lambda m:'<script>'+(ROOT/m[1]).read_text().replace('</script','<\\/script')+'</script>',html)

def build(out):
    lab=json.dumps(inline('comparisons.html'),ensure_ascii=True).replace('<','\\u003c');page=inline('index.html');pos=page.index('</head>');out.write_text(page[:pos]+'<script>window.BST_COMPARISON_HTML='+lab+';</script>'+page[pos:]);
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);build(p.parse_args().output)
