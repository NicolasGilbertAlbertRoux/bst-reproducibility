"""Create the compact offline atlas from full-resolution neutral GIFs."""
import argparse
from pathlib import Path
from PIL import Image

def convert(source,target):
    with Image.open(source) as original:
        frames=[];durations=[];elapsed=0
        for i in range(original.n_frames):
            original.seek(i);delay=original.info.get('duration',90)
            if i%3==0:
                if frames:durations.append(elapsed)
                elapsed=0;image=original.convert('RGB');image.thumbnail((256,158),Image.Resampling.LANCZOS)
                frames.append(image.quantize(96,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE))
            elapsed+=delay
        durations.append(elapsed);frames[0].save(target,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True,disposal=1)

def compact(source,destination):
    destination.mkdir(parents=True,exist_ok=True)
    for path in sorted(source.glob('*.gif')):convert(path,destination/path.name)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();compact(a.source,a.destination)
