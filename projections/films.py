"""Render the French preview films from fixed camera channels."""
from pathlib import Path
from plate_labels import translate
from optics import spectral_colour
from functools import lru_cache
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.special import j1
from trajectories import FRAMES
from project import DATA_OUT

HOME_DIR=Path(__file__).resolve().parent
OUT=DATA_OUT.parent/'movies';OUT.mkdir(exist_ok=True)
REV=DATA_OUT.parent/'stills';REV.mkdir(exist_ok=True)
FONTS=HOME_DIR/'fonts'
W,H=860,546
IVORY=(241,241,231);MUTED=(176,189,193);GOLD=(250,183,90);RULE=(56,67,69);NEUTRAL=(228,228,228)
SPECTRAL=[516.5,473.7,563.5]
@lru_cache(None)
def font(name,size):
    fname={'regular':'NotoSansDisplay-Regular','bold':'NotoSansDisplay-Bold','title':'NotoSerifDisplay-Regular'}[name]
    return ImageFont.truetype(str(FONTS/(fname+'.ttf')),size)
def label(im,xy,s,size=16,col=IVORY,face='regular',anchor='lm'):
    ImageDraw.Draw(im).text(xy,translate(s),font=font(face,size),fill=col,anchor=anchor)
def line(im,pts,colour=RULE,width=1):
    ImageDraw.Draw(im).line(pts,fill=colour,width=width)
def fr(v,n=2):
    return f'{v:.{n}f}'.replace('.',',')
def dye(col):
    return np.array(col,dtype=float)/255
def colourize(channels,view,col):
    inner,bound,coarse=channels.astype(np.float32)/255
    if view=='fine':
        return Image.fromarray(np.rint(255*inner).astype('uint8'),'L').convert('RGB')
    read=bound if view=='molecular' else coarse
    shell=read**.64
    grain=.16*inner if view=='molecular' else np.zeros_like(inner)
    rgb=1-(1-shell[:,:,None]*dye(col))*(1-grain[:,:,None])
    return Image.fromarray(np.rint(rgb*255).astype('uint8'))
def fitted(im,picture,box):
    x,y,w,h=box;photo=picture.copy();photo.thumbnail((w,h),Image.Resampling.LANCZOS)
    im.paste(photo,(int(x+(w-photo.width)/2),int(y+(h-photo.height)/2)))
def spatial_fine(channels,species,pos,eq):
    x=int(255.5-pos/(eq/2)*96/2)
    a=106;whole=colourize(channels,'fine',IVORY)
    return whole.crop((x-a,124-a,x+a,124+a))
def palette(col):
    vals=[(g,g,g) for g in np.linspace(0,255,47).round().astype(int)]
    c=dye(col)
    for base in np.linspace(.015,1,33):
        for grain in [0.,.045,.09,.14,.20]:
            vals.append(tuple(map(int,np.rint(255*(1-(1-c*base)*(1-grain))))))
    # Include immutable letter colours and the three filter marks.
    for extra in [NEUTRAL,GOLD,MUTED,IVORY,RULE]+[spectral_colour(x) for x in SPECTRAL]:
        vals.extend(tuple(int(a*f) for a in extra) for f in [.3,.64,.83,1.])
    vals+= [(0,0,0)]*(256-len(vals))
    pal=Image.new('P',(1,1));pal.putpalette([int(c) for item in vals for c in item]);return pal
def panel(species,channels,d,phase,info,i,lam):
    kr=species=='Kr2';col=NEUTRAL if kr else spectral_colour(lam);im=Image.new('RGB',(W,H))
    title='Les manteaux du krypton' if kr else 'La couleur du dicarbone'
    label(im,(30,32),title,34,face='title');label(im,(829,32),'BST   /   Kr₂' if kr else 'BST   /   C₂ EXCITÉ',15,GOLD,'bold','rm')
    sub='Deux amas de 309 sites. La forme du manteau suit leurs positions.' if kr else 'Une forme calculée sur 34 sites par atome, éclairée par la teinte du filtre choisi.'
    label(im,(31,72),sub,17,MUTED);line(im,(30,93,830,93))
    label(im,(33,121),'À la résolution moléculaire',20,face='bold')
    view=colourize(channels,'molecular',col)
    im.paste(view.resize((544,264),Image.Resampling.LANCZOS),(25,134))
    line(im,(586,107,586,425))
    label(im,(609,120),'Détail atomique',16,face='bold')
    fitted(im,spatial_fine(channels,species,d[i],info['equilibrium_A']),(637,139,160,115))
    line(im,(611,262,828,262))
    label(im,(609,284),'Manteaux seuls',16,face='bold')
    fitted(im,colourize(channels,'ensemble',col),(608,304,222,110))
    label(im,(264,414),'Distance entre les centres   '+fr(d[i],3)+' Å',17)
    x,y,sw,sh=37,410,203,14
    series=(d-d.min())/np.ptp(d)
    pts=[(x+j*sw/(len(series)-1),y+sh*(1-v)) for j,v in enumerate(series)]
    line(im,pts,(108,121,123),1);x1,y1=pts[i];ImageDraw.Draw(im).ellipse((x1-3,y1-3,x1+3,y1+3),fill=IVORY)
    line(im,(30,445,830,445))
    if kr:
        label(im,(32,466),'Gris · vitesse de déformation du manteau.',18,NEUTRAL,'bold')
        label(im,(32,496),'Le gris garde une trace du mouvement interne. Les deux vues de droite suivent le même instant.',15,MUTED)
        label(im,(32,522),'Temps du dimère   '+fr(phase[i]*info['period_ps'])+' ps',14,MUTED)
    else:
        label(im,(32,467),'Filtre   '+fr(lam,1)+' nm',18,IVORY,'bold')
        for j,wave in enumerate(SPECTRAL):
            px=254+j*158;swatch=spectral_colour(wave)
            ImageDraw.Draw(im).ellipse((px,458,px+13,471),fill=swatch)
            label(im,(px+20,465),fr(wave,1)+' nm',15,IVORY if wave==lam else MUTED,'bold' if wave==lam else 'regular')
            if wave==lam:line(im,(px,478,px+105,478),IVORY,1)
        label(im,(32,499),'Bandes de Swan mesurées. Leur localisation sur les manteaux et leur éclat sont hypothétiques.',15,MUTED)
        label(im,(32,523),'Même mouvement rejoué sous trois filtres   ·   Temps de la vibration   '+fr(phase[i]*info['period_fs'],1)+' fs',14,MUTED)
    return im
def save_gif(frames,target,duration,comment):
    frames[0].save(target,save_all=True,append_images=frames[1:],duration=duration,loop=0,disposal=1,
                   optimize=False,interlace=False,comment=comment.encode('ascii'))
def molecular_movie(species):
    with np.load(DATA_OUT/(species+'.npz')) as obj:
        c=obj['channels'];d=obj['separation'];phase=obj['phase'];info=json.loads(str(obj['info']))
    frames=[];filters=[None] if species=='Kr2' else SPECTRAL
    for lam in filters:
        col=NEUTRAL if lam is None else spectral_colour(lam);pal=palette(col)
        for i in range(FRAMES):
            im=panel(species,c[i],d,phase,info,i,lam)
            if i==15 and lam==filters[0]:
                im.save(REV/(species+'-panel.png'))
                colourize(c[i],'molecular',col).save(REV/(species+'-plate.png'))
            frames.append(im.quantize(palette=pal,dither=Image.Dither.NONE))
    name='BST_Molecule_de_krypton.gif' if species=='Kr2' else 'BST_Couleur_du_dicarbone.gif'
    comment=('Kr2. Numerical strain rate on diffuse geometry, greyscale mechanical readout.' if species=='Kr2' else
             'C2. The same illustrative multiscale motion replayed in three measured Swan bands. Spatial radiance is hypothetical, not a resolved observation.')
    save_gif(frames,OUT/name,75 if len(filters)==1 else 65,comment)
def optical_spot(lam,size,neutral=False):
    y,x=np.mgrid[:size,:size]-(size-1)/2
    z=np.hypot(x,y)*.118*550/lam
    airy=np.ones_like(z);mask=z>0;airy[mask]=(2*j1(z[mask])/z[mask])**2
    tint=(242,242,242) if neutral else spectral_colour(lam)
    return Image.fromarray(np.rint(np.sqrt(np.clip(airy,0,1))[:,:,None]*tint).astype('uint8'))
def atomic_uv(source):
    source=Image.open(source);frames=[]
    palette_frame=Image.new('RGB',(400,200));draw=ImageDraw.Draw(palette_frame)
    hues=[(255,255,255),spectral_colour(557.02894),spectral_colour(587.09160),MUTED,GOLD]
    for j,co in enumerate(hues):
        for i in range(80):draw.rectangle((i*5,j*40,i*5+4,(j+1)*40),fill=tuple(int(t*i/79) for t in co))
    pal=palette_frame.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
    filters=[123.58378,557.02894,587.09160];marks=['UV · gris','Visible','Visible']
    plate=Image.new('RGB',(300,110))
    for j,lam in enumerate(filters):plate.paste(optical_spot(lam,98,j==0),(j*100+1,6))
    plate.save(REV/'atom-uv-plate.png')
    for i in range(source.n_frames):
        source.seek(i);im=source.convert('RGB');d=ImageDraw.Draw(im)
        d.rectangle((22,53,825,89),fill=(0,0,0))
        label(im,(31,73),'À gauche, le mouvement des sites. À droite, trois raies du krypton neutre excité.',17,MUTED)
        d.rectangle((505,105,832,442),fill=(0,0,0))
        label(im,(516,140),'Trois filtres optiques',21,face='bold')
        im.paste(plate,(514,200))
        for j,lam in enumerate(filters):
            x=564+j*100;co=IVORY if j==0 else spectral_colour(lam)
            label(im,(x,335),fr(lam,2)+' nm',15,co,'bold','mm')
            label(im,(x,360),marks[j],14,MUTED,'regular','mm')
        label(im,(658,411),'Source non résolue · même ouverture',15,MUTED,'regular','mm')
        d.rectangle((20,456,830,538),fill=(0,0,0))
        label(im,(30,476),'Le blanc de l’amas code le mouvement des 309 sites. Déplacements internes agrandis × 12.',16,IVORY)
        label(im,(30,507),'Dans le filtre UV, le gris rend visible une raie mesurée à 123,58 nm. Éclats normalisés.',15,MUTED)
        if i==15:im.save(REV/'KrI-panel.png')
        frames.append(im.quantize(palette=pal,dither=Image.Dither.NONE))
    save_gif(frames,OUT/'BST_Atome_de_krypton.gif',70,'Archived Kr I fine-site view. Measured wavelengths: 123.58378 nm (UV mapped to grey), 557.02894 and 587.09160 nm. Filter shapes are illustrative.')
if __name__=='__main__':
    import sys
    for s in ('Kr2','C2'):molecular_movie(s)
    if len(sys.argv)>1:atomic_uv(sys.argv[1])
