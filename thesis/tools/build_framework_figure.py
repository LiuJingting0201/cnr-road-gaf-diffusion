"""Render the implemented framework diagram without loading experiment data."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
def finish(fig,name):
 for ext in ('pdf','png'):fig.savefig(a.output/(name+'.'+ext),bbox_inches='tight',dpi=180)
 plt.close(fig)
fig,ax=plt.subplots(figsize=(10.8,7.1));ax.set(xlim=(0,12),ylim=(0,8.4));ax.axis('off')
def box(x,y,w,h,label,color='#edf1f4'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc=color,ec='#71818d',lw=.8))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
def arrow(x,y,u,v):ax.annotate('',xy=(u,v),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':1.1,'color':'#455b6b'})
ax.text(.1,8.1,'Training: observed images teach codes that condition trajectory denoising',weight='bold')
box(.15,6.7,2.1,.85,'Observed timed XY');box(3,7.15,2.7,.7,'Original Hilbert image','#f0e8f4');box(3,6.1,2.7,.7,'Derived speed image','#e2f0ec')
arrow(2.33,7.2,2.9,7.5);arrow(2.33,7,2.9,6.45)
box(6.4,6.6,2.2,.9,'Partly trainable CNNs\n32D + 32D codes');arrow(5.8,7.5,6.3,7.15);arrow(5.8,6.45,6.3,6.9)
box(9.4,6.55,2.3,1.,'Joint training objective\nNoise loss + class CE','#e5edf5');arrow(8.7,7.05,9.3,7.05)
ax.text(7.5,5.8,'Denoising gradients reach CNN tails',ha='center',fontsize=9,color='#40586a')
ax.plot([.1,11.8],[5.4,5.4],c='#ccd4da',lw=1)
ax.text(.1,5.05,'New synthesis: available conditions and sampled codes only',weight='bold')
box(.15,3.75,2.3,.9,'Map + OD + class\nPublic bus stops');box(3.1,3.75,2.3,.9,'Learned route policy\nDuration prior')
arrow(2.55,4.2,3,4.2);box(6.2,3.75,2.3,.9,'Conditional code prior\n50 DDIM steps','#f0e8f4');arrow(5.5,4.2,6.1,4.2)
box(9.3,3.65,2.4,1.1,'Motion diffusion\n100 DDIM steps\nCode at every step','#e5edf5');arrow(8.6,4.2,9.2,4.2)
box(9.3,1.65,2.4,1.,'Road decoder\nComplete timed XY','#e2f0ec');arrow(10.5,3.55,10.5,2.75)
box(3.1,1.65,5.4,1.,'Independent physical quality, diversity, coverage\nReal-data augmentation on separate cohorts')
arrow(9.2,2.15,8.6,2.15);ax.text(.2,.65,'Out-of-sample inference: requires only map, OD, class, and sampled latent codes.',fontsize=10)
finish(fig,'g40-framework')
