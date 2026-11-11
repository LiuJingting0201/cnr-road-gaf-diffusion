"""Plot a fixed validation request for each class and its coordinate-derived speed."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
d=np.load('private-B-8000.npz')
fig,axs=plt.subplots(2,5,figsize=(12,4.5),layout='constrained')
for col,label in enumerate(np.unique(d['labels'])):
    i=np.flatnonzero(d['labels']==label)[0]
    for k,color,name in [('reference','#246e56','SUMO'),('generated','#b34d27','DiffTraj')]:
        xy=d[k][i]
        axs[0,col].plot(*xy,color=color,lw=.85,label=name)
        speed=np.linalg.norm(np.diff(xy),axis=0)/d['dt'][i]
        axs[1,col].plot(np.arange(1,200)*d['dt'][i],speed,color=color,lw=.7)
    axs[0,col].set_aspect('equal',adjustable='datalim')
    axs[0,col].set_title({'bycicle':'Bicycle','taxi_or_uber':'Taxi / ride-hailing','private_car':'Private car'}.get(label,label.capitalize()),fontsize=10)
    axs[0,col].set_xlabel('East (m)')
    axs[1,col].set_xlabel('Elapsed time (s)')
    for ax in axs[:,col]:
        ax.tick_params(labelsize=7)
        ax.grid(True,alpha=.15)
axs[0,0].set_ylabel('North (m)')
axs[1,0].set_ylabel('Speed (m/s)')
axs[0,0].legend(fontsize=7)
fig.savefig('baseline-examples.pdf')
