from scipy import *
from pylab import *
import sys,re

def PlotBands(filename, ci, label):
    fi = open(filename, 'r')
    data = fi.readline()
    fi.close()
    s = data[1:].strip()
    s=re.sub('leg=', '', s)
    leg=eval(s)
    data = loadtxt(filename).T
    x_k = data[0]
    for ib in range(1,len(data)):
        if ib==1:
            plot(data[0], data[ib], ci, label=label)
        else:
            plot(data[0], data[ib], ci)
    y0, y1 = ylim()
    for ik,name in leg.items():
        plot([x_k[ik], x_k[ik]], [y0,y1], 'k:', lw=0.3)
    xticks([x_k[ik] for ik,name in leg.items()], labels=[name for ik,name in leg.items()])
    plot([x_k[0],x_k[-1]], [0,0], 'k:', lw=0.3)
    ylim([y0,y1])
    xlim([x_k[0], x_k[-1]])
        
    
if __name__=='__main__':
    #filename = sys.argv[1]
    #PlotBands(filename)
    PlotBands('data/GW_bands_Wannier.dat', 'C0', 'Wannier interp')
    PlotBands('data/GW_bands_Pickett.dat', 'C1', 'Picket interp')
    PlotBands('data/KS_bands.dat', 'C2:', 'DFT')
    legend(loc='best')
    show()
