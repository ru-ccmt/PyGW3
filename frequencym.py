from numpy import *
import mcommon as mcmn
import numpy.polynomial.legendre as legendre
import numpy.polynomial.laguerre as laguerre

class FrequencyMesh:
    def __init__(self, iopfreq, nomeg, omeg_min, omeg_max, iopMultiple, fout):
        #iopMultiple=20
        fginfo = ['Equally spaced mesh', 'Grid for Gauss-Laguerre quadrature', 'Grid for double Gauss-Legendre quadrature,', 'Grid of Tan-mesh for convolution', 'Using SVD basis and Tan-mesh for convolution']
        self.iopfreq = iopfreq
        print('Frequnecy grid for convolution: iopfreq='+str(iopfreq)+' om_max='+str(omeg_max)+' om_min='+str(omeg_min)+' nom='+str(nomeg)+': ', fginfo[iopfreq-1], file=fout)
        if iopfreq == 1:     # Equaly spaced mesh (for tests or for emac calcultions, not for integration)
            self.omega = linspace(omeg_min, omeg_max, nomeg)
            self.womeg = zeros(nomeg)
        elif iopfreq == 2:   # Grid for Gauss-Laguerre quadrature
            self.omega, wu = laguerre.laggauss(nomeg)
            self.womeg = wu * exp(self.omega)
        elif iopfreq == 3:   # Double Gauss-Legendre quadrature from 0 to omegmax and from omegmax to infinity 
            n = int(nomeg/2)
            u, wu = legendre.leggauss(n)
            self.omega, self.womeg = zeros(2*n), zeros(2*n)
            self.omega[:n] = 0.5*omeg_max*(u+1)
            self.womeg[:n] = 0.5*omeg_max*wu
            u  = u[::-1]   # turn them around
            wu = wu[::-1]  # turn them around
            self.omega[n:] = 2*omeg_max/(u+1)
            self.womeg[n:] = 2*omeg_max*wu/(u+1)**2
        elif iopfreq == 4:   # tan mesh
            # good combination : omeg_max = 20, omeg_min = 0.02, nomeg=32
            om0, dom0 = mcmn.Give2TanMesh(omeg_min, omeg_max, nomeg)
            n = int( len(om0)/2 )
            self.omega, self.womeg = om0[n:], dom0[n:]
            # another iopMultiple-times more precise mesh for self-energy integration
            minx = omeg_min/(1 + 0.2*(iopMultiple-1.))
            om1, dom1 = mcmn.Give2TanMesh(minx, omeg_max, nomeg*iopMultiple)
            n = int(len(om1)/2)
            self.omega_precise, self.womeg_precise = om1[n:], dom1[n:]
            #self.omega_precise, self.womeg_precise = self.omega, self.womeg
            #print >> fout, 'Frequnecy with tan mesh for convolution nom=', len(self.omega)
        elif iopfreq == 5:
            # good combination : omeg_max = 20, omeg_min = 0.02, nomeg=32
            om0, dom0 = mcmn.Give2TanMesh(omeg_min, omeg_max, nomeg)
            n = int(len(om0)/2)
            self.omega, self.womeg = om0[n:], dom0[n:]
        else:
          print('ERROR: wrong iopfreq=', iopfreq, ' should be between 1...3', file=fout)
          
        #print >> fout, 'frequencies and weights'
        for i in range(len(self.omega)):
            print('%3d  x_i=%16.10f w_i=%16.10f' % (i+1, self.omega[i], self.womeg[i]), file=fout)
