#!/usr/bin/env python
# @Copyright 2020 Kristjan Haule

Parallel = True
if Parallel :
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    msize = comm.Get_size()
    mrank = comm.Get_rank()
    master=0
else:
    msize = 1
    mrank = 0
    master = 0

#from scipy import *
import sys, os, re
from numpy import *
from numpy import linalg
from timeit import default_timer as timer
from scipy import optimize
import itertools
from functools import reduce
from numba import jit
from numba.typed import List
import shutil, subprocess
from scipy import interpolate

import gwienfile as w2k

#from inout import *
#from kqmesh import *
#from mcommon import *
from cmn import Ry2H,H2eV
from inout import InOut
from kqmesh import KQmesh
import mcommon as mcmn
from kohnsham import KohnShamSystem
from planewaves import PlaneWaves
from matel2band import MatrixElements2Band
from productbasis import ProductBasis
from kweights import Kweights
from wannier import CWannier90,Path_distance,PlotBands,PrintBands,SaveBandPlot
#import os,psutil
#from pympler import asizeof
from pylab import *

ddir = 'data'

class LightFrequencyMesh:
    def __init__(self, io, omega, womeg, fout):
        fginfo = ['Equally spaced mesh', 'Grid for Gauss-Laguerre quadrature', 'Grid for double Gauss-Legendre quadrature,', 'Grid of Tan-mesh for convolution', 'Using SVD basis and Tan-mesh for convolution']
        self.iopfreq = io.iopfreq
        self.omega, self.womeg = omega, womeg
        if io.iopfreq == 4:
            # another iopMultiple-times more precise mesh for self-energy integration
            minx = io.omegmin/(1 + 0.2*(io.iopMultiple-1.))
            om1, dom1 = mcmn.Give2TanMesh(minx, io.omegmax, io.nomeg*io.iopMultiple)
            n = int(len(om1)/2)
            self.omega_precise, self.womeg_precise = om1[n:], dom1[n:]
        print('Frequnecy grid for convolution: iopfreq='+str(io.iopfreq)+' om_max='+str(io.omegmax)+' om_min='+str(io.omegmin)+' nom='+str(io.nomeg)+': ', fginfo[io.iopfreq-1], file=fout)
        for i in range(len(self.omega)):
            print('%3d  x_i=%16.10f w_i=%16.10f' % (i+1, self.omega[i], self.womeg[i]), file=fout)

class SCGW0:
    def __init__(self, io):
        self.sigx  = load(ddir+'/Sigmax.npy')
        self.sigc  = load(ddir+'/Sigmac.npy')
        self.Vxct  = load(ddir+'/Vxct.npy')
        self.omega = load(ddir+'/omega.npy')
        self.womeg = load(ddir+'/womeg.npy')
        self.Ul, self.dUl = None, None
        self.iopfreq = io.iopfreq
        if io.iopfreq == 5:
            self.Ul = load(ddir+'/Ul' )
            self.dUl= load(ddir+'/dUl')
        self.fr = LightFrequencyMesh(io, self.omega, self.womeg, io.out)
        
    def ReadKSEnergy(self, case, nspin, core, strc, kqm, io, fout):
        # Reading w2k energy files and its KS-eigenvalues
        spflag = ['up','dn'] if nspin==2 else ['']
        (self.klist, self.wegh, Ebnd, self.hsrws) = w2k.Read_energy_file(case+'.energy'+spflag[0], strc, fout)
        band_max = min(list(map(len,Ebnd)))
        self.Ebnd = zeros( (nspin,len(Ebnd),band_max) )
        for ik in range(len(Ebnd)):
            self.Ebnd[0,ik,:] = Ebnd[ik][:band_max]
        if nspin==2:
            (self.klist, self.wegh, Ebnd, self.hsrws) = w2k.Read_energy_file(case+'.energy'+spflag[1], strc, fout)
            for ik in range(len(Ebnd)):
                self.Ebnd[1,ik,:] = Ebnd[ik][:band_max]
        # converting to Hartrees
        self.Ebnd *= Ry2H  # convert bands to Hartree
        #######
        # Recompute the Fermi energy, if neeeded
        if io['efermi'] >= 1e-2: # Recompute the Fermi energy
            (EF, Eg, evbm, ecbm, eDos) = mcmn.calc_Fermi(self.Ebnd[0], kqm.atet, kqm.wtet, core.nval, nspin)
            print('Fermi energy was recomputed and set to ', EF*H2eV, file=fout)
        else:
            print(' Use the Fermi energy from case.ingw', file=fout)
            evbm = max( [x for x in self.Ebnd.flatten() if x < EF] )
            ecbm = min( [x for x in self.Ebnd.flatten() if x > EF] )
            Eg = ecbm - evbm
            eDos = sum([ft.dostet(EF, self.Ebnd, kqm.atet, kqm.wtet) for isp in range(nspin)])*2.0/nspin
        # Printing results of Fermi energy and gaps
        if Eg >= 0:
            print('\n'+'-'*32+'\nFermi: Insulating, KS E_Fermi[eV]=%-12.6f Gap[eV]=%-12.6f  EVBM[eV]=%-12.6f  ECBM[eV]=%-12.6f' % (EF*H2eV, Eg*H2eV, evbm*H2eV, ecbm*H2eV), file=fout)
        else:
            print('\n'+'-'*32+'\nFermi: Metallic, KS E_Fermi[eV]=%-12.6f  DOS[E_f]=%-12.6f' % (EF*H2eV, eDos), file=fout)
        print('-'*32, file=fout)
        self.EFermi = EF
        # We want the Fermi energy to be at zero
        # Now we change the band energies so that EF=0
        self.Ebnd -= EF
        if len(core.corind)>0:
            nsp = len(core.eig_core)       # bug jul.7 2020                                                                                                          
            nat = len(core.eig_core[0])    # bug jul.7 2020                                                                                                          
            #nsp,nat,nc = shape(core.eig_core)
            for isp in range(nsp):
                for iat in range(nat):
                    core.eig_core[isp][iat][:] = array(core.eig_core[isp][iat][:])*Ry2H - EF
        self.EF = 0.0
        self.Eg = Eg
        print('Set EF to ', self.EF, file=fout)
        
        ncg = len(core.corind) # number of all core states
        if io['iop_core'] in [0,1]:
            self.ncg_c = ncg
        else:
            self.ncg_c = 0
        print('ncg_c=', self.ncg_c, file=fout)
        
        nkp = shape(self.Ebnd)[1]
        nomax_numin = [0,10000]                        # [index of the last valence band, index of the first conduction band]
        for isp in range(nspin):
            nocc_at_k = [len([x for x in self.Ebnd[isp,ik,:] if x<0]) for ik in range(nkp)] # how many occuiped bands at each k-point
            nomax = max(nocc_at_k)-1                 # index of the last valence band
            numin = min(nocc_at_k)                   # index of the first conduction band
            nomax_numin[0] = max(nomax_numin[0],nomax)
            nomax_numin[1] = min(nomax_numin[1],numin)
        self.nomax_numin = nomax_numin
        print(' Highest occupied band: ', self.nomax_numin[0], file=fout)
        print(' Lowest unoccupied band:', self.nomax_numin[1], file=fout)
        # Set the total number of bands considered in the summation over states
        # for the calculations the exchange (x) and correlation self-energies
        if io['ibgw'] < 0:
            nocc_at_k = [[len([x for x in self.Ebnd[isp,ik,:] if x<io['emingw']]) for ik in range(nkp)] for isp in range(nspin)]# how many bands below io['emingw'] at each k-point
            self.ibgw = min(list(map(min,nocc_at_k)))
            #print 'ibgw=', self.ibgw,  'nocc_at_k=', nocc_at_k
        else:
            self.ibgw = io['ibgw']
        if self.ibgw > self.nomax_numin[1]:
            print('KohnShamSystem: WARNING - range of gw bands!! ibgw=',self.ibgw,'numin=',self.nomax_numin[1], file=fout)
            print('*Now we will set ibgw to 0', file=fout)
            self.ibgw = 0
        print('ibgw=', self.ibgw, file=fout)
        
        
    def Compute_selfc(self, bands, core, kqm, fout, PRINT=True):
        nirkp = len(kqm.weight)
        Nallkp = len(kqm.qlist)*nirkp

        iqs,iqe,sendcounts,displacements = mcmn.mpiSplitArray(mrank, msize, len(kqm.qlist) )
        mwm = load(ddir+'/mwm.0.0.npy')
        (nom, nb2, nb1) = shape(mwm)
        Ndiag=nb1
        if io.MatrixSelfEnergy:
            which_indices = load(ddir+'/wich.0.0.npy', allow_pickle = True)
            Ndiag = count_nonzero([x[0]==x[1] for x in which_indices]) 
            sigc = zeros( (nirkp, Ndiag, Ndiag, len(self.omega) ), dtype=complex )
        else:
            sigc = zeros( (nirkp, nb1, len(self.omega) ), dtype=complex )
            
        t_read, t_cmp = 0.0, 0.0
        
        for iq in range(iqs,iqe):
            #dsigc = zeros( (nirkp, nb1, len(self.omega) ), dtype=complex )
            for irk in range(nirkp):
                t1 = timer()
                mwm = load(ddir+'/mwm.'+str(iq)+'.'+str(irk)+'.npy')
                if io.MatrixSelfEnergy:
                    which_indices = load(ddir+'/wich.'+str(iq)+'.'+str(irk)+'.npy', allow_pickle = True)
                t2 = timer()
                dsig = mcmn.Compute_selfc_inside(iq, irk, bands, mwm, self.fr, kqm, self.ncg_c, core, self.Ul, fout, which_indices, MatrixSelfEnergy=io.MatrixSelfEnergy, PRINT=PRINT)
                
                t3 = timer()
                t_read += t2-t1
                t_cmp  += t3-t2
                
                if io.MatrixSelfEnergy:
                    for i,(i1,i3) in enumerate(which_indices):
                        sigc[irk,i1,i3,:] += dsig[i,:]
                else:
                    for i in range(nb1):
                        sigc[irk,i,:] += dsig[i,:]
                
                for i in range(Ndiag):
                    ie1=i+self.ibgw
                    iom=0
                    print('dSigc[iq=%3d,irk=%3d,ie1=%3d,ie3=%3d]=%16.12f%16.12f' % (iq,irk,ie1,ie1,dsig[i,iom].real,dsig[i,iom].imag), file=fout)
                for i,(i1,i3) in enumerate(which_indices[Ndiag::2]):
                    ie1=i1+self.ibgw
                    ie3=i3+self.ibgw
                    iom=0
                    print('dSigc[iq=%3d,irk=%3d,ie1=%3d,ie3=%3d]=%16.12f%16.12f' % (iq,irk,ie1,ie3,dsig[Ndiag+2*i,iom].real,dsig[Ndiag+2*i,iom].imag), file=fout)
         
        print('## Compute_selfc : t_read    =%10.5f' % (t_read,), file=fout)
        print('## Compute_selfc : t_compute =%10.5f' % (t_cmp,), file=fout)

        if Parallel:
            sigc = comm.allreduce(sigc, op=MPI.SUM)
        
        if PRINT:
            for irk in range(nirkp):
                for i in range(Ndiag):
                    ie1=i+self.ibgw
                    iom=0
                    if io.MatrixSelfEnergy:
                        sig = sigc[irk,i,i,iom]
                    else:
                        sig = sigc[irk,i,iom]
                    print('dSigc[iq=%3d,irk=%3d,ie1=%3d,ie3=%3d]=%16.12f%16.12f' % (iq,irk,ie1,ie1,sig.real,sig.imag), file=fout)
                if io.MatrixSelfEnergy:
                    for i in range(Ndiag):
                        for j in range(i+1,Ndiag):
                            ratio = sum(abs(sigc[irk,i,j,:])+abs(sigc[irk,j,i,:]))/sum(abs(sigc[irk,i,i,:])+abs(sigc[irk,j,j,:]))
                            iom=0
                            if ratio> 0.1*io.sigma_off_ratio:
                                sig = sigc[irk,i,j,iom]
                                print('dSigc[iq=%3d,irk=%3d,ie1=%3d,ie3=%3d]=%16.12f%16.12f' % (iq,irk,ie1,ie3,sig.real,sig.imag), file=fout)
        return sigc
    
    def calceqp(self, io, strc, kqm, nval, core, fout):
        if io.MatrixSelfEnergy:
            (nirkp, nbnd, nbnd2, nom) = shape(self.sigc)
        else:
            (nirkp, nbnd, nom) = shape(self.sigc)
        
        print("SCGW0 : calceqp", file=fout)
        #anc_type=['old-fashioned Pade with n='+str(io.npar_ac-1),'modified Pade with '+str(io.npar_ac)+' coefficients','Simple quasiparticle approximation']
        print("# Parameters used:", file=fout)
        print("#  Analytic continuation (iop_ac) =", io.iop_ac, 'i.e., '+io.anc_type[io.iop_ac], file=fout)
        print("#  Fermi level shift    (iop_es)  =", io.iop_es, file=fout)
        print("#  Nr.freq points  (nomeg)        =", len(self.omega), file=fout)
        print("#  Number of AC poles (npar_ac/2) =", io.npar_ac/2, file=fout)
        isp=0
        
        EF_qp = 0
        bands = copy(self.Ebnd[isp])
        
        if not os.path.isfile(ddir+'/KSenk') and mrank==master:
            save(ddir+'/KSenk', bands[:,self.ibgw:nbnd+self.ibgw])
        
        # quasiparticle energies for G0W0 scheme
        nst,nend = self.ibgw,self.ibgw+nbnd
        (self.eqp0, eqp_im) = mcmn.Compute_quasiparticles(bands[:,nst:nend], self.Ebnd[isp][:,nst:nend], self.sigc, self.sigx, self.Vxct[:,nst:nend,nst:nend], self.omega, io, isp, fout, PRINT=True)
        
        # the fermi energy for G0W0 scheme
        print('total nval=', nval, 'but with ibgw=', self.ibgw, 'the resulting nval=', nval-self.ibgw*2, file=fout)
        (EF, Eg, evbm, ecbm, eDos) = mcmn.calc_Fermi(self.eqp0, kqm.atet, kqm.wtet, nval-self.ibgw*2, io.nspin)
        print(':E_FERMI_QP(eV)=  %12.4f' % (EF*H2eV,), file=fout)
        if io.iop_esgw0 == 1:
            self.eqp0 -= EF # shift bands so that EF=0
            EF_qp += EF # remember what shift we applied
            EF = 0      # now set EF to zero
        if Eg > 0:
            print(':BandGap_QP(eV)=  %12.4f' % (Eg*H2eV,), file=fout)
        else:
            print(':DOS_at_Fermi_QP= %12.4f' % (eDos,), file=fout)
        print('Fermi: evbm=%12.4f  ecbm=%12.4f ' % (evbm*H2eV, ecbm*H2eV), file=fout)
        # First analyzing Kohn-Sham bands
        (nomax,numin) = mcmn.Band_Analys(bands, self.EF, nbnd, 'KS', kqm, fout)
        # Next analyzing G0W0 bands
        (nomax,numin) = mcmn.Band_Analys(self.eqp0, EF, nbnd, 'GW', kqm, fout)
        if False and mrank==master:
            save(ddir+'/GW_qp', self.eqp0)
            save(ddir+'/KS_qp', bands[:,nst:nend])

        eqp = copy(self.eqp0)
        #if (False):
        if (nomax >= numin): # metallic
            print('metallic bands, we will not consider GW0 scheme', file=fout)
            return None
        else:                # insulating k-dependent gap
            if (nomax < numin): # insulating
                Egk = copy(bands[:,numin]-bands[:,nomax])
            else:
                Egk = copy(bands[:,numin])
            
            mix = io.mix_sc
            for isc in range(io.nmax_sc):
                # update_bande
                if io.shift_semicore:
                    de_vb = sum(eqp[:,nst]-bands[:,nst])/shape(bands)[0] # energy shift of the first calculated band
                else:
                    de_vb = 0
                bands[:,:nst] = bands[:,:nst] + de_vb                # shift semi-core bands so that they do not shift with respect to valence states
                bands[:,nst:nend] = copy(eqp[:,:])
                #bands[:,nst:nend] = bands[:,nst:nend]*(1-mix) + copy(eqp[:,:])*mix
                if io.iop_esgw0 == 1 and EF != 0:
                    bands -= EF # shift bands so that EF=0
                    if len(core.corind)>0: # also core if available
                        nsp, nat = len(core.eig_core), len(core.eig_core[0])
                        for isp in range(nsp):
                            for iat in range(nat):
                                core.eig_core[isp][iat][:] += de_vb - EF
                    EF_qp += EF # remember what shift we applied
                    EF = 0      # now set EF to zero
                
                
                if (nomax < numin): # insulating
                    Egk_new = copy(bands[:,numin]-bands[:,nomax])
                else:
                    Egk_new = copy(bands[:,numin])
                
                ediff = max(abs(Egk-Egk_new))
                Egk = Egk_new
                print('#scgw: isc=', isc, 'ediff=', ediff, 'de_vb=', de_vb, 'Egk=', (Egk*H2eV).tolist(), file=fout)
                #print  '#scgw: isc=', isc, 'ediff=', ediff, 'Egk=', Egk.tolist()
                io.out.flush()
                if ediff < io.eps_sc and isc>0: break

                # Recompute correlation self-energy using quasiparticle's green's function
                sigc = self.Compute_selfc(bands, core, kqm, fout, True)
                # mixing self-energy
                self.sigc = self.sigc*(1-mix) + mix*sigc
                #self.sigc = sigc
                
                # Compute the new quasiparticle energies
                io.iop_es = -1
                (eqp, eqp_im) = mcmn.Compute_quasiparticles(bands[:,nst:nend], self.Ebnd[isp][:,nst:nend], self.sigc, self.sigx, self.Vxct[:,nst:nend,nst:nend], self.omega, io, isp, fout, PRINT=True)
                
                # and recompute the Fermi energy on this quasiparticle bands
                (EF, Eg, evbm, ecbm, eDos) = mcmn.calc_Fermi(eqp, kqm.atet, kqm.wtet, core.nval-self.ibgw*2, io.nspin)
                
                print(':E_FERMI_QP(eV)=  %12.4f' % (EF*H2eV,), file=fout)
                #if io.iop_esgw0 == 1:
                #    eqp -= EF # shift bands so that EF=0
                #    EF_qp += EF # remember what shift we applied
                #    EF = 0      # now set EF to zero
                if Eg > 0:
                    print(':BandGap_QP(eV)=  %12.4f' % (Eg*H2eV,), file=fout)
                else:
                    print(':DOS_at_Fermi_QP= %12.4f' % (eDos,), file=fout)
                print('Fermi: evbm=%12.4f  ecbm=%12.4f ' % (evbm*H2eV, ecbm*H2eV), file=fout)
                print('eferqp0=', EF_qp, file=fout)
                
            if ediff >= 5e-2:
                print('WARNING : GW0 did not converge. Will not analyze', file=fout)
                return None
            else:
                mcmn.Band_Analys(eqp, EF, nbnd, 'GW0', kqm, fout)
                return eqp
    
class QPs:
    def __init__(self, rbas, tizmat, rmax, nkp, fout):
        " Creates the star of the space group. I think it might not work in non-symorphic groups, becuase of the screw axis is not used"
        for itt in range(5):    
            rvec = zeros(3)
            for i in range(3):
                rvec[i] = linalg.norm(rbas[i,:])

            nr = array(list(map(int,1./rvec * rmax)))*2
            self.rbas = rbas
            #print 'nr=', nr, 'rmax=', rmax
            rlen=[]
            rind=[]
            for ii,ir in enumerate(itertools.product(list(range(-nr[0],nr[0]+1)),list(range(-nr[1],nr[1]+1)),list(range(-nr[2],nr[2]+1)))):
                rvec = dot(ir,rbas)
                rr = linalg.norm(rvec)
                if (rr <= rmax):
                    rlen.append(rr)
                    rind.append(ir)
            indx = argsort(rlen) # kind='stable')  # just obtaining index to the sorted sequence
            # rearange arrays so that they are sorted
            self.rlen = zeros(shape(rlen))           # |R|
            self.rind = zeros(shape(rind), dtype=int)           # \vR in cartesian
            for i0,i in enumerate(indx):
                self.rlen[i0]   = rlen[i]
                self.rind[i0,:] = rind[i]
            invrind = -ones((2*nr[0]+1,2*nr[1]+1,2*nr[2]+1),dtype=int16)
            for i,r in enumerate(self.rind):
                invrind[r[0]+nr[0],r[1]+nr[1],r[2]+nr[2]] = i
            
            self.slen=[0.0]  # length of vector Rm in for each star
            self.rst = zeros((len(self.rlen),2),dtype=int)  # contains stars of distance R
            self.rst[0,1] = len(tizmat)                # R=0 is nsym degenerate
            ist=0                                 # R>0 will start with ist=1
            for ippw,r in enumerate(self.rind):
                if ippw==0: continue
                #print 'ippw=', ippw, 'r=', r, 'rst['+str(ippw)+',0]=',self.rst[ippw,0]
                if self.rst[ippw,0]==0:              # this r did not occur yet, hence it belongs to a new star
                    ist += 1                    # index of the new star
                    self.slen.append(self.rlen[ippw])
                    self.rst[ippw,0] = ist           # remember which star this r corresponds to
                    #print 'ist=', ist, 'rst['+str(ippw)+',0]='+str(ist), 'r=', r
                    for isym in range(len(tizmat)):  # now we go over all group operations, and generate all members of this star
                        r_star = dot(tizmat[isym],r) # new member of the star
                        jppw = invrind[r_star[0]+nr[0],r_star[1]+nr[1],r_star[2]+nr[2]] # should exist, where?
                        if jppw >= 0:
                            #print 'ist=', ist, 'rst['+str(jppw)+',0]='+str(ist), ' r_star=', r_star
                            self.rst[jppw,0] = ist  # this is in the same star
                            self.rst[jppw,1] += 1   # and how many times the same vector appears in this star, i.e., degeneracy
            self.nst = ist+1
            print('Number of k-points=', nkp, 'Number of stars=', self.nst, file=fout)
            if self.nst > nkp*2.0 :
                break
            rmax = rmax*1.3
            print('Since the number of stars should be substantially bigger than number of k-points, I need to increase rmax. New rmax=', rmax, file=fout)
            
    def ReadGap_qp(self,fname):
        fi = open(fname, 'r')
        dat = fi.next().split()
        (ib0_kip,ib1_kip,nkp1,nsp_qp) = list(map(int,dat[:4]))
        eferqp1 = float(dat[4])
        klist1 = zeros((nkp1,3),dtype=int)
        kvecs1 = zeros((nkp1,3))
        eks1 = zeros((nsp_qp,nkp1,ib1_kip-ib0_kip+1))
        eqp1 = zeros((nsp_qp,nkp1,ib1_kip-ib0_kip+1))
        for is1 in range(nsp_qp):
            nvbm = 0
            ncbm = ib1_kip
            for ik in range(nkp1):
                dat = list(map(int,fi.next().split()))
                ikvec, idv = dat[2:5], dat[5]
                klist1[ik] = ikvec
                kvecs1[ik] = array(ikvec)/float(idv)
                io, iu = ib0_kip-1, ib1_kip-1
                for ib in range(ib0_kip-1,ib1_kip):
                    line = next(fi)
                    #print '"'+line[4:24]+'"', '"'+line[24:44]+'"'
                    ii, eks1[is1,ik,ib], eqp1[is1,ik,ib] = int(line[:4]), float(line[4:24]), float(line[24:44])
                    if eks1[is1,ik,ib] < 0:
                        if io<ib : io=ib
                    else:
                        if iu>ib : iu=ib
                    #print 'ik=', ik, 'ib=', ib, 'eks1=', eks1[is1,ik,ib], 'eqp1=', eqp1[is1,ik,ib]
                next(fi)
                if io > nvbm : nvbm = io
                if iu < ncbm : ncbm = iu
            #print 'nvbm,ncbm=', nvbm+1,ncbm+1
            if nvbm >= ncbm:
                #print 'nvbm >= ncbm', nvbm+1, ncbm+1
                #print '  nvbm = ncbm-1 is forced'
                nvbm = ncbm - 1
        fi.close()
        return (kvecs1,eks1,eqp1)
    
    def Pickett_Interpolate(self, kvecs1, eks1, eqp1, kvecs2, eks2, fout):
        """ This interpolation algorithm is described in PRB 38, 2721 (1988).
        """
        nkp1, nb1 = shape(eks1)
        nkp2, nb2 = shape(eks2)
        dek1 = eqp1-eks1 # this is what we interpolate, i.e., only the difference between QP-bands and KS bands
        
        print('number of stars=', self.nst, 'number of computed k-points=', nkp1, file=fout)
        
        den = float(self.rst[0,1]) # number of all group operations, i.e., den = nsymp = rst[0,1]
        smat1 = zeros((nkp1,self.nst), dtype=complex) # "star function" of the input bands, small k-point mesh
        for ir,r in enumerate(self.rind):
            ist, pref = self.rst[ir,:]
            smat1[:,ist] += exp(2*pi*dot(kvecs1,r)*1j) * pref/den # This is the "star function" Sm[ik,ist]
        
        smat2 = zeros((nkp2,self.nst), dtype=complex)   # "star function" on the dense mesh of k-points
        for ir,r in enumerate(self.rind):
            ist, pref = self.rst[ir,:]
            smat2[:,ist] += exp(2*pi*dot(kvecs2,r)*1j) * pref/den # The star function Sm[k,ist]

        c1,c2 = 0.25,0.25      # The two coefficients mentioned in the paper as C1 and C2
        rho = zeros(self.nst)  # We use here rho = 1 - 2*c1*R^2 + c1^2*R^4 + c2*R^6
        rho[0]=1.
        rmin = self.slen[1]
        for ist in range(self.nst):
            x2 = (self.slen[ist]/rmin)**2
            x6 = x2**3
            rho[ist] = (1-c1*x2)**2 + c2*x6
        
        # Now we start solving the equations in the paper
        sm2 = zeros((nkp1-1,self.nst),dtype=complex) # sm2 <- Sm[k_i]-Sm[k_n] in the paper
        nb = min(nb1,nb2)                            # number of input energies.
        dele = zeros((nkp1-1,nb))
        
        for ik in range(nkp1-1):
            sm2[ik,:]  = smat1[ik+1,:]  - smat1[0,:]       #  sm2[ik,istar] = Sm[k_i]-Sm[k_0]
            dele[ik,:] = dek1[ik+1,:nb] - dek1[0,:nb]      #  dele <- e[k_j]-e[k_0] in the paper
        
        h = zeros((nkp1-1,nkp1-1),dtype=complex)          # H_ij in the paper
        for ik in range(nkp1-1):
            for jk in range(nkp1-1):
                h[ik,jk] += sum(sm2[ik,:]*conj(sm2[jk,:])/rho[:])
        
        Hinv = linalg.inv(h)
        Hf = dot(conj(sm2.T), Hinv)
        for ist in range(self.nst):
            Hf[ist,:] *= 1/rho[ist]
        coef = dot( Hf, dele )
        coef[0,:] = dek1[0,:nb] - dot(smat1[0,:],coef) # epsilon_m in the paper
        
        # dek2[ik,nb] = smat2.T[ik,ist] * coef[ist,ib]
        dek2 = dot(smat2, coef)     # this is the resulting energy on the dense grid : e_m * S_m(k) in the paper
        eqp2 = eks2 + dek2.real     # finally, adding back the Kohn-Sham energy
        return eqp2

#def toSmallerArray(ebnd2, nbs, nbe):
#    eks2 = zeros((len(ebnd2),nbe-nbs))
#    for ik in range(len(ebnd2)):
#        eks2[ik,:] = array(ebnd2[ik][nbs:nbe])
#    return eks2



#def SaveBandPlot(filename, bands, klist2, knames):
#    def PrintLegend(knames):
#        leg = '{'
#        for ik in range(len(knames)):
#            name = knames[ik].strip()
#            if name:
#                leg += str(ik)+':"'+name+'",'
#        leg += '}'
#        return leg
#    
#    nk, nb = shape(bands)
#    fgw = open(filename, 'w')
#    # k-path distance
#    xc = Path_distance( array(klist2) )
#    print('# leg=' + PrintLegend(knames), file=fgw)
#    for ik in range(len(klist2)):
#        print('%10.6f  ' % (xc[ik],), ('%14.8f '*nb) % tuple(bands[ik,:]*H2eV), file=fgw)
#    fgw.close()


def Find_which_Gs_from_pw_are_present(pw, heads, all_Ek, all_Gs):
    #
    indgkir=[]
    for irk in range(len(all_Ek)):
        k, kname, _wgh_, _ios_, n0, nb = heads[irk]
        nG,n3 = shape(all_Gs[irk])
        ind = zeros(nG,dtype=int)
        for i in range(nG):
            iG = all_Gs[irk][i]          # reciprocal vector read from vector file
            ii = pw.ig0[tuple(iG)]   # what is the index of this G in my gvec (previously generated fixed reciprocal mesh)?
            if (ii not in ind):
                ind[i] = ii
                #print('irk=%3d i=%3d indgkir=%3d' % (irk+1, i, ind[i]), 'iG=', iG )#, file=fout)
            #else:
                #print('irk=%3d i=%3d indgkir=%3d' % (irk+1, i, ind[i]), 'iG=', iG, 'this is LO' )#, file=fout)
                
        indgkir.append(ind)
    return indgkir


def Find_which_G_present_everywhere(mhsrws, indgkir):
    Gs_present=[]
    Gs_index = -ones(mhsrws, dtype=int)
    ini=0
    for ii in range(mhsrws):
        GG_present_everywhere  = True
        for irk in range(len(indgkir)):
            GG_present_everywhere = GG_present_everywhere and (ii in indgkir[irk])
        if GG_present_everywhere:
            Gs_present.append(ii)
            Gs_index[ii] = ini
            ini+=1
    return (Gs_present, Gs_index)


def Find_which_G_present(irk1,irk2,hsrws,indgkir):
    """
    It turns out that wien2k uses different order and set of G vectors at different k-points. This has to be remedied in order to compute overlaps.
    Here we assembe a list of G-points which are common to two different k-points, irk1 and irk2. If G_ind[ii]>=0, then this ii is present with index G_ind[ii].
    The total number of common G-points is ni.
    """
    mhm = min(hsrws[irk1],hsrws[irk2])
    G_ind = -ones(mhm, dtype=intc)
    ni=0
    for ii in range(mhm):
        if (ii in indgkir[irk1]) and (ii in indgkir[irk2]):
            G_ind[ii] = ni
            ni += 1
    return(ni, G_ind)

def Check_ek_equal(all_Ek, ebnd2, band_vector_file, band_energy_file):
    for irk in range(len(all_Ek)):
        if len(all_Ek[irk]) != len(ebnd2[irk]):
            print('ERROR ', band_vector_file, 'and', band_energy_file, 'incompatible len(ek1)=', len(all_Ek[irk]), 'len(ek2)=', len(ebnd2[irk]))
        diff = sum(abs(all_Ek[irk]-ebnd2[irk]))
        if diff > 1e-3:
            print('ERROR enrgies from', band_vector_file, 'and', band_energy_file, 'incompatible diff=', diff)
    return True

    
def PrintM(A):
    ni,nj = shape(A)
    for i in range(ni):
        for j in range(nj):
            print('%6.3f '% A[i,j], end='')
        print()
    
@jit(nopython=True)
def FindDegeneracies(eks):
    """ Returns a list of degeneracies in eks[:]. Assumes that eks[:] contains a sorted array of energies.
    """
    # find all degenerices
    degs=[]
    previously_degenerate=False
    for i in range(1,len(eks)):
        if abs(eks[i-1]-eks[i])<1e-6:
            if previously_degenerate:
                degs[-1].append(i)
            else:
                previously_degenerate = True
                degs.append([i-1,i])
        else:
            previously_degenerate = False
    return degs


@jit(nopython=True)
def FillPsiChi(psi_chi, Ask, indgkir_ik, G_ind, nbs, nbe):
    """
    After reading vector file, we want to take <psi_k|chi_{k+G}> in sorted order, so that G vectors are sorted the same way in all k-points.
    It turns out that wien2k uses different order and set of G vectors at different k-points. This has to be remedied in order to compute overlaps.
    Here we use the index to common G-points, stored in G_ind[ii], and assemble <psi_{k,i}|chi_{k+G}> == psi_chi[i,iG]
    """
    for i,ii in enumerate(indgkir_ik):
        if ii < len(G_ind) and G_ind[ii]>=0:
            iii = G_ind[ii]
            psi_chi[:,iii] = Ask[nbs:nbe,i]
    return psi_chi

def Overlap_Eigvec_PW(Ask,indgkir_ik,G_ind,nbs,nbe,ni):
    """ Here we compute the singular value decomposition <psi_{k,i}|chi_{k+G}> == psi_chi[i,iG] and take the unitary part of it.
    After reading vector file, we want to take <psi_k|chi_{k+G}> in sorted order, so that G vectors are sorted the same way in all k-points.
    It turns out that wien2k uses different order and set of G vectors at different k-points. This has to be remedied in order to compute overlaps.
    We first call the above routine FillPsiChi, which uses the index to common G-points, stored in G_ind[ii], and assemble <psi_{k,i}|chi_{k+G}> == psi_chi[i,iG]
    And later we perform svd on <psi_{k,i}|chi_{k+G}>, i.e., <psi_{k,i}|chi_{k+G}> = u*s*vh and we return u*vh unitary part
    """
    psi_chi = zeros((nbe-nbs,ni))
    psi_chi = FillPsiChi(psi_chi, Ask, indgkir_ik, G_ind, nbs, nbe)
    u, s, vh = linalg.svd(psi_chi, full_matrices=True)
    uv = dot(u, vh[:nbe-nbs,:])
    return uv

def FindNextShell(shell, sdstnc, debug=False):
    shell1=[]
    to_delete=[]
    path=[]
    dist=0

    for ii,(ik,d) in enumerate(sdstnc):
        if dist!=0 and abs(d-dist) > 1e-3: break
        
        k0_in_shell = ik[0] in shell
        k1_in_shell = ik[1] in shell
        one_in_shell = (k0_in_shell and not(k1_in_shell)) or (k1_in_shell and not(k0_in_shell))
        
        if one_in_shell and (dist==0 or abs(d-dist)<1e-6):
            if k0_in_shell:
                shell1.append( ik[1] )
                path.append( (ik,d) )
            else:
                shell1.append( ik[0] )
                path.append( ((ik[1],ik[0]),d) )
            to_delete.append(ii)
            dist = d
    
    for ii in to_delete[::-1]:
        del sdstnc[ii]

    return ( list( set(shell1) ), path )


def ResortEnergies(uv0, korder, s_eks, s_uvs, all_Ek, all_As, indgkir, Gs_index, nbs, nbe):
    #
    ini = max(Gs_index)+1
    nbnd = nbe-nbs
    #
    for iik,irk in enumerate(korder): #range(len(all_Ek)):
        eks = all_Ek[irk][nbs:nbe]
        print('ik=', irk, 'starting energy=')
        print( '%8.4f, '*len(eks) % tuple(eks) )
        # Finds degeneracy of the eigenvalues, and later it will optimize the overlap of degenerate eigenvectors
        degs = FindDegeneracies(eks)
        print(irk, 'degs=', degs)
        # After reading vector file, we want to take <psi_k|chi_{k+G}> in sorted order, so that G vectors are sorted the same way in all k-points.
        # It turns out that wien2k uses different order and set of G vectors at different k-points. This has to be remedied in order to compute overlaps.
        # After we assemble <psi_k|chi_{k+G}>, we compute its svd = <psi_{k,i}|chi_{k+G}> = u_{i,a} s_a vh_{a,G}, and unitary transformation becomes uv1 = u_{i,a}*vh_{a,G}
        # he closest unitary transformation between the set of bands psi_{k,i} and G vectors chi_{k+G} is uv1_{i,G} = u_{i,a}*vh_{a,G}
        uv1 = Overlap_Eigvec_PW(all_As[irk], indgkir[irk], Gs_index, nbs,nbe, ini)
        
        # Now we compute overlap between uv from previous k-point and this k-point
        # which is Barry connection <psi_{k-previous}|d/dk|psi_k> = psi_psi_{i,j} = \sum_G <psi_{k_0,i}|chi_{G+k_0}><chi_{G+k}|psi_{k,j}>
        psi_psi = dot(uv0, conj(uv1).T)
        
        PrintM(psi_psi)
        # If eigenvalues are degenerate, eigenvectors can be arbitrarily mixed up. We will find the optimal linear combination of eigenvalues, which maximizes overlap <psi_{k-previous}|psi_k>
        # If <psi_{k-previous,i}|psi_{k,j}> = oo(i,j) is degenerate in a small degenerate subspace of (i,j), than we will find a closest unitary transformation oo^{-1}, namely closet(oo^{-1})==ut.
        # More precisely, if svd of oo gives, oo = u*s*vh, than u*vh is close to oo, and (u*vh)^{-1} is close to oo^{-1} and unitary, and therefore <psi_{k-previous,i}|psi_{k,j}>*(u*vh)_{jj'}
        # should be much closer to unity, and hence prefered linear combination of degenerate eigenvectors.
        for dg in degs:
            # dg contains degenerate bands at this k-point
            # isi are degenerate bands from previous k-point, which correspond to degenerate subspace dg.
            isi=[]
            cols = psi_psi[:,dg]                                           # just a few degenerate columns of <psi_{k_previous,i}|psi_{k,degenerate}>
            in_previous_k = [(i,sum(cols[i,:]**2)) for i in range(nbnd)]   # now we compute in_previous_k(i)=(i, \sum_degenerate (<psi_{k_previous,i}|psi_{k,degenerate}>)^2 )
            in_previous_k = sorted(in_previous_k, key=lambda x:-x[1])      # now we sort that, so that i with largest over to degenerate subgroup comes first
            isi = sorted([in_previous_k[i][0] for i in range(len(dg))])    # now we just take the first few from previous k-point that span the same dimension as degenerate k-points at this k-point
            oo = (psi_psi[isi,:])[:,dg]                                    # now oo is a small n x n matrix in this degenerate subspace of psi_psi
            u, s, vh = linalg.svd(oo, full_matrices=True)                  # the small matrix is svd and the closest unitary matrix of oo^{-1} is found
            ut = linalg.inv(dot(u,vh))
            # Here we actually take the relevant linear combination of eigenvectors psi_chi(i,k+G) = ut.T(i,i')<psi_{i',k}|chi_{k+G}>
            uv1[dg,:] =  dot(ut.T, uv1[dg,:])
            # Finally, we use this closest(oo^{-1})==ut matrix to transform psi_psi. This is only for debugging purposes. We will not use this further
            npsi_psi = dot(psi_psi[:,dg],ut)
            # 
            print('dg=', dg, 'isi=', isi, 'oo=', list(oo), 's=', s)
        if degs: # If any degeneracy occurs in the eigensystem, psi_chi was changed, and now we recompute psi_psi between previous and this k-point
            # psi_chi = <psi_{i,k}|chi_{k+G}>, which is decomposed by u*s*vh and its unitary component u*vh is used to find relation between this and previous k-point
            psi_psi = dot(uv0, conj(uv1).T) # <psi_{i,k_previous}|chi_{k+G}><chi_{k+G}|psi_{j,k}>
            PrintM(psi_psi)

        # Now we will try to rearange the order of bands in this k-point so that the previous and this k-point are as close as possible, i.e., we will
        # resort index j in psi_psi(i,j)=<psi_{i,k_previous}|psi_{j,k}> so that the psi_psi is as close to unity as possible.
        # indx will contain the index array, and allrows contains information about which columns have been already used.
        indx = array(arange(nbnd),dtype=int)
        allrows = list(range(nbnd))
        allcols = list(range(nbnd))
        # We will start rearanging those bands that show the largest overlap, i.e., where abs(psi_psi(i,j)) is largest and close to unity.
        # Finds largest matrix element
        iijj = argmax(abs(psi_psi))
        # indices of the largest matrix element, and we will start with jjs column in the loop below
        iis, jjs = int(iijj/nbnd), iijj%nbnd
        for iijj in range(nbnd):
            col = abs(psi_psi[:,jjs])
            print('before=', list(col))
            col += 0.5/((arange(nbnd)-jjs)**2+1)    # THIS IS THE LAST CHANGE: 2022: Added a small term to not exchange columns which are very far apart. We get bonus of 0.3 for columns close by.
            print('after=', list(col))
            ii = argmax(col)     # for current band jjs, ii is likely the corresponding band in previous k-point.
                
            print('jcol=', jjs, 'irow=', ii, (ii in allrows), '%10.6f' % col[ii], 'rows_left=', allrows, 'cols_left=', allcols)
            while ii not in allrows: # but if ii was already used in combination with some previous eigenvector (which has even larger overlap), than we can not use it here.
                col[ii]=0        # Go for next largest
                ii = argmax(col) # by setting this one to zero.
            # We found corresponding band at previous k-point. Namely band jjs at this k-point corresponds to band ii at previous k-point. Hence store its index
            indx[ii]=jjs
            allrows.remove(ii)  # also remember that ii was alredy used
            allcols.remove(jjs) # and remember that jjs was used

            # below we determine which column in <psi_{k_previous}|psi_{k,jjs}> should we check now, i.e., what is the new jjs
            nn = len(allcols) # how many columns/rows are left
            if nn>1:
                left_here = abs(psi_psi[allrows,:][:,allcols]) # these rows and columns of psi_psi are still to be determined
                iijj = argmax(left_here)                       # we again check what is the largest element in the remaining matrix
                i, j = int(iijj/nn), iijj%nn                   # and these are the 2d indices to this largest element
                iis, jjs = allrows[i], allcols[j]              # however, the indices of the original large array are actually (iis, jjs)
                print('jnext=', jjs, (iis,jjs), (i,j), '%12.7f' % (left_here[i,j],))
            elif nn==1:
                jjs = allcols[0]      # this is the only entry left
                
        # Now we have the index table to rearange energies and eigenvectors
        print(irk, 'indx=',  [ (j, indx[j]) for j in range(len(indx)) if j!=indx[j] ] )  # This is now the order in which the bands should be resorted.
        
        indx_is_identity = array_equal(indx, range(nbnd)) # check if we need to rearange eigenvectors at all.
        if not indx_is_identity: # if we have something to rearange
            # yes, eigenvectors and eigenvalues need to be rearanged.
            eks = eks[indx]   # rearanging energies
            # We rearange uv1, which is unitary part of <psi_{k_previous,i}|psi_{k,j}>
            nuv1 = copy(uv1)
            for i in range(nbnd):
                if indx[i]!=i:
                    nuv1[i,:] = uv1[indx[i],:]  # resorting eigenvectors in psi_chi[iband,iG]
            uv1 = nuv1
            # and our final overlap psi_psi is determined here
            psi_psi = dot(uv0, conj(uv1).T)   # and now we find overlap between psi from previous k-point and this k-point.
            #
            print('energy after correction=')
            print( '%8.4f, '*len(eks) % tuple(eks) )
            print(irk, 'after correction')
            PrintM(psi_psi)
        else:
            print('indx=identity')
        # remeber energy 
        s_eks[irk,:] = eks[:]
        s_uvs[irk] = uv1
        # remember uv
        uv0 = copy(uv1)
    return (s_eks, s_uvs)

class WannierInterpolation:
    def __init__(self, io, nbs, nbe, strc, latgen, kqm, fout):
        self.w90_exe = shutil.which('wannier90.x')
        if self.w90_exe is None:
            print('ERROR: Expecting wannier90.x in path, but could not find it', file=fout)
            print('ERROR: Expecting wannier90.x in path, but could not find it')
            sys.exit(0)
        self.case = io.case
        
        self.nbs, self.nbe = nbs, nbe
        self.strc = strc
        self.latgen = latgen
        self.kqm = kqm
        #self.strc_vpos, self.strc_mult, self.strc_aname = strc.vpos, strc.mult, strc.aname
        #self.latgen_rbas = latgen.rbas
        self.w90 = CWannier90(self.nbs,self.nbe, io.case, self.strc.mult, self.strc.aname, self.kqm.ndiv, self.kqm.klist, self.kqm.LCM, self.kqm.k2icartes)
        
    def Compute_Amn_Mmn(self, io, wcore, k_path, fout):
        ks = KohnShamSystem(io.case, self.strc, io.nspin, fout)
        in1 = w2k.In1File(io.case, self.strc, fout, io.lomax)
        ks.Set_nv(in1.nlo_tot)
        pw = PlaneWaves(ks.hsrws, io.kmr, io.pwm, io.case, self.strc, in1, self.latgen, self.kqm, False, fout)
        ks.VectorFileRead(io.case, self.strc, self.latgen, self.kqm, pw, fout, in1)
        (Elapw, Elo) = w2k.get_linearization_energies(io.case, in1, self.strc, io.nspin, fout)
        in1.Add_Linearization_Energy(Elapw, Elo) # We could use linearlization energies from vector file ks.Elo, however, the local orbital energy positions in vector file depend on lomax variable in wien2k during vector file writing....
        
        Vr = w2k.Read_Radial_Potential(io.case, self.strc.nat, io.nspin, self.strc.nrpt, fout)
        radf = w2k.RadialFunctions(in1,self.strc,ks.Elapw,ks.Elo,Vr,io.nspin,fout)
        del Vr
        radf.get_ABC(in1, self.strc, fout)
        
        (EF, Eg, evbm, ecbm, eDos) = mcmn.calc_Fermi(ks.Ebnd[0], self.kqm.atet, self.kqm.wtet, wcore.nval, ks.nspin)
        ks.Ebnd -= EF
        if Eg >= 0:
            print('\n'+'-'*32+'\nFermi: Insulating, KS E_Fermi[eV]=%-12.6f Gap[eV]=%-12.6f  EVBM[eV]=%-12.6f  ECBM[eV]=%-12.6f' % (EF*H2eV, Eg*H2eV, evbm*H2eV, ecbm*H2eV), file=fout)
        else:
            print('\n'+'-'*32+'\nFermi: Metallic, KS E_Fermi[eV]=%-12.6f  DOS[E_f]=%-12.6f' % (EF*H2eV, eDos), file=fout)
            
        
        self.orbs = self.w90.FindRelevantOrbitals(self.strc, self.latgen, ks, self.kqm, pw, in1, radf, fout)
        
        self.w90.Prepare_win_file(self.orbs, self.latgen.rbas, self.strc.vpos, self.strc.mult, self.strc.aname, k_path, semi_cartesian=True)
        
        # execute wannier90.x -pp case.win
        process = subprocess.Popen([self.w90_exe, '-pp', io.case+'.win'], stdout=fout)
        process.wait()
        
        self.w90.Compute_and_Save_Projection_amn(io.case, self.orbs, self.strc, self.latgen, ks, self.kqm, pw, in1, radf, fout, naive_projection=True)
        
        (pair, pair_umklap) = self.w90.Read_nnkp()
        self.w90.Find_All_Possible_kpair_distances(pair, pair_umklap)
        self.w90.Compute_and_Save_BandOverlap_Mmn(pair, pair_umklap, self.case, self.orbs, self.strc, self.latgen, ks, self.kqm, pw, in1, radf, fout, DMFT1=False, naive_projection=True, debug=False)
        
        self.ks_Ebnd = ks.Ebnd[0,:,:]
        
        self.InitialRun=True
        
        
    def Wannierize(self, eqp1, klist2, eks2, kind2, k_path, fout):
        nkp,nb2 = shape(eqp1)
        Ediff = zeros((nkp,self.nbe))
        Ediff[:,self.nbs:self.nbe] = eqp1[:,:self.nbe-self.nbs]
        self.w90.Save_Eigenvalues(self.case, Ediff, self.kqm)  # Does not change units anymore
        
        if self.InitialRun:
            self.InitialRun=False
        else:
            self.w90.Prepare_win_file(self.orbs, self.latgen.rbas, self.strc.vpos, self.strc.mult, self.strc.aname, k_path, semi_cartesian=True, _w90p_={'restart':'plot'})
        # Execute wannier90. At the first pass it does not have restart option. Later it is just restart.
        process = subprocess.Popen([self.w90_exe, self.case+'.win'], stdout=fout, stderr=subprocess.PIPE, universal_newlines=True)
        process.wait()
        output, errors = process.communicate()
        if errors: print('ERROR:', errors)
        # got bands on the path, but points are choose differently here, so we need to interpolate.
        (x_k, gw_k, kind) = self.w90.ReadBandsAfter(self.case)
         
        Ediff[:,self.nbs:self.nbe] = self.ks_Ebnd[:,self.nbs:self.nbe]
        self.w90.Save_Eigenvalues(self.case, Ediff, self.kqm)  # Does not change units anymore
        self.w90.Prepare_win_file(self.orbs, self.latgen.rbas, self.strc.vpos, self.strc.mult, self.strc.aname, k_path, semi_cartesian=True, _w90p_={'restart':'plot'})
        process = subprocess.Popen([self.w90_exe, self.case+'.win'], stdout=fout, stderr=subprocess.PIPE, universal_newlines=True)
        process.wait()
        output, errors = process.communicate()
        if errors: print('ERROR:', errors)
        # got bands on the path, but points are choose differently here, so we need to interpolate.
        (x_k, dft_k, kind) = self.w90.ReadBandsAfter(self.case)
        
        Ediff_k = gw_k - dft_k
        #for ib in range(self.nbe-self.nbs):
        #    plot(x_k, Ediff_k[:,ib])
        #show()
        #sys.exit(0)
        
        # Distance on the path defined by klist2 (used in wien2k and dft run)
        x_c = Path_distance( array(klist2), self.kqm.k2cartes )
        eqp2 = zeros(shape(eks2))
        for ib in range(self.nbe-self.nbs):
            fEdiff = interpolate.UnivariateSpline(x_k, Ediff_k[:,ib], s=0)  # mesh from wannier90 converted to wien2k klist2 mesh
            eqp2[:,ib] = eks2[:,ib] + fEdiff(x_c) # now interpolation of the difference executed and stored.
        
        (nkp,nb) = shape(eqp1)

        savetxt('eqp1.dat', vstack((range(nkp),eqp1.T*H2eV)).T)
        savetxt('ebdn.dat', vstack((range(nkp),self.ks_Ebnd[:,self.nbs:self.nbe].T*H2eV)).T)
        
        PrintBands('ediff.dat', x_k, Ediff_k*H2eV, kind) # Ediff in eV
        PrintBands('eks2.dat',  x_k, dft_k*H2eV, kind2)    # 
        PrintBands('eqp2.dat',  x_k, gw_k*H2eV, kind2)     # eqp2 in Hartree
        
        return eqp2

    def OnlyRerunWannier(self, eqp1, klist2, eks2, kind2, wcore, fout):
        nkp,nb2 = shape(eqp1)

        ks = KohnShamSystem(io.case, self.strc, io.nspin, fout)
        in1 = w2k.In1File(io.case, self.strc, fout, io.lomax)
        ks.Set_nv(in1.nlo_tot)
        pw = PlaneWaves(ks.hsrws, io.kmr, io.pwm, io.case, self.strc, in1, self.latgen, self.kqm, False, fout)
        ks.VectorFileRead(io.case, self.strc, self.latgen, self.kqm, pw, fout, in1)
        self.w90.Check_Irreducible_wedge(ks, self.kqm, fout)
        
        
        Ediff = zeros((nkp,self.nbe))
        Ediff[:,self.nbs:self.nbe] = eqp1[:,:self.nbe-self.nbs]
        self.w90.Save_Eigenvalues(self.case, Ediff, self.kqm)  # Does not change units anymore
        
        self.w90.CheckIsRestart()
        
        # Execute wannier90. At the first pass it does not have restart option. Later it is just restart.
        process = subprocess.Popen([self.w90_exe, self.case+'.win'], stdout=fout, stderr=subprocess.PIPE, universal_newlines=True)
        process.wait()
        output, errors = process.communicate()
        if errors: print('ERROR:', errors)
        # got bands on the path, but points are choose differently here, so we need to interpolate.
        (x_k, gw_k, kind) = self.w90.ReadBandsAfter(self.case)

        
        (EF, Eg, evbm, ecbm, eDos) = mcmn.calc_Fermi(ks.Ebnd[0], self.kqm.atet, self.kqm.wtet, wcore.nval, ks.nspin)
        ks.Ebnd -= EF
        if Eg >= 0:
            print('\n'+'-'*32+'\nFermi: Insulating, KS E_Fermi[eV]=%-12.6f Gap[eV]=%-12.6f  EVBM[eV]=%-12.6f  ECBM[eV]=%-12.6f' % (EF*H2eV, Eg*H2eV, evbm*H2eV, ecbm*H2eV), file=fout)
        else:
            print('\n'+'-'*32+'\nFermi: Metallic, KS E_Fermi[eV]=%-12.6f  DOS[E_f]=%-12.6f' % (EF*H2eV, eDos), file=fout)
        self.ks_Ebnd = ks.Ebnd[0,:,:]
        
        
        
        Ediff[:,self.nbs:self.nbe] = self.ks_Ebnd[:,self.nbs:self.nbe]
        self.w90.Save_Eigenvalues(self.case, Ediff, self.kqm)  # Does not change units anymore

        
        process = subprocess.Popen([self.w90_exe, self.case+'.win'], stdout=fout, stderr=subprocess.PIPE, universal_newlines=True)
        process.wait()
        output, errors = process.communicate()
        if errors: print('ERROR:', errors)
        # got bands on the path, but points are choose differently here, so we need to interpolate.
        (x_k, dft_k, kind) = self.w90.ReadBandsAfter(self.case)
        
        Ediff_k = gw_k - dft_k
        
        # Distance on the path defined by klist2 (used in wien2k and dft run)
        x_c = Path_distance( array(klist2), self.kqm.k2cartes )
        eqp2 = zeros(shape(eks2))
        for ib in range(self.nbe-self.nbs):
            fEdiff = interpolate.UnivariateSpline(x_k, Ediff_k[:,ib], s=0)  # mesh from wannier90 converted to wien2k klist2 mesh
            eqp2[:,ib] = eks2[:,ib] + fEdiff(x_c) # now interpolation of the difference executed and stored.
        
        (nkp,nb) = shape(eqp1)

        savetxt('eqp1.dat', vstack((range(nkp),eqp1.T*H2eV)).T)
        savetxt('ebdn.dat', vstack((range(nkp),self.ks_Ebnd[:,self.nbs:self.nbe].T*H2eV)).T)
        
        PrintBands('ediff.dat', x_k, Ediff_k*H2eV, kind) # Ediff in eV
        PrintBands('eks2.dat',  x_k, dft_k*H2eV, kind2)    # 
        PrintBands('eqp2.dat',  x_k, gw_k*H2eV, kind2)     # eqp2 in Hartree
        
        return eqp2
    

def Find_kpts_in_path(k_path, kqm, io_k0shift, divs):
    kpts = [k[1] for k in k_path]                    # take out k-points in k_path
    x_c = Path_distance( array(kpts), kqm.k2cartes ) # calculate distance between points on the path
    icartes2f = linalg.inv(kqm.k2icartes)  # how to get fractional k-points
    
    kshft = array(io_k0shift[:3]) # shift of k-mesh in calculations
    
    kpts_in_path=[]
    for jk in range(len(kpts)-1):   # over all segments of the k-path
        k1, k2 = dot(icartes2f, kpts[jk]), dot(icartes2f, kpts[jk+1]) # two points define a segment
        d1, d2 = x_c[jk], x_c[jk+1]                                   # distance on the two points definint the segment
        x2 = k2-k1                   # if a k-points lies on the segment, we must have k[i]-k1[i] = (k2[i]-k1[i])*x
        ii=0                         # where x is a number between 0-1. We need to find ii for which k2[ii]-k1[ii]!=0.
        while(abs(x2[ii])<1e-6):
            ii+=1
        for ik in range(len(kqm.kii_ind)):  # These arereducible  k-points in our calculated mesh
            k = (kqm.indx2k(ik) + kshft/2) * 1/divs # actual k-point in fractional representation
            irk = kqm.kii_ind[ik]           # this is its irreducible counterpart
            x1 = (k-k1)                     # now we prepare k-k1, which needs to be compared with k2-k1=x2
            x = x1[ii]/x2[ii]               # we checked that x2[ii]!=0, hence can compute the distance along the segment, which is x.
            if sum(abs(x1[:]-x*x2[:]))<1e-6 and x>-1e-6 and x<1+1e-6: # checking that this point is indeed along the segment
                d = d1+(d2-d1)*x             # in which case all three components give the same x, i.e., distance along the segment
                #print('Found x=', x, 'd=', d)  # distance d from the begining of k-path
                kpts_in_path.append([d, irk])  # remember such k-point and at which distance it appears
            #print('   ik=', ik, irk, k, 'x=', x)
    # sorting with respect to distance along the path
    kpts_in_path = sorted(kpts_in_path, key=lambda x: x[0])
    # some appear multiple times, and should be eliminated
    eliminate=[]
    for i in range(len(kpts_in_path)-1):
        if abs(kpts_in_path[i][0]-kpts_in_path[i+1][0])<1e-6:
            eliminate.append(i+1)
    #print('All:=', kpts_in_path)
    for i in eliminate[::-1]:
        del kpts_in_path[i]
    #print('eliminate=', eliminate)
    return kpts_in_path

def FindIndex(ediff_plot, ediff_data):
    nb = len(ediff_plot)
    dst = zeros((nb,nb))
    for i in range(nb):
        dst[i,:] = abs(ediff_data[:]-ediff_plot[i])
    
    if False:
        dstf = ravel(dst)
        iind = sorted(range(len(dstf)), key=lambda x: dstf[x])
        for i in range(nb):
            ii = iind[i]
            l, m = int(ii/nb), ii%nb
            print('ii=%3d l=%2d m=%2d %12.7g' % (ii, l, m, dst[l,m]))
        print()
    
    allrows = list(range(nb))
    allcols = list(range(nb))
    
    ind = zeros(nb, dtype=int)
    for i in range(nb):
        left_here = dst[allcols,:][:,allrows]
        mleft_here = ravel(left_here)
        
        ii = argmin(mleft_here)
        il, im = int(ii/(nb-i)), ii%(nb-i)
        l, m = allcols[il], allrows[im]

        col = dst[l,:]
        m = argmin(col)
        while m not in allrows: # it was already used
            col[m] = 1000  # set to large value so that it is eliminated in argmin
            m = argmin(col)# find the next largest
        ind[l] = m
        #print('ii=', ii, 'l=', l, 'm=', m, 'dst=', dst[l,m], 'ind['+str(l)+']=', m)
        allrows.remove(m)
        allcols.remove(l)
    return ind


def main(band_energy_file, band_EFermi, io, mode):
    fout = io.out
    
    strc = w2k.Struct(io.case, fout)
    latgen = w2k.Latgen(strc, fout)
    latgen.Symoper(strc, fout)
    
    divs = array(io.nkdivs, dtype=int)
    kqm = KQmesh(divs, io.k0shift, strc, latgen, fout)
    kqm.tetra(latgen, strc, fout)
    wcore = w2k.CoreStates(io.case, strc, io.nspin, fout)
    io_data={ 'emax_pol': io.emax_pol, 'emax_sc':  io.emax_sc, 'iop_core': io.iop_core, 'efermi': io.efermi, 'ibgw': io.ibgw, 'nbgw': io.nbgw, 'emingw': io.emingw, 'emaxgw': io.emaxgw, 'ibgw':io.ibgw, 'emingw':io.emingw, 'emaxgw':io.emaxgw}
        
    ansp = SCGW0(io)
    ansp.ReadKSEnergy(io.case, io.nspin, wcore, strc, kqm, io_data, fout)
    
    if band_energy_file and mrank==master: # Can produce the band plot
        eks1 = load(ddir+'/KS_qp.npy')
        nkp1 = shape(eks1)[0]
        
        klist1 = kqm.kirlist/float(kqm.LCM) # k-points on which Sigma exists
        qps = QPs(latgen.rbas, latgen.tizmat, io.rmax, nkp1, fout ) # will compute quasiparticle energies on computed k-point mesh klist1
        
        
        (klist2, wegh, ebnd2, hsrws, knames) = w2k.Read_energy_file(band_energy_file, strc, fout, give_kname=True)
        nbs = ansp.ibgw # ths first band to use in interpolation
        nbe = min([len(ebnd2[ik]) for ik in range(len(ebnd2))]) # the last band to use
        nbe = min(shape(eks1)[1]+ansp.ibgw, nbe)
        if io.nbgw>0: nbe = min(io.nbgw, nbe)
        eks2 = array( [ebnd2[ik][nbs:nbe] for ik in range(len(ebnd2))] )*Ry2H - band_EFermi
        
        k_path=[]
        kind2=[]
        for ik in range(len(klist2)):
            kname = knames[ik].strip()
            if kname:
                k_path.append([kname, klist2[ik] ])
                kind2.append([ik,kname])
        
        kvecs2 = mcmn.cart2int(klist2,strc,latgen)
        kvecs1 = mcmn.cart2int(klist1,strc,latgen)
        
        
        SaveBandPlot(ddir+'/KS_bands.dat', eks2, klist2, kqm.k2cartes, knames)
        
        qpfile = ddir+'/GW_qp.npy'
        print('mode=', mode, file=fout)
        if os.path.isfile(qpfile):
            eqp1 = load(ddir+'/GW_qp.npy')
            wint = WannierInterpolation(io, nbs, nbe, strc, latgen, kqm, fout)

            if mode in ['all', 'Pickett']:
                eqp20 = qps.Pickett_Interpolate(kvecs1, eks1, eqp1, kvecs2, eks2, fout)
                SaveBandPlot(ddir+'/GW_bands_Pickett.dat', eqp20, klist2, kqm.k2cartes, knames)
                SaveBandPlot(ddir+'/Diff_bands_Pickett.dat', eqp20-eks2, klist2, kqm.k2cartes, knames)
                
            if mode=='all':
                wint.Compute_Amn_Mmn(io, wcore, k_path, fout)
                eqp21 = wint.Wannierize(eqp1, klist2, eks2, kind2, k_path, fout)
            elif mode=='Wannier':
                eqp21 = wint.OnlyRerunWannier(eqp1, klist2, eks2, kind2, wcore, fout)
            if mode in ['all', 'Wannier']:
                SaveBandPlot(ddir+'/GW_bands_Wannier.dat', eqp21, klist2, kqm.k2cartes, knames)
                SaveBandPlot(ddir+'/Diff_bands_Wannier.dat', eqp21-eks2, klist2, kqm.k2cartes, knames)

    eqpn = None
    if io.save_mwm:
        # This is the self-consistent GW0 run in which self-consistent bands are used to calculate self-energy
        eqpn = ansp.calceqp(io, strc, kqm, wcore.nval, wcore, fout)
        
        if band_energy_file and mrank==master: # Can produce band plot
            if mode in ['all', 'Pickett']:
                eqp02 = qps.Pickett_Interpolate(kvecs1, eks1, ansp.eqp0, kvecs2, eks2, fout)
                SaveBandPlot(ddir+'/G0W0_bands_Pickett.dat', eqp02, klist2, kqm.k2cartes, knames)
                
            if mode=='all' and wint.InitialRun:
                wint.Compute_Amn_Mmn(io, wcore, k_path, fout)
                eqp21 = wint.Wannierize(ansp.eqp0, klist2, eks2, kind2, k_path, fout)
            elif (mode=='all' and not wint.InitialRun) or mode=='Wannier':
                f_out=open(os.devnull, 'w')
                eqp21 = wint.OnlyRerunWannier(ansp.eqp0, klist2, eks2, kind2, wcore, f_out)
            if mode in ['all', 'Wannier']:
                SaveBandPlot(ddir+'/G0W0_bands_Wannier.dat', eqp21, klist2, kqm.k2cartes, knames)
            
    #if eqpn is not None and mrank==master:
    #    save(ddir+'/GW0_qp', eqpn)
    #    if band_energy_file: # Can produce band plot
    #        eqpn2 = qps.Pickett_Interpolate(kvecs1, eks1, eqpn, kvecs2, eks2, fout)
    #        eqp21 = wint.OnlyRerunWannier(eqpn, klist2, eks2, kind2, wcore, fout)
    #        SaveBandPlot(ddir+'/GW0_bands_Pickett.dat', eqpn2, klist2, kqm.k2cartes, knames)
    #        SaveBandPlot(ddir+'/GW0_bands_Wannier.dat', eqp21, klist2, kqm.k2cartes, knames)
    

if __name__ == '__main__':
    
    band_energy_file = ''
    if len(sys.argv)>1 and os.path.isfile(sys.argv[1]):
        band_energy_file = sys.argv[1]
        if len(sys.argv)<3:
            print('When you give energy file for interpolation, you must also give as additional argument the Fermi energy for this bands')
            sys.exit(0)
        else:
            band_EFermi = float(sys.argv[2])*Ry2H

    io = InOut("gw.inp", "pypp.out", mrank==master)
    mode = 'all'
    if len(sys.argv)>3:
        if sys.argv[3]=='Pickett':
            mode = 'Pickett'
        elif sys.argv[3]=='Wannier':
            mode = 'Wannier'
        else:
            print('mode', sys.argv[3], 'not yet implemented. Should be [Pickett|Wannier]' )
            sys.exit(0)
    
    main(band_energy_file, band_EFermi, io, mode)
    
