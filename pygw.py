#!/usr/bin/env python3
# @Copyright 2020 Kristjan Haule
import os,sys
from scipy import optimize
from timeit import default_timer as timer
from datetime import datetime

import gwienfile as w2k
#import gaunt as gn          # gaunt coefficients
#import for_tetrahedra as ft #
#import radials as rd        # radial wave functions
#import fnc                  # fermi and gauss functions

from cmn import *
from cmn2 import Check_Equal_k_lists
from inout import InOut
from kqmesh import KQmesh
import mcommon as mcmn
from matel2band import MatrixElements2Band
from productbasis import ProductBasis
from kohnsham import KohnShamSystem
from planewaves import PlaneWaves
from frequencym import FrequencyMesh
from kweights import Kweights
from linhardt import Polarization_weights
from svdfunc import svd_functions
from numpy import *

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

Debug_Print = False
Save_eps = False

class G0W0:
    ddir = 'data'
    def __init__(self):
        if mrank==master and not os.path.isdir(self.ddir):
            os.mkdir(self.ddir)
            
    def PrintRadials(self, radf, strc, in1):
        isp=0
        lomaxp1 = shape(in1.nlo)[0]
        for iat in range(strc.nat):
            rx, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
            for l in range(in1.nt):
                frf = open(self.ddir+'/RadialFunctions.'+str(iat)+'.'+str(l), 'w')
                print('#   u   udot', end=' ', file=frf)
                if l < lomaxp1:
                    for ilo in in1.nLO_at_ind[iat][l]:
                        print('   ulo[ilo='+str(ilo)+']', end=' ', file=frf)
                print(file=frf)
                for ir,r in enumerate(rx):
                    print(r, radf.ul[isp,iat,l,ir], radf.udot[isp,iat,l,ir], end=' ', file=frf)
                    if l < lomaxp1:
                        for ilo in in1.nLO_at_ind[iat][l]:
                            print(radf.ulo[isp,iat,l,ilo,ir], end=' ', file=frf)
                    print(file=frf)
        
    def Compute(self, io, ps):
        (case, nspin, fout) = (io.case, io.nspin, io.out)
        
        #print >> fout, 'mem-usage[io]=', ps.memory_info().rss*b2MB,'MB'
        
        print('-'*32, "Set Struct", '-'*32, file=fout)
        fout.flush()
        strc = w2k.Struct(case, fout)
        strc.debugprint(fout)
        
        print('-'*32, "Lattice generation", '-'*32, file=fout)
        fout.flush()
        latgen = w2k.Latgen(strc, fout)
        
        
        print('-'*32, file=fout)
        fr = FrequencyMesh(io.iopfreq, io.nomeg, io.omegmin, io.omegmax, io.iopMultiple, fout)
        print('-'*32, file=fout)

        if io.iopfreq==5: # This means SVD is on
            om_min = io.omegmin if io.omegmin > 1e-3 else 1e-3
            om_max = io.omegmax if io.omegmax > 10 else 10
            om_nom = io.nomeg*3 if io.nomeg > 30 else 30*3
            (Ul, dUl) = svd_functions(fr.omega, fr.womeg, om_min, om_max, om_nom, io.svd_cutoff, fout)
            save(self.ddir+'/Ul', Ul)
            save(self.ddir+'/dUl', dUl)
        else:
            Ul, dUl = None, None
        
        #print >> fout, 'mem-usage[strc,latgen]=', ps.memory_info().rss*b2MB,'MB'
        
        print('-'*32, "w2k_readin1", '-'*32, file=fout)
        fout.flush()
        in1 = w2k.In1File(case, strc, fout, io.lomax)
        # call set_lapwlo         !* set up LAPW+lo basis
        (Elapw, Elo) = w2k.get_linearization_energies(case, in1, strc, nspin, fout)
        in1.Add_Linearization_Energy(Elapw, Elo)
         
        #print >> fout, 'mem-usage[in1]=', ps.memory_info().rss*b2MB,'MB'
        
        Vr = w2k.Read_Radial_Potential(case, strc.nat, nspin, strc.nrpt, fout)
        radf = w2k.RadialFunctions(in1,strc,Elapw,Elo,Vr,nspin,fout)

        if mrank==master:
            self.PrintRadials(radf, strc, in1)
        
        del Vr
        radf.get_ABC(in1, strc, fout)
        
        #print >> fout, 'mem-usage[Vr,radf]=', ps.memory_info().rss*b2MB,'MB'

        divs = io.nkdivs
        #divs = [15,15,15]
        kqm = KQmesh(divs, io.k0shift, strc, latgen, fout)
        kqm.tetra(latgen, strc, fout)
        
        #print >> fout, 'mem-usage[KQmesh]=', ps.memory_info().rss*b2MB,'MB'

        print('-'*32, 'w2k_readcore', file=fout)
        fout.flush()
        
        wcore = w2k.CoreStates(case, strc, nspin, fout)
        
        #print >> fout, 'mem-usage[wcore]=', ps.memory_info().rss*b2MB,'MB'
        
        print('-'*32, "set_mixbasis", '-'*32, file=fout)
        fout.flush()
        pb = ProductBasis( (io.kmr, io.pwm, io.lmbmax, io.wftol, io.lblmax, io.mb_emin, io.mb_emax), strc, in1, radf, wcore, nspin,  fout)
        pb.cmp_matrices(strc, in1, radf, wcore, nspin, fout)
        pb.generate_djmm(strc,latgen.trotij,in1.nt,fout)
        
        #print >> fout, 'mem-usage[ProductBasis]=', ps.memory_info().rss*b2MB,'MB'
        
        
        io_data={ 'emax_pol': io.emax_pol, 'emax_sc':  io.emax_sc, 'iop_core': io.iop_core, 'efermi': io.efermi,
                  'ibgw': io.ibgw, 'nbgw': io.nbgw, 'emingw': io.emingw, 'emaxgw': io.emaxgw,
                  'emin_tetra': io.emin_tetra, 'emax_tetra':io.emax_tetra}
        #ks = KohnShamSystem(io_data, kqm, case, in1, strc, wcore, radf, nspin, fout)
        #ks = KohnShamSystem(io_data, kqm, case, strc, wcore, nspin, fout)
        ks = KohnShamSystem(case, strc, nspin, fout)
        ks.Set_Fermi_and_band_params(io_data, kqm, wcore, fout)
        ks.Set_nv(in1.nlo_tot)
        ks.ComputeRadialIntegrals(case, in1, strc, wcore, radf, nspin, fout)
        
        #print >> fout, 'mem-usage[KohnSham]=', ps.memory_info().rss*b2MB,'MB'
        
        Check_Equal_k_lists(ks.klist, kqm, fout)
        
        pw = PlaneWaves(ks.hsrws, io.kmr, io.pwm, case, strc, in1, latgen, kqm, False, fout)
        
        #print >> fout, 'mem-usage[PlaneWaves]=', ps.memory_info().rss*b2MB,'MB'
        
        #  determine minimum rmt
        self.rmtmin = min(strc.rmt)
        print('minimum Rmt=', self.rmtmin, file=fout)
        print('volume fraction of each MT-sphere is', pw.vmt, file=fout)
        fout.flush()
        
        ks.VectorFileRead(case, strc, latgen, kqm, pw, fout)
        
        del ks.klist
        
        ks.Vxc(case, in1, strc, radf, fout)
        
        #print >> fout, 'mem-usage[ks.VectorFileRead,ks.Vxc]=', ps.memory_info().rss*b2MB,'MB'
        #print >> fout, 'mem-usage[FrequencyMesh]=', ps.memory_info().rss*b2MB,'MB'
        
        kw = Kweights(io, ks, kqm, fout)
        
        #print >> fout, 'mem-usage[Kweights]=', ps.memory_info().rss*b2MB,'MB'
        
        me = MatrixElements2Band(io, pw, strc, latgen)
        Vxct = me.Vxc(strc, in1, latgen, kqm, ks, radf, pw, pb, fout)
        if mrank==master:
            save(self.ddir+'/Vxct', Vxct)
        
        fout.flush()
        
        del ks.uxcu
        del ks.Vxcs
        del ks.ksxc
        del ks.lxcm_f
        del ks.lmxc_f

        #print >> fout, 'mem-usage[MatrixElements2Bands,Vxct]=', ps.memory_info().rss*b2MB,'MB'

        if io.MatrixSelfEnergy:
            sigx = zeros( (len(kqm.weight),ks.nbgw-ks.ibgw,ks.nbgw-ks.ibgw) )
            sigc = zeros( (len(kqm.weight),ks.nbgw-ks.ibgw,ks.nbgw-ks.ibgw,len(fr.omega)), dtype=complex )
        else:
            sigx = zeros( (len(kqm.weight),ks.nbgw-ks.ibgw) )
            sigc = zeros( (len(kqm.weight),ks.nbgw-ks.ibgw,len(fr.omega)), dtype=complex )
            
        t_Coul, t_wgh, t_selfx, t_setev, t_calc_eps, t_calc_sfc = 0, 0, 0, 0, 0, 0

        #print >> fout, 'mem-usage[sigx,sigc]=', ps.memory_info().rss*b2MB,'MB'

        iqs,iqe,sendcounts,displacements = mcmn.mpiSplitArray(mrank, msize, len(kqm.qlist) )
        print('processor rank=', mrank, 'will do', list(range(iqs,iqe)), file=fout)
        fout.flush()
        #iqs,iqe = 0, len(kqm.qlist)
        #iqs,iqe = len(kqm.qlist)-1,len(kqm.qlist)
        PartialTetra=True
        for iq in range(iqs,iqe):
            t1 = timer()
            # Calculates matrix elements of the Coulomb interaction
            me.Coulomb(iq, io, strc, in1, latgen, kqm, ks, radf, pw, pb, fout)
            t2 = timer()
            #print >> fout, 'mem-usage[Coulomb(iq='+str(iq)+']=', ps.memory_info().rss*b2MB,'MB'
            #me.Coulomb_from_PW(iq, io, strc, in1, latgen, kqm, ks, radf, pw, pb, fout)
            
            # removes eigenvalues of the Coulomb interaction which are very small

            me.Coul_setev(iq, fout, io.iop_coul_x, 1e-7)
            t3 = timer()

            # calculates the exchange self-energy
            sigx = me.calc_selfx(sigx, iq, strc, in1, latgen, kqm, ks, radf, pw, pb, wcore, kw, io, fout)
            t4 = timer()
            
            # removes more eigenvalues of the Coulomb interaction, which are smaller than io.barcevtol
            me.Coul_setev(iq, fout, io.iop_coul_x, io.barcevtol)
            t5 = timer()

            # Using tetrahedron method, computes polarization in the band basis (Lindhardt formula), but is integrated over momentum
            print('***** iq=', iq, file=fout)
            kcw = Polarization_weights(iq, ks, kqm, wcore, fr, io.iop_bzintq, io.fflg, dUl, fout, PartialTetra=PartialTetra)

            if False:
                kcw1 =Polarization_weights(iq, ks, kqm, wcore, fr, io.iop_bzintq, io.fflg, dUl, fout, PartialTetra=False)
                #kcw1(No,Ne, nom, nkp) 
                nkall = len(kqm.kii_ind)
                ncg = ks.ncg_p 
                nvbm, ncbm = ks.nomax_numin[0]+1, ks.nomax_numin[1]
                No = ncg+nvbm          # occupied
                Ne = ks.nbmaxpol-ncbm  # empty
                isp=0
                #Noactual = ncg+ks.ibmin_tetra+1
                #Neactual = ks.nbmaxpol-ks.ibmax_tetra
                print('No=', No, 'Ne=', Ne)
                #
                for ik,irk in enumerate(kqm.kii_ind):
                    jk = kqm.kqid[ik,iq]
                    jrk = kqm.kii_ind[jk]
                
                    enk = zeros(No)
                    for icg in range(ncg):
                        iat, idf, ic = wcore.corind[icg][0:3]
                        enk[icg] = wcore.eig_core[isp][iat][ic]
                    enk[ncg:No] = ks.Ebnd[isp,irk,:(No-ncg)]
                    dwo = hstack( (ones(ncg), nkall*kw.kiw[irk,:(No-ncg)]) )
                    
                    enq = ks.Ebnd[isp,jrk,ncbm:ks.nbmaxpol]  # energy at k-q
                    dwe = 1-nkall*kw.kiw[jrk,ncbm:ks.nbmaxpol]
                    
                    omega = fr.omega[0]
                    for io in range(No):
                        for ie in range(Ne):
                            kcw1_ = kcw1[io,ie,0,ik]
                            dE = enq[ie] - enk[io]
                            dw = dwe[ie]*dwo[io]  # like f(-ie)*f(io)
                            kcw2_ = -2*dw/nkall*dE/( omega**2 + dE**2)
                            if kcw1_!=0:
                                ratio = kcw2_/kcw1_
                            else:
                                ratio = 0
                                if kcw2_ == 0:
                                    ratio = 1
                            print('%2d %2d  %3d %3d %16.13f %16.13f ratio=%16.13f dwe=%10.5f dwo=%10.5f dw=%10.5f' % (ik,jk,io,ie, kcw1_, kcw2_, ratio, dwe[ie],dwo[io],dw))
                sys.exit(0)
            
            t6 = timer()
            #print >> fout, 'mem-usage[sigx,Polarization_weights(iq='+str(iq)+']=', ps.memory_info().rss*b2MB,'MB'
            
            if iq==0:  # expansion around q=0 of V*P
                # Note : head_quantities = (head, mmatcv, mmatvv, mst)
                head_quantities = me.calc_head(strc, in1, latgen, kqm, ks, radf, pw, pb, wcore, kw, fr, kcw, io.iop_drude, io.eta_head, dUl, Ul, fout, PartialTetra)
            else:
                head_quantities = None
            
            # Calculates V*(1/epsilon-1) = W-V
            (eps, epsw1, epsw2, head) = me.calc_eps(iq, head_quantities, strc, in1, latgen, kqm, ks, radf, pw, pb, wcore, kw, fr, kcw, self.ddir, dUl, Ul, fout, PartialTetra)
            del kcw
            
            if Save_eps:
                # eps(matsize,matsize,nom_nil)
                save(self.ddir+'/epsw.'+str(iq), eps)     # 1/(1+V*P)-1 = W/V-1
                if iq==0:
                    # epsw1(matsize,nom_nil)
                    # epsw2(matsize,nom_nil)
                    # head(nom_nil)
                    save(self.ddir+'/epsw1', epsw1)
                    save(self.ddir+'/epsw2', epsw2)
                    save(self.ddir+'/head', head)
                    
            t7 = timer()
            #print >> fout, 'mem-usage[eps(iq='+str(iq)+')]=', ps.memory_info().rss*b2MB,'MB'
            # Calculates correlation self-energy
            
            #print('save_mwm in pygw.py=', io.save_mwm)
            sigc = me.calc_selfc(sigc, iq, eps, epsw1, epsw2, head, strc, in1, latgen, kqm, ks, radf, pw, pb, wcore, kw, fr, self.ddir, dUl, Ul, fout, io)#save_mwm=io.save_mwm)
            t8 = timer()
            #print >> fout, 'mem-usage[sigc(iq='+str(iq)+')]=', ps.memory_info().rss*b2MB,'MB'
            fout.flush()
            
            t_Coul += t2-t1
            t_setev += t3-t2+t5-t4
            t_wgh += t6-t5
            t_selfx += t4-t3
            t_calc_eps += t7-t6
            t_calc_sfc += t8-t7
            
        print('q-loop finished', file=fout)
        fout.flush()
        
        if Parallel:
            sigx = comm.reduce(sigx, op=MPI.SUM, root=master)
            sigc = comm.reduce(sigc, op=MPI.SUM, root=master)
            
        if mrank==master:
            save(self.ddir+'/Sigmax', sigx)
            save(self.ddir+'/Sigmac', sigc)
            save(self.ddir+'/omega', fr.omega)
            save(self.ddir+'/womeg', fr.womeg)
            if fr.iopfreq == 4:
                save(self.ddir+'/omega_precise', fr.omega_precise)
                save(self.ddir+'/womeg_precise', fr.womeg_precise)
            
            #print >> fout, 'mem-usage[after q-look]=', ps.memory_info().rss*b2MB,'MB'
            
            if io.MatrixSelfEnergy:
                for irk in range(len(kqm.weight)):
                    for ie1 in range(ks.ibgw, ks.nbgw):
                        sg = sigx[irk,ie1-ks.ibgw,ie1-ks.ibgw]
                        print(' Sigx[irk=%3d,ie1=%3d,ie3=%3d]=%16.12f' % (irk, ie1, ie1, sg), file=fout)
                for irk in range(len(kqm.weight)):
                    for ie1 in range(ks.ibgw, ks.nbgw):
                        for ie3 in range(ie1+1, ks.nbgw):
                            i,j = ie1-ks.ibgw,ie3-ks.ibgw
                            ratio = (abs(sigx[irk,i,j])+abs(sigx[irk,j,i]))/(abs(sigx[irk,i,i])+abs(sigx[irk,j,j]))
                            if ratio>io.sigma_off_ratio:
                                print(' Sigx[irk=%3d,ie1=%3d,ie3=%3d]=%16.12f' % (irk, ie1, ie3, sigx[irk,i,j]), file=fout)
                for irk in range(len(kqm.weight)):
                    for ie1 in range(ks.ibgw, ks.nbgw):
                        iom=0
                        sg = sigc[irk,ie1-ks.ibgw,ie1-ks.ibgw,iom]
                        print(' Sigc[irk=%3d,ie1=%3d,ie3=%3d,iom=%3d]=%16.12f%16.12f' % (irk, ie1, ie1, iom, sg.real, sg.imag), file=fout)
                twhich_indices=[]
                for irk in range(len(kqm.weight)):
                    which_indices=[(i1,i1) for i1 in range(0,ks.nbgw-ks.ibgw)]
                    for ie1 in range(ks.ibgw, ks.nbgw):
                        for ie3 in range(ie1+1, ks.nbgw):
                            i,j = ie1-ks.ibgw, ie3-ks.ibgw
                            ratio = sum(abs(sigc[irk,i,j,:])+abs(sigc[irk,j,i,:]))/sum(abs(sigc[irk,i,i,:])+abs(sigc[irk,j,j,:]))
                            iom=0
                            if ratio> 0.1*io.sigma_off_ratio:
                                which_indices.append([i,j])
                                which_indices.append([j,i])
                                sg = sigc[irk,i,j,iom]
                                print(' Sigc[irk=%3d,ie1=%3d,ie3=%3d,iom=%3d]=%16.12f%16.12f' % (irk, ie1, ie3, iom, sg.real, sg.imag), file=fout)
                    twhich_indices.append(which_indices)
                    
            else:
                twhich_indices=[]
                for irk in range(len(kqm.weight)):
                    for ie1 in range(ks.ibgw, ks.nbgw):
                        sg = sigx[irk,ie1-ks.ibgw]
                        print(' Sigx[irk=%3d,ie=%3d]=%16.12f' % (irk, ie1, sg), file=fout)
                for irk in range(len(kqm.weight)):
                    for ie1 in range(ks.ibgw, ks.nbgw):
                        for iom in range(len(fr.omega)):
                            sg = sigc[irk,ie1-ks.ibgw,iom]
                            print(' Sigc[irk=%3d,ie=%3d,iom=%3d]=%16.12f%16.12f' % (irk, ie1, iom, sg.real, sg.imag), file=fout)
            
            
                            
            print('## Coulomb:     t(Coulomb)         =%14.9f' % (t_Coul,), file=fout)
            print('## Coulomb:     t(setev)           =%14.9f' % (t_setev,), file=fout)
            print('## calc_selfx:  t(selfx)           =%14.9f' % (t_selfx,), file=fout)
            print('## eps weights: t(kcw)             =%14.9f' % (t_wgh,), file=fout)
            print('## eps calc_eps:t(calc_eps)        =%14.9f' % (t_calc_eps,), file=fout)
            print('## calc_selfc:  t(selfc)           =%14.9f' % (t_calc_sfc,), file=fout)
            
            nbnd, nom = ks.nbgw-ks.ibgw, len(fr.omega)
            print("AnalizeSpectra : calceqp", file=fout)
            if io.iop_es >=0 :
                print("# Parameters used:", file=fout)
                print("#  Analytic continuation (iop_ac) =", io.iop_ac, file=fout)
                print("#  Fermi level shift    (iop_es)  =", io.iop_es, file=fout)
                print("#  Nr.freq points  (nomeg)        =", nom, file=fout)
                print("#  Number of AC poles (npar_ac/2) =", io.npar_ac/2, file=fout)

            EF_qp = 0
            isp=0
            bands = copy(ks.Ebnd[isp])
            save(self.ddir+'/KS_qp', bands[:,ks.ibgw:ks.nbgw]) 
            # quasiparticle energies for G0W0 scheme
            (eqp, eqp_im) = mcmn.Compute_quasiparticles(bands[:,ks.ibgw:ks.nbgw], bands[:,ks.ibgw:ks.nbgw], sigc, sigx, Vxct[:,ks.ibgw:ks.nbgw,ks.ibgw:ks.nbgw], fr.omega, io, isp, fout, PRINT=True)
            # the fermi energy for G0W0 scheme
            (EF, Eg, evbm, ecbm, eDos) = mcmn.calc_Fermi(eqp, kqm.atet, kqm.wtet, wcore.nval-ks.ibgw*2, io.nspin)
            print(':E_FERMI_QP(eV)=  %12.4f' % (EF*H2eV,), file=fout)
            if Eg > 0:
                print(':BandGap_QP(eV)=  %12.4f' % (Eg*H2eV,), file=fout)
            else:
                print(':DOS_at_Fermi_QP= %12.4f' % (eDos,), file=fout)
            print('Fermi: evbm=%12.4f  ecbm=%12.4f ' % (evbm*H2eV, ecbm*H2eV), file=fout)
            save(self.ddir+'/GW_qp', eqp-EF)
            
            # First analyzing Kohn-Sham bands
            (nomax,numin) = mcmn.Band_Analys(bands[:,ks.ibgw:ks.nbgw], ks.EF, nbnd, 'KS', kqm, fout)
            # Next analyzing G0W0 bands
            (nomax,numin) = mcmn.Band_Analys(eqp, EF, nbnd, 'GW', kqm, fout)
            print('Done ',datetime.now(), file=fout)

if __name__ == '__main__':
    #pid = os.getpid()
    #ps = psutil.Process(pid)
    io = InOut("gw.inp", "pygw.out", mrank==master)
    #io = InOut("gw.inp", 'pygw.'+str(mrank), True)
    wgw = G0W0()
    #wgw.Compute(io, ps)
    wgw.Compute(io, None)
    if mrank==master:
        print('PYGW DONE')

