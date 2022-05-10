from numpy import *
from functools import reduce


import radials as rd        # radial wave functions
import radd                 # radial derivative of a function

from cmn import Ry2H, H2eV
import gwienfile as w2k
import mcommon as mcmn

Debug_Print = False

def Convert2CubicPotential(Vlm, lmxc, strc):
    c_kub = zeros((11,11))
    c_kub[ 0, 0] = 1.0
    c_kub[ 3, 2] = 1.0
    c_kub[ 4, 0] = 0.5*sqrt(7./3.)
    c_kub[ 4, 4] = 0.5*sqrt(5./3.)
    c_kub[ 6, 0] = 0.5*sqrt(0.50)
    c_kub[ 6, 2] = 0.25*sqrt(11.0)
    c_kub[ 6, 4] =-0.5*sqrt(7.0/2.0)
    c_kub[ 6, 6] =-0.25*sqrt(5.0)
    c_kub[ 7, 2] = 0.5*sqrt(13./6.)
    c_kub[ 7, 6] = 0.5*sqrt(11./16.)
    c_kub[ 8, 0] = 0.125*sqrt(33.)
    c_kub[ 8, 4] = 0.25*sqrt(7./3.)
    c_kub[ 8, 8] = 0.125*sqrt(65./3.)
    c_kub[ 9, 2] = 0.25*sqrt(3.)
    c_kub[ 9, 4] = 0.5*sqrt(17./6.)
    c_kub[ 9, 6] =-0.25*sqrt(13.)
    c_kub[ 9, 8] =-0.5*sqrt(7./6.)
    c_kub[10, 0] = 0.125*sqrt(65./6.)
    c_kub[10, 2] = 0.125*sqrt(247./6.)
    c_kub[10, 4] =-0.25*sqrt(11./2.)
    c_kub[10, 6] = 0.0625*sqrt(19./3.)
    c_kub[10, 8] =-0.125*sqrt(187./6.)
    c_kub[10,10] =-0.0625*sqrt(85.)
    sq2 = sqrt(2.0)
    for iat in range(strc.nat):
        #print 'lmxc=', lmxc[iat]
        if strc.iatnr[iat]>0:  # this means this atom has cubic environment, and we will transform potential to cubic
            lxc = 0
            while lxc < len(lmxc[iat]):
                lx,mx = lmxc[iat][lxc]
                #print 'iat=', iat, 'lxc=', lxc, 'l=', lx, 'm=', mx
                if lx==0:
                    if mx == 0:
                        lxc += 1
                    else:
                        break
                elif lx==-3:
                    if mx == 2:
                        Vlm[iat][lxc,:] *= -1.0/sq2
                        lxc += 1
                    else:
                        break
                elif lx in (4,6,-7,-9):
                    la = abs(lx)
                    c1, c2 = c_kub[la,mx], c_kub[la,mx+4]
                    if mx == 0:
                        sq1 = 1.0
                    else:
                        sq1 = sq2
                    tmp  =  Vlm[iat][lxc,:] * c1 + Vlm[iat][lxc+1,:] * c2
                    Vlm[iat][lxc  ,:] =  tmp * (c1/sq1)
                    Vlm[iat][lxc+1,:] =  tmp * (c2/sq2)

                    #print 'c1=', c1, 'c2=', c2, 'l=', lx, 'm=', mx, 'V[0]=', Vlm[iat][lxc,-1], Vlm[iat][lxc+1,-1]
                    lxc=lxc+2
                elif lx in (8,10):
                    if mx == 0:
                        sq1 = 1.0
                    else:
                        sq1 = sq2
                    la = abs(lx)
                    c1, c2, c3 = c_kub[la,mx], c_kub[la,mx+4], c_kub[la,mx+8]
                    tmp = Vlm[iat][lxc,:] * c1 + Vlm[iat][lxc+1,:] * c2 + Vlm[iat][lxc+2,:] * c3
                    Vlm[iat][lxc  ,:] = tmp * c1/sq1
                    Vlm[iat][lxc+1,:] = tmp * c2/sq2
                    Vlm[iat][lxc+2,:] = tmp * c3/sq2
                    lxc += 3
                else:
                    break
            
                    
# instead of:
#     ks = KohnShamSystem(io_data, kqm, io.case, strc, wcore, io.nspin, fout)
# we should use
#     ks = KohnShamSystem(io.case, strc, io.nspin, fout)
#     ks.Set_Fermi_and_band_params(io_data, kqm, wore, fout)
#     ks.Set_nv(in1.nlo_tot)


class KohnShamSystem:
    """ This class set up Kohn-Sham reference system, including core
        stat`<es, KS energies and orbitals, KS exchange-correlation potential 
    """
    PRINT = False
    #def __init__(self, io, kqm, case, strc, core, nspin, fout):
    def __init__(self, case, strc, nspin, fout):
        self.nspin = nspin
        # Reading w2k energy files and its KS-eigenvalues
        spflag = ['up','dn'] if self.nspin==2 else ['']
        (self.klist, wegh, Ebnd, self.hsrws) = w2k.Read_energy_file(case+'.energy'+spflag[0], strc, fout)
        band_max = min(list(map(len,Ebnd)))
        self.Ebnd = zeros( (self.nspin,len(Ebnd),band_max) )
        for ik in range(len(Ebnd)):
            self.Ebnd[0,ik,:] = Ebnd[ik][:band_max]
        if self.nspin==2:
            (self.klist, wegh, Ebnd, self.hsrws) = w2k.Read_energy_file(case+'.energy'+spflag[1], strc, fout)
            for ik in range(len(Ebnd)):
                self.Ebnd[1,ik,:] = Ebnd[ik][:band_max]
        # converting to Hartrees
        self.Ebnd *= Ry2H  # convert bands to Hartree
        self.EFermi = 0    # For now set to zero, but should be recomputed in Set_Fermi_and_band_params
        
    def Set_Fermi_and_band_params(self, io, kqm, core, fout):
        # Setting up several cutoff-parameters, which depend on KS-energies
        self.set_band_par(io['emax_pol'], io['emax_sc'], io['iop_core'], core, self.nspin, fout)

        # Recompute Fermi energy if neeeded
        if io['efermi'] >= 1e-2: # Recompute the Fermi energy
            (EF, Eg, evbm, ecbm, eDos) = mcmn.calc_Fermi(self.Ebnd[0], kqm.atet, kqm.wtet, core.nval, self.nspin)
            
        else:
            print(' Use the Fermi energy from case.ingw', file=fout)
            evbm = max( [x for x in self.Ebnd.ravel() if x < EF] )
            ecbm = min( [x for x in self.Ebnd.ravel() if x > EF] )
            Eg = ecbm - evbm
            eDos = sum([ft.dostet(EF, self.Ebnd, kqm.atet, kqm.wtet) for isp in range(self.nspin)])*2.0/self.nspin
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
        #EF = 0.0
        self.EF = 0.0
        self.Eg = Eg
        print('Set EF to ', self.EF, file=fout)
        
        if False:
            print('KS: bande', file=fout)
            for ik in range(shape(self.Ebnd)[1]):
                for ie in range(shape(self.Ebnd)[2]):
                    print('%3d %3d %18.12f' % (ik,ie,self.Ebnd[0,ik,ie]), file=fout)
        
        
        # Setting some extra cutoff-parameters
        self.set_band_par2(io, core, self.nspin, fout)

    def Set_nv(self, in1_nlo_tot):
        # number of basis functions excluding local orbitals
        self.nv = array(self.hsrws)-in1_nlo_tot
    
    def VectorFileRead(self, case, strc, latgen, kqm, pw, fout, in1=None):
        def Shift_to_1BZ(k):
            " This gives k_{1BZ}-k"
            if k==0:
                return 0
            elif k>0:
                return -int(floor(k))
            elif k<0:
                return int(ceil(-k))
            
        nsp, nkp, band_max = shape(self.Ebnd)
        if in1 is not None:
            lomaxp1, nloat = shape(in1.nlo)[0], amax(in1.nlo)

            # WARNING: I think we should have here loop over spin, because we are currently reading only one spin vector file
            (heads, all_Gs, self.all_As, all_Ek, self.Elapw, self.Elo) = w2k.Read_vector_file(case, strc.nat, fout, in1.nt, lomaxp1, nloat)
            for iat in range(strc.nat):
                for l in range(in1.nt):
                    if not in1.lapw[l,iat]:
                        self.Elapw[iat,l] -= 200
                #print(shape(Elo), shape(Elapw))
                for l in range(lomaxp1):
                    if in1.lapw[l,iat]: # This is LAPW+LO, hence local orbitals start at ilo=1 and not at ilo=0
                        #Elo[iat,l,:] = roll(Elo[iat,l,:],1)
                        #print('iat=', iat, 'l=', l, '1:'+str(nloat)+'->0:'+str(nloat-1))
                        self.Elo[iat,l,1:nloat] = self.Elo[iat,l,0:(nloat-1)]
            # WARNING: because we currently do not read two vector files, this should not work for magnetic case!
            if nsp==1:
                (nat,lmaxp1) = shape(self.Elapw)
                (nat,lomaxp1,nloatp1) = shape(self.Elo)
                self.Elapw = reshape(self.Elapw, (1,nat,lmaxp1))
                self.Elo   = reshape(self.Elo,   (1,nat,lomaxp1,nloatp1))
            else:
                print('ERROR in kohnsham.py:VectorFileRead: Here we need to read both vector files for up and dn!')
                sys.exit(0)
            print('Eapw from vector file:',file=fout)
            for iat in range(strc.nat):
                for l in range(in1.nt):
                    print('Eapw[iat='+str(iat)+',l='+str(l)+']='+str(self.Elapw[0,iat,l]), file=fout)
            print('Elo from vector file:', file=fout)
            for iat in range(strc.nat):
                for l in range(lomaxp1):
                    for ilo in range(nloat):
                        if self.Elo[0,iat,l,ilo]  < 997.0:
                            print('Elo[iat='+str(iat)+',l='+str(l)+',ilo='+str(ilo)+']='+str(self.Elo[0,iat,l,ilo]), file=fout)
        else:
            (heads, all_Gs, self.all_As, all_Ek) = w2k.Read_vector_file(case, strc.nat, fout)
            self.Elapw, self.Elo=None,None
            
        #print('Elinear=', Elinear)
        self.indgkir = []
        if self.PRINT:
            print('indgkir: only some G-vectors generated above are used in the vector file. indgkir is index to this subset of G-vectors', file=fout)


        for irk in range(len(all_Ek)):
            diffE = sum(abs(all_Ek[irk][:band_max]*Ry2H-self.EFermi - self.Ebnd[0,irk,:]))
            k, kname, wgh, ios, n0, nb = heads[irk]
            diffk = sum(abs(k-self.klist[irk]))
            if diffk > 1e-3:
                print('WARNING it seems k-points from vector file and energy file are not compatible. diffk=', diffk, file=fout)
            if diffE > 1e-3:
                print('WARNING it seems band energies from vector file and energy file are not compatible. diffE=', diffE, file=fout)
            nG,n3 = shape(all_Gs[irk])
            print('k-point from vector file=', k, 'klist[irk]=', self.klist[irk], '#Gs at that k-point=', nG, file=fout)
            ind = zeros(nG,dtype=int)
            for i in range(nG):
                iG = all_Gs[irk][i]          # reciprocal vector read from vector file
                ind[i] = pw.ig0[tuple(iG)]   # what is the index of this G in my gvec (previously generated fixed reciprocal mesh)?
                if self.PRINT:
                    print('irk=%3d i=%3d indgkir=%3d  G=(%3d %3d %3d)' % (irk+1, i+1, ind[i]+1, iG[0],iG[1],iG[2]), file=fout)
            self.indgkir.append(ind)
            # zzk == all_As[irk][ib,hsrows]
            # zzkall(:,:,irk,isp)=zzk
        
        # In the next several lines, we will find permutation of the reciprocal vectors needed to transform
        # eigenvectors from an irreducible k-point mesh to generic all-k point mesh
        #Gbas = array(round_(kqm.gbas * kqm.aaa), dtype=int) # This transforms from lattice to cartesian coordinate system
        #Gbas = kqm.k2icartes   # This transforms from lattice to semi-cartesian coordinate system (or does not transform when ortho=False)
        
        ankp = kqm.ndiv[0]*kqm.ndiv[1]*kqm.ndiv[2]          # all k-points, reducible as well
        #print >> fout, 'Gbas=', Gbas
        if True: #self.PRINT:
            print('Umklap shifts for all k-points. (If any component is not 0 or 1, gap2 code would give different result)', file=fout)
            for ik in range(ankp):
                irk = kqm.kii_ind[ik]
                #write(6,'(I4,1x,I4,1x,A,3I3,A,2x,A,3I3,A)') ikp, kpirind(ikp), '(', ikvec(:), ')', '(', irkvec(:), ')'
                #print >> fout, '%4d %4d' % (ik+1, irk+1), tuple(kqm.klist[ik,:]), tuple(kqm.kirlist0[irk,:])
                if kqm.k_ind[irk] != ik : # this k-point is rreducible
                    isym = kqm.iksym[ik]       # which group operation was used to get this momentum vector
                    jkvec = dot(kqm.kirlist0[irk], latgen.tizmat[isym])
                    #g0 = map(lambda x: int( (1-sign(x))/2. ), jkvec)        # is equal to 1 if jkvec is negative
                    g0 = [Shift_to_1BZ(x/kqm.LCM) for x in jkvec]
                    g0 = dot(kqm.k2icartes, g0)
                    #if latgen.ortho or strc.lattic[:3]=='CXZ':
                    #    g0 = dot(Gbas, g0)
                    print('%3d' % (ik+1,), '%3d'*3 % tuple(g0), file=fout)

        if self.PRINT:
            print('vector_G', file=fout)
        self.phase_arg = []
        self.indgk = []   # permutation index for all k-points
        for ik in range(ankp): # over all k-points, i.e, k_generic
            irk = kqm.kii_ind[ik]
            if kqm.k_ind[irk] == ik : # this k-point is irreducible, hence the order of reciprocal vectors stays the same
                self.indgk.append( self.indgkir[irk] )
                self.phase_arg.append( 0.0 )
            else:
                Gs = all_Gs[irk]
                isym = kqm.iksym[ik]       # which group operation was used to get this momentum vector
                # Notice that k_generic*timat = k_irr + delta_G*timat   or   k_generic = k_irr*timat + delta_G
                # where k_generic and k_irr are in the 1-BZ, while k_irr*timat is generically not,
                #    and needs delta_G, which is called g0 here.
                # Note that g0 is here computed in lattice coordinates when ortho=True,
                #    and needs to be converted to cartesian below.
                
                ## timat*(k+G) == k_irr + G_irr
                ## tizmat*(k0+G0) == k_irr0 + G_irr0
                jkvec = dot(kqm.kirlist0[irk], latgen.tizmat[isym])  # generick k-point, but possibly outside 1BZ. Note that this is done in integer (lattice) coordinates, not semi-cartesians.
                #g0 = map(lambda x: int( (1-sign(x))/2. ), jkvec)    # BUG WAS HERE: This shift delta_G to bring k-point into 1BZ
                g0 = [Shift_to_1BZ(x/kqm.LCM) for x in jkvec]   # This shift delta_G to bring k-point into 1BZ, i.e., k_{1BZ}-k
                g0 = dot(kqm.k2icartes, g0)                          # now transforming to semi-cartesian coordinates (from lattice integer coordinates)
                # This final g0 should be like g0=dot(strc.timat[isym,:,:].T,kqm.kirlist[irk])
                tsymat = strc.timat[isym]  # symmetry operation that transforms k_generic[ik] = k_irr[irk]*timat + delta_G
                # Notice that this transforms the vector directly to cartesian coordinate system, not lattice system, like latgen.tizmat[isym]

                # WARNING,dmft1: To be compatible with DMFT1, we would need to have phase G[irr]+kmq.kirlist[irk]/float(kqm.LCM)
                # because dmft1 applies symmetry operation on the irreducible (k+G).
                #G_p_k = all_Gs[irk][:,:] + kqm.klist[ik,:]/float(kqm.LCM) # k+G in semi-cartesian
                # WARNING : Change in April 2022: we replaced here klist[ik] with kirlist[irk], because only than this is compatible with DMFT1
                G_p_k = all_Gs[irk][:,:] + kqm.kirlist[irk,:]/float(kqm.LCM) # k+G in semi-cartesian
                self.phase_arg.append( dot(G_p_k, strc.tau[isym,:]) )     # positions are in the same system, i.e., semi-cartesian : (k+G)*tau[isym]
                #phase = exp(-2*pi*self.phase_arg[-1]*1j)
                
                ind = zeros(shape(Gs)[0],dtype=int)
                # We have <k+G| H |k+G> given in terms of G's in indgkir order (as read from vector file).
                # We know that  G + k_irr = G + (k_generic-delta_G)*timat. If we change the order of G's, such that
                # G_new*timat = G, than we have  (G_new*timat + (k_generic-delta_G)*timat)*r = (G_new+k_generic-delta_G)*timat*r
                # and because timat is symmetry operation timat*r should be equivalent to r.
                # Now we see that G_new-delta_G will work for generik k-vector in the same way as G works for 
                # irreducible k_irr. Notice that timat^2=1 hence G_new = G*timat, and we need G*timat-delta_G
                iG_transformed = dot(all_Gs[irk], tsymat)   # all G-vectors in cartesian corrdinate systems are transformed at once here. This is equivalent to: fortran_timat[isym].G
                iG_transformed -= g0                        # We need G_new - delta_G, possible umklap shift
                for i in range(shape(Gs)[0]):               # over all reciprocal vectors in the plane wave basis
                    iG = iG_transformed[i]                  # for generic k-point, this is the correct set of G's, i.e, fortran_timat[isym].(k+G) == G_{transformed}+k_1BZ
                    ind[i] = pw.ig0[tuple(iG)]              # this gives index of each G-vector, and hence gives the necessary permutation of the eigenvector
                    if self.PRINT:
                        print('%s%3d '*6 % ('ik=', ik+1, 'irk=', irk+1, 'isym=', isym+1, 'indgkir=', self.indgkir[irk][i]+1, 'ig=', i+1, 'indgk=', ind[i]+1), '%s%12.7f' %('arg=', args[i]), file=fout)
                self.indgk.append(ind)                      # permutation of the eigenvectors, which should work for generic k-point
                #print 'rred ik=', ik, 'with irred ik=', irk, 'and shift g0=', g0, 'jkvec=', jkvec, 'isym=', isym, 'and timat=', (tsymat.T).tolist()

        #print 'indgk=', len(self.indgk), map(shape,self.indgk)
        print('-'*32, '\n', file=fout)
        
    def ComputeRadialIntegrals(self, case, in1, strc, core, radf, nspin, fout):
        # Calculating many radial integrals needed for evaluating the Coulomb repulsion
        self.core_valence_integrals(case, in1, strc, core, radf, nspin, fout)
        self.valence_valence_integrals(case, in1, strc, core, radf, nspin, fout)
        print('hsrws=', self.hsrws, file=fout)
        print('nv=', list(self.nv), file=fout)
        print('\n'+'-'*32, file=fout)


    def Vxc(self, case, in1, strc, radf, fout):
        (lmxc, Vxclm, self.ksxc, self.Vxcs) = w2k.Read_xc_file(case, strc, fout)
        for iat in range(strc.nat): Vxclm[iat] *= Ry2H
        self.Vxcs  *= Ry2H
        Convert2CubicPotential(Vxclm, lmxc, strc)
        
        maxlxc = max([len(lmxc[iat]) for iat in range(strc.nat)])
        # create fortran equivalent of lmxc
        self.lxcm_f = zeros(strc.nat, dtype=int)
        self.lmxc_f = zeros((2,maxlxc,strc.nat), dtype=int, order='F')  
        for iat in range(strc.nat):
            self.lxcm_f[iat] = len(lmxc[iat])    # bug jul.7 2020
            for lxc in range(len(lmxc[iat])):
                self.lmxc_f[:,lxc,iat] = lmxc[iat][lxc][:]
        
        # number of functions at each (iat,l)
        nrf = [[ 2+len(in1.nLO_at_ind[iat][l]) if l < len(in1.nLO_at_ind[iat]) else 2 for l in range(in1.nt)] for iat in range(strc.nat)]   
        nrmax = max(list(map(max,nrf)))
        self.uxcu = zeros( (strc.nat, maxlxc, in1.nt**2, nrmax**2) )
        #print 'shape(self.uxcu)=', shape(self.uxcu)
        
        isp=0
        # computing matrix elements <l2,m2| V_{bl,m} |l1,m1>
        for iat in range(strc.nat):
            #npt = strc.nrpt[iat]
            #dh  = log(strc.rmt[iat]/strc.r0[iat])/(npt - 1)      # logarithmic step for the radial mesh
            #dd = exp(dh)
            #rx = strc.r0[iat]*dd**range(npt)
            rx, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
            
            ars = [[] for l in range(in1.nt)]
            for l in range(len(in1.nLO_at_ind[iat])):
                ars[l] = [radf.ul[isp,iat,l,:]/rx, radf.udot [isp,iat,l,:]/rx]
                ars[l] += [radf.ulo[isp,iat,l,ilo,:]/rx for ilo in in1.nLO_at_ind[iat][l]]
            for l in range(len(in1.nLO_at_ind[iat]),in1.nt):
                ars[l] = [radf.ul[isp,iat,l,:]/rx, radf.udot [isp,iat,l,:]/rx]
            
            brs = [[] for l in range(in1.nt)]
            for l in range(len(in1.nLO_at_ind[iat])):
                brs[l] = [radf.us[isp,iat,l,:]/rx, radf.usdot [isp,iat,l,:]/rx]
                brs[l] += [radf.uslo[isp,iat,l,ilo,:]/rx for ilo in in1.nLO_at_ind[iat][l]]
            for l in range(len(in1.nLO_at_ind[iat]),in1.nt):
                brs[l] = [radf.us[isp,iat,l,:]/rx, radf.usdot [isp,iat,l,:]/rx]

            #for l in range(in1.nt):
            #    print 'nrf=', len(ars[l]), nrf[iat][l]
            for lxc in range(len(lmxc[iat])):
                lx,mx = lmxc[iat][lxc]
                lb = abs(lx)
                for l2 in range(in1.nt):                        # right radial function
                    for l1 in range(in1.nt):                    # left radial function
                        if not( abs(lb-l2) <= l1  and l1 <= (lb+l2) ) and not( abs(lb-l1) <= l2 and l2 <= (lb+l1) ): continue
                        nr1 = len(ars[l1])
                        for ir2 in range(len(ars[l2])):  # how many functions we have at l2?
                            a3 = ars[l2][ir2] * Vxclm[iat][lxc]
                            b3 = brs[l2][ir2] * Vxclm[iat][lxc]
                            for ir1 in range(len(ars[l1])): # how many functions we have at l1?
                                #print 'iat=', iat, 'lxc=', lxc, 'l2=', l2, 'l1=', l1, 'ir2=', ir2, 'ir1=', ir1, 'res=', rr
                                self.uxcu[iat, lxc, l2*in1.nt+l1, ir2*nr1+ir1 ] = rd.rint13g(strc.rel, ars[l1][ir1], brs[l1][ir1], a3, b3, dh, npt, strc.r0[iat])
        if self.PRINT:
            print('-'*32, file=fout)
            print('matrix elements of Vxc', file=fout)
            print('-'*32, file=fout)
            for iat in range(strc.nat):
                for lxc in range(len(lmxc[iat])):
                    lb = abs(lmxc[iat][lxc][0])
                    for l2 in range(in1.nt):
                        nr2 = nrf[iat][l2]
                        for l1 in range(in1.nt):
                            if not( abs(lb-l2) <= l1  and l1 <= (lb+l2) ) and not( abs(lb-l1) <= l2 and l2 <= (lb+l1) ): continue
                            nr1 = nrf[iat][l1]
                            for ir2 in range(nr2):
                                for ir1 in range(nr1):
                                    print('iat=%2d lxc=%2d l2=%2d l1=%2d ir2=%2d ir1=%2d uxcu=%14.10f' % (iat+1, lxc+1, l2, l1, ir2+1, ir1+1, self.uxcu[iat, lxc, l2*in1.nt+l1, ir2*nr1+ir1 ] ), file=fout)
                                
                                
    def set_band_par2(self, io, core, nspin, fout):
        (io_ibgw, io_nbgw, io_emingw, io_emaxgw) = (io['ibgw'], io['nbgw'], io['emingw'], io['emaxgw'])
        
        nbnd = shape(self.Ebnd)[2]
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
        self.nbands_x = self.ncg_x + self.nomax_numin[0]+1 # only occupied bands are needed
        self.nbands_c = self.ncg_c + self.nbmaxsc     # some unoccupied bands are also needed
        print(' Number of bands considered in Sx(nbands_x):', self.nbands_x, file=fout)
        print(' Number of bands considered in Sc(nbands_c):', self.nbands_c, file=fout)
        if io_ibgw < 0:
            nocc_at_k = [[len([x for x in self.Ebnd[isp,ik,:] if x<io_emingw]) for ik in range(nkp)] for isp in range(nspin)]# how many bands below io['emingw'] at each k-point
            self.ibgw = min(list(map(min,nocc_at_k)))
        else:
            self.ibgw = io_ibgw
        if io_nbgw <= 0:
            nocc_at_k = [[len([x for x in self.Ebnd[isp,ik,:] if x<io_emaxgw]) for ik in range(nkp)] for isp in range(nspin)]# how many bands below io['emingw'] at each k-point
            self.nbgw = max(list(map(max,nocc_at_k)))
            #print [[filter(lambda x: x<io_emaxgw, self.Ebnd[isp,ik,:]) for ik in range(nkp)] for isp in range(nspin)]
            #print 'nbgw=', self.nbgw, 'nocc_at_k=', nocc_at_k
        else:
            self.nbgw = min(io_nbgw,nbnd)

        if self.ibgw > self.nomax_numin[1]:
            print('*KohnShamSystem: WARNING - range of gw bands!! ibgw=', self.ibgw, 'numin=', self.nomax_numin[1], file=fout)
            print("*Now we will set ibgw to 1 ", file=fout)
            self.ibgw = 0
        
        if self.nbgw <= self.nomax_numin[0]:
            print("*KohnShamSystem: WARNING - range of gw bands!! nbgw,nomax=", self.nbgw, self.nomax_numin[0], file=fout)
            print("*Now we will set nbgw to nbmax", file=fout)
            self.nbgw = nbnd

        #print >> fout, ' Index of bands considered in GW calculations: (%d,%d)' %(self.ibgw,self.nbgw), 'in energy range',io_emingw, 'to', io_emaxgw, 'Hartree'
        for ib in range(nbnd):
            if any(self.Ebnd[0,:,ib] > io['emin_tetra']):
                self.ibmin_tetra = ib-1
                break
        for ib in range(nbnd-1,-1,-1):
            if any(self.Ebnd[0,:,ib] < io['emax_tetra']):
                self.ibmax_tetra = ib+1
                break
        nbandsgw = self.nbgw - self.ibgw
        nvelgw = core.nval - 2*self.ibgw
        print(' Nr. of bands (nbmax)               :', nbnd, file=fout)
        print(' Nr. of bands used in P (nbmaxpol)   :', self.nbmaxpol, file=fout)
        print(' Nr. of bands for Sc(nbmaxsc)       :', self.nbmaxsc, file=fout)
        print(' Nr. of gw bands (nbandsgw)         :', nbandsgw, file=fout)
        print(' Range of GW bands (ibgw,nbgw)      :', self.ibgw,self.nbgw, 'with energy range', io_emingw, 'to', io_emaxgw, 'Hartree', file=fout)
        print(' Nr. val. electrons (nvel)          :', int(core.nval), file=fout)
        print(' Nr. val. electrons in GW (nvelgw)  :',int(nvelgw), file=fout)
        print(' Nr. core states(ncg)                     :', self.ncg, file=fout)
        print(' Nr. core states for exchange selfx(ncg_x):', self.ncg_x, file=fout)
        print(' Nr. core states for correlat selfc(ncg_c):', self.ncg_c, file=fout)
        print(' Nr. core states for polariz  eps  (ncg_p):', self.ncg_p, file=fout)

        print(' First band using tetrahedron method (ibmin_tetra):', self.ibmin_tetra, file=fout)
        print(' Last  band using tetrahedron method (ibmax_tetra):', self.ibmax_tetra, file=fout)
        
        print('-'*32, file=fout)
            
        
    def set_band_par(self, io_emax_pol, io_emax_sc, io_iop_core, core, nspin, fout):
        band_max = shape(self.Ebnd)[2] # this is max number of bands which are present at all k-points. Some k-points might have more bands, but not all points have them.
        # here we set nbmaxpol, which determines the number of bands used for polmat calculations
        if io_emax_pol < 0:
            self.nbmaxpol = band_max
        else:
            self.nbmaxpol = max([len([x for x in self.Ebnd[0,ik,:] if x < io_emax_pol]) for ik in range(len(self.Ebnd[0]))])
            if nspin==2:
                nbmaxpol = max([len([x for x in self.Ebnd[1,ik,:] if x < io_emax_pol]) for ik in range(len(self.Ebnd[1]))])
                self.nbmaxpol = max(self.nbmaxpol,nbmaxpol)
            
        if io_emax_sc < 0:
            self.nbmaxsc = band_max
        else:
            self.nbmaxsc = max([len([x for x in self.Ebnd[0,ik,:] if x < io_emax_sc]) for ik in range(len(self.Ebnd[0]))])
            if nspin==2:
                nbmaxsc = max([len([x for x in self.Ebnd[1,ik,:] if x < io_emax_sc]) for ik in range(len(self.Ebnd[1]))])
                self.nbmaxsc = max(self.nbmaxsc,nbmaxsc)
        
        #  Set the number of core states considered in the summation over states
        ncg = len(core.corind) # number of all core states
        self.ncg_x = ncg       # for exchange
        if io_iop_core == 0:
            self.ncg_c, self.ncg_p  = ncg, ncg
        elif io_iop_core == 1:
            self.ncg_p, self.ncg_c  = 0, ncg
        else:
            self.ncg_c, self.ncg_p = 0, 0
        self.ncg = ncg
        
    def core_valence_integrals(self, case, in1, strc, core, radf, nspin, fout):
        # momradintc
        ilocals={}
        lomaxp1 = shape(in1.nlo)[0]
        # iul_ucl   = [<   u^v_{l+1}| d/dr -l/r |u^c_l>, <   u^v_{l-1}| d/dr +(l+1)/r |u^c_l>]
        # iudl_ucl  = [<udot^v_{l+1}| d/dr -l/r |u^c_l>, <udot^v_{l-1}| d/dr +(l+1)/r |u^c_l>]
        # iulol_ucl = [<ulo_{l+1}| d/dr - l/r |u^c_l>, <ulo_{l-1}| d/dr+(l+1)/r |u^c_l>]
        # iucl_ul   = [< u^c_l | d/dr -(l-1)/r|   u^v_{l-1}>, < u^c_l | d/dr +(l+2)|   u^v_{l+1}>]
        # iucl_udl  = [< u^c_l | d/dr -(l-1)/r|udot^v_{l-1}>, < u^c_l | d/dr +(l+2)|udot^v_{l+1}>]
        # iucl_ulol = [< u^c_l | d/dr -(l-1)/r|   ulo_{l-1}>, < u^c_l | d/dr +(l+2)|   ulo_{l+1}>]
        # iucl_ucl  = [< u^c_l | d/dr -(l-1)/r|   u^c_{l-1}>, < u^c_l | d/dr +(l+2)/r |u^c_{l+1}>] 
        #
        # <u^v_{l+1}|d/dr  -  l/r|u^c_l> = Integrate[ {r*d/dr(u_{c,l}/r) - l   u_{c,l}/r }* u_{v,l+1} , {r,0,R_MT}] = Integrate[ (d/dr(u_{c,l}) - u_{c,l}/r - l   u_{c,l}/r) * u_{v,l+1}, {r,0,R_MT}]
        # <u^v_{l-1}|d/dr+(l+1)/r|u^c_l> = Integrate[ {r*d/dr(u_{c,l}/r)+(l+1) u_{c,l}/r }* u_{v,l-1} , {r,0,R_MT}] = Integrate[ (d/dr(u_{c,l}) - u_{c,l}/r+(l+1) u_{c,l}/r) * u_{v,l-1}, {r,0,R_MT}]
        self.iul_ucl  = [[[] for i in range(strc.nat)] for j in range(nspin)]
        self.iudl_ucl = [[[] for i in range(strc.nat)] for j in range(nspin)]
        self.iulol_ucl  = [[[] for i in range(strc.nat)] for j in range(nspin)]
        # <u^c_l|d/dr-(l-1)/r|u^v_{l-1}> = Integrate[ u_{c,l} *{ r*d/dr(u_{v,l-1}/r) -(l-1)*u_{v,l-1}/r }, {r,0,R_MT}] = Integrate[ u_{c,l} *{ d/dr(u_{v,l-1}) - u_{v,l-1}/r -(l-1)u_{v,l-1}/r } , {r,0,R_MT}]
        # <u^c_l|d/dr+(l+2)/r|u^v_{l+1}> = Integrate[ u_{c,l} *{ r*d/dr(u_{v,l+1}/r) +(l+2)*u_{v,l+1}/r }, {r,0,R_MT}] = Integrate[ u_{c,l} *{ d/dr(u_{v,l+1}) - u_{v,l+1}/r +(l+2)u_{v,l+1}/r } , {r,0,R_MT}]
        # Note that major (u_v) and minor component (u_{minor}) for valence electrons are related by derivative:
        #                           r*d/dr(u_v/r) = (d u_v/dr - u_v/r) = 2*u_{minor} 
        self.iucl_ul  = [[[] for i in range(strc.nat)] for j in range(nspin)]
        self.iucl_udl = [[[] for i in range(strc.nat)] for j in range(nspin)]
        self.iucl_ulol = [[[] for i in range(strc.nat)] for j in range(nspin)]
        # <u^c_l| d/dr - (l-1)/r |u^c_{l-1}> = Integrate[ u_{core,l} * { r d/dr (u_{core,l-1}/r) - (l-1) * u_{core,l-1}/r }, {r,0,R_MT}]
        # <u^c_l| d/dr + (l+2)/r |u^c_{l+1}> = Integrate[ u_{core,l} * { r d/dr (u_{core,l+1}/r) + (l+2) * u_{core,l+1}/r }, {r,0,R_MT}]
        for isp in range(nspin):
            for iat in range(strc.nat):
                rx, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
                _iul_ucl_   = zeros((2,len(core.l_core[iat])),order='F')
                _iudl_ucl_  = zeros((2,len(core.l_core[iat])),order='F')
                _iucl_ul_   = zeros((2,len(core.l_core[iat])),order='F')
                _iucl_udl_  = zeros((2,len(core.l_core[iat])),order='F')
                al_lo = reduce(lambda x,y: x+y, in1.nLO_at_ind[iat])
                nloat = 1
                if al_lo: nloat = max(al_lo) + 1
                _iulol_ucl_ = zeros((2,nloat,len(core.l_core[iat])),order='F')
                _iucl_ulol_ = zeros((2,nloat,len(core.l_core[iat])),order='F')
                #_iucl_ucl_ = zeros((2,len(core.l_core[iat]),len(core.l_core[iat])))
                for ic,lc in enumerate(core.l_core[iat]):
                    ucl  = core.ul_core[isp][iat][ic][:npt]
                    ucor = ucl/rx
                    ucpl = radd.derv(ucl,rx) - ucor  # ucpl == r d(u/r)/dr = du/dr - u/r
                    cfxs = [lc, -(lc+1)]
                    for dl in [1,-1]:
                        # Integrate[ {r*d/dr(u_{c,l}/r) -cfx * u_{c,l}/r }* u_{v,l+-1} , {r,0,R_MT}]
                        lv = lc + dl
                        il = int( (1-dl)/2 )
                        cfx = cfxs[il]
                        if lv >= 0:
                            #  iul_ucl  = [<   u^v_{l+1}| d/dr -l/r |u^c_l>, <   u^v_{l-1}| d/dr +(l+1)/r |u^c_l>]
                            #  iudl_ucl = [<udot^v_{l+1}| d/dr -l/r |u^c_l>, <udot^v_{l-1}| d/dr +(l+1)/r |u^c_l>]
                            #           = int(u_{l+1} d u_{core_l}/dr r^2 dr) - l * int(u_{l+1} u_{core_l} r dr)
                            ul   = radf.ul  [isp,iat,lv]
                            udl  = radf.udot[isp,iat,lv]
                            _iul_ucl_[il,ic]  = rd.rint13_nr(ul, ucpl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(ul, ucor, dh, npt, strc.r0[iat])
                            #  iudl1ucl = <udot_{l+1}| d/dr -l/r | u^c_l> = int(udot_{l+1} d u_{core_l}/dr r^2 dr) - l * int(udot_{l+1} u_{core_l} r dr)
                            _iudl_ucl_[il,ic] = rd.rint13_nr(udl, ucpl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(udl, ucor, dh, npt, strc.r0[iat])
                            #  Now for local orbitals, l< lomax
                            if lc+1 < lomaxp1:
                                for ilo in in1.nLO_at_ind[iat][lv]:
                                    # iulol_ucl = [<ulo_{l+1}| d/dr-l/r |u^c_l>, <ulo_{l-1}| d/dr+(l+1)/r |u^c_l>]
                                    # iulol1ucl = int(ulo_{l+1} ducore_l/dr r^2 dr) - l*int(ulo_{l+1} ucore_l r dr)
                                    ulol = radf.ulo [isp,iat,lv,ilo]
                                    #ilocals[('iulol_ucl',il,isp,iat,ic,ilo)] = rd.rint13_nr(ulol, ucpl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(ulol, ucor, dh, npt, strc.r0[iat])
                                    _iulol_ucl_[il,ilo,ic] = rd.rint13_nr(ulol, ucpl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(ulol, ucor, dh, npt, strc.r0[iat])

                    cfxs = [lc-1, -(lc+2)]
                    for dl in [-1,1]:
                        # iucl_ul = [<u^c_l|d/dr -(l-1)| u^v_{l-1}>, <u^c_l|d/dr +(l+2)| u^v_{l+1}>] = Integrate[ u_{c,l} *{ r*d/dr(u_{v,l-+1}/r) -cfx * u_{v,l-+1}/r }, {r,0,R_MT}]
                        # iucl_udl= [<u^c_l|d/dr -(l-1)| udot^v_{l-1}>, <u^c_l|d/dr +(l+2)| udot^v_{l+1}>]
                        # iucl_ulol=[<u^c_l|d/dr -(l-1)|    ulo_{l-1}>, <u^c_l|d/dr +(l+2)|    ulo_{l+1}>]
                        lv = lc + dl
                        il = int( (dl+1)/2 )
                        cfx = cfxs[il]
                        if lv >= 0:
                            upl  = 2.0*radf.us   [isp,iat,lv,:npt]    # this is equivalent to r*d/dr u_l
                            udpl = 2.0*radf.usdot[isp,iat,lv,:npt]    # this is equivalent to r*d/dr udot_l
                            uor  =     radf.ul   [isp,iat,lv,:npt]/rx # this is u_l/r
                            udor =     radf.udot [isp,iat,lv,:npt]/rx # this is udot_l/r
                            #  iucl1ul = int( u_{core,l} d/dr u_{lv} r^2 dr) - cfx * int( u_{core_l} u_{lv} r dr)
                            _iucl_ul_[il,ic]  = rd.rint13_nr(ucl, upl,  dh, npt, strc.r0[iat]) - cfx*rd.rint13_nr(ucl, uor,  dh, npt, strc.r0[iat])
                            #  iul1udl = int( u_{core,l} d/dr udot_{lv} r^2 dr) - cfx * int( u_{core_l} udot_{lv} r dr)
                            _iucl_udl_[il,ic] = rd.rint13_nr(ucl, udpl, dh, npt, strc.r0[iat]) - cfx*rd.rint13_nr(ucl, udor, dh, npt, strc.r0[iat])
                            #  And now for local orbitals l< lomax
                            if lv < lomaxp1:
                                for ilo in in1.nLO_at_ind[iat][lv]:
                                    ulopl = 2.0*radf.uslo[isp,iat,lv,ilo,:npt]   # this is equivalent to d*d/dr u_{lo}
                                    uloor =     radf.ulo[isp,iat,lv,ilo,:npt]/rx # this is u_{lo}/r
                                    # iucl1ulol = int( u_{core,l} d/dr ulo_{lv} r^2 dr) - cfx * int( u_{core,l} ulo_{lv} r dr)
                                    #ilocals[('iucl_ulol',il,isp,iat,ic,ilo)] = rd.rint13_nr(ucl, ulopl, dh, npt, strc.r0[iat]) - cfx*rd.rint13_nr(ucl, uloor, dh, npt, strc.r0[iat])
                                    _iucl_ulol_[il,ilo,ic] = rd.rint13_nr(ucl, ulopl, dh, npt, strc.r0[iat]) - cfx*rd.rint13_nr(ucl, uloor, dh, npt, strc.r0[iat])
                    
                    # Now the core core elements
                    # iucl_ucl = [< u^c_l |d/dr-(l-1)/r|u^c_{l-1}>, < u^c_l |d/dr+(l+2)/r|u^c_{l+1}>] = Integrate[ u_{core,l} * { r d/dr (u_{core,l-+1}/r) - cfx * u_{core,l-+1}/r }, {r,0,R_MT}]
                    cfxs = [lc-1, -(lc+2)]
                    for jc,cl in enumerate(core.l_core[iat]):
                        for dl in [-1,1]:
                            if cl == lc + dl:
                                il = int( (dl+1)/2 )
                                cfx = cfxs[il]
                                ul  = core.ul_core[isp][iat][jc][:npt]
                                uor = ul/rx
                                upl = radd.derv(ul,rx) - uor  # upl == r d(u/r)/dr = du/dr - u/r
                                #_iucl_ucl_[il,ic,jc] = rd.rint13_nr(ucl, upl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(ucl, uor, dh, npt, strc.r0[iat])
                                ilocals[('iucl_ucl',il,isp,iat,ic,jc)] = rd.rint13_nr(ucl, upl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(ucl, uor, dh, npt, strc.r0[iat])
                self.iul_ucl  [isp][iat] = _iul_ucl_
                self.iudl_ucl [isp][iat] = _iudl_ucl_
                self.iulol_ucl[isp][iat] = _iulol_ucl_
                self.iucl_ul  [isp][iat] = _iucl_ul_
                self.iucl_udl [isp][iat] = _iucl_udl_
                self.iucl_ulol[isp][iat] = _iucl_ulol_
                #
        if Debug_Print:
            for isp in range(nspin):
                for iat in range(strc.nat):
                    for ic,lc in enumerate(core.l_core[iat]):
                        print('nLO_at_ind[l+1,iat]=', in1.nLO_at_ind[iat][lc+1], file=fout)
                        if lc>0 : print('nLO_at_ind[l-1,iat]=', in1.nLO_at_ind[iat][lc-1], file=fout)
                        
                        print('iul1ucl [%2d,%2d]=%10.7f' %(iat,ic,self.iul_ucl [isp][iat][0,ic]), file=fout)
                        print('iudl1ucl[%2d,%2d]=%10.7f' %(iat,ic,self.iudl_ucl[isp][iat][0,ic]), file=fout)
                        print('iucl1ul [%2d,%2d]=%10.7f' %(iat,ic,self.iucl_ul [isp][iat][0,ic]), file=fout)
                        print('iucl1udl[%2d,%2d]=%10.7f' %(iat,ic,self.iucl_udl[isp][iat][0,ic]), file=fout)
                        #     
                        print('iulucl1 [%2d,%2d]=%10.7f' %(iat,ic,self.iul_ucl [isp][iat][1,ic]), file=fout)
                        print('iudlucl1[%2d,%2d]=%10.7f' %(iat,ic,self.iudl_ucl[isp][iat][1,ic]), file=fout)
                        print('iuclul1 [%2d,%2d]=%10.7f' %(iat,ic,self.iucl_ul [isp][iat][1,ic]), file=fout)
                        print('iucludl1[%2d,%2d]=%10.7f' %(iat,ic,self.iucl_udl[isp][iat][1,ic]), file=fout)
                        #
                        for ilo in in1.nLO_at_ind[iat][lc+1]:
                            print('iulol1ucl[%2d,%2d,%2d]=%10.7f' % (iat,ic,ilo,self.iulol_ucl[isp][iat][0,ilo,ic]), file=fout)
                        if lc>0:
                            for ilo in in1.nLO_at_ind[iat][lc-1]:
                                print('iulolucl1[%2d,%2d,%2d]=%10.7f' % (iat,ic,ilo,self.iulol_ucl[isp][iat][1,ilo,ic]), file=fout)
                            for ilo in in1.nLO_at_ind[iat][lc-1]:
                                print('iucl1ulol[%2d,%2d,%2d]=%10.7f' % (iat,ic,ilo,self.iucl_ulol[isp][iat][0,ilo,ic]), file=fout)
                        for ilo in in1.nLO_at_ind[iat][lc+1]:
                            print('iuclulol1[%2d,%2d,%2d]=%10.7f' % (iat,ic,ilo,self.iucl_ulol[isp][iat][1,ilo,ic]), file=fout)
                        for jc,cl in enumerate(core.l_core[iat]):
                            if ('iucl_ucl',0,isp,iat,ic,jc) in ilocals:
                                print('iucl1ucl[%2d,%2d,%2d]=%10.7f' % (iat,ic,jc, ilocals[('iucl_ucl',0,isp,iat,ic,jc)]), file=fout)
                            if ('iucl_ucl',1,isp,iat,ic,jc) in ilocals:
                                print('iuclucl1[%2d,%2d,%2d]=%10.7f' % (iat,ic,jc, ilocals[('iucl_ucl',1,isp,iat,ic,jc)]), file=fout)
                
                #iucl1ucl == iucl_ucl[0]
                #iuclucl1 == iucl_ucl[1]
                #iul1ucl  [isp][iat] = _iul_ucl_[0,:]
                #iudl1ucl [isp][iat] = _iudl_ucl_[0,:]
                #iulol1ucl[isp][iat] = _iulol_ucl_[0,:]
                #iulucl1  [isp][iat] = _iul_ucl_[1,:]
                #iudlucl1 [isp][iat] = _iudl_ucl_[1,:]
                #iulolucl1[isp][iat] = _iulol_ucl_[1,:]
                #iucl1ul  [isp][iat] = _iucl_ul_[0,:]
                #iucl1udl [isp][iat] = _iucl_udl_[0,:]
                #iucl1ulol[isp][iat] = _iucl_ulol_[0,:]
                #iuclul1  [isp][iat] = _iucl_ul_[1,:]
                #iucludl1 [isp][iat] = _iucl_udl_[1,:]
                #iuclulol1[isp][iat] = _iucl_ulol_[1,:]
                #call momradintv(iat,isp)
        
    def valence_valence_integrals(self, case, in1, strc, core, radf, nspin, fout):
        # momradintv
        self.iul_ul   = zeros((2,in1.nt-1,strc.nat,nspin),order='F')
        self.iul_udl  = zeros((2,in1.nt-1,strc.nat,nspin),order='F')
        self.iudl_ul  = zeros((2,in1.nt-1,strc.nat,nspin),order='F')
        self.iudl_udl = zeros((2,in1.nt-1,strc.nat,nspin),order='F')
        lomaxp1 = shape(in1.nlo)[0]
        self.ilocals={}
        for isp in range(nspin):
            for iat in range(strc.nat):
                #npt = strc.nrpt[iat]
                #dh  = log(strc.rmt[iat]/strc.r0[iat])/(npt - 1)      # logarithmic step for the radial mesh
                #dd = exp(dh)
                #rx = strc.r0[iat]*dd**range(npt)
                rx, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
                for l in range(in1.nt-1):
                    l1s = [l+1,l]
                    l2s = [l,l+1]
                    cfxs = [l,-(l+2)]
                    for il in [0,1]:
                        l1, l2, cfx = l1s[il], l2s[il], cfxs[il]
                        ul   =   radf.ul   [isp,iat,l1]
                        udl  =   radf.udot [isp,iat,l1]
                        upl  = 2*radf.us   [isp,iat,l2] # note 2*us == d/dr ul(r) - ul(r)/r = r d/dr( ul(r)/r )
                        udpl = 2*radf.usdot[isp,iat,l2]
                        uor  =   radf.ul   [isp,iat,l2]/rx
                        udor =   radf.udot [isp,iat,l2]/rx
                        #  iul_ul  = < u_{l1}  | d/dr - cfx/r |  u_{l2} > = int(u_{l1} d/dr(u_l2) r^2 dr) - cfx*int(u_{l1} u_l2 r dr)
                        self.iul_ul[il,l,iat,isp]  = rd.rint13_nr(ul, upl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(ul, uor, dh, npt, strc.r0[iat])
                        #  iul_udl = < u_{l1}  | d/dr - cfx/r |udot_{l2}> = int(u_{l1} d udot_l2/dr r^2 dr) - cfx*int(u_{l1} udot_l2 r dr)
                        self.iul_udl[il,l,iat,isp] = rd.rint13_nr(ul, udpl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(ul, udor, dh, npt, strc.r0[iat])
                        #  iudl_ul = <udot_{l1}| d/dr - cfx/r |  u_{l2} > = int(udot_{l1} du_l2/dr r^2 dr) - cfx*int(udot_{l1} u_l2 r dr)
                        self.iudl_ul[il,l,iat,isp] = rd.rint13_nr(udl, upl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(udl, uor, dh, npt, strc.r0[iat])
                        #  iudl_udl= <udot_{l1}| d/dr - cfx/r |udot_{l2}> = int(udot_{l1} d udot_l2/dr r^2 dr) - cfx*int(udot_{l1} udot_l2 r dr)
                        self.iudl_udl[il,l,iat,isp]= rd.rint13_nr(udl, udpl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(udl, udor, dh, npt, strc.r0[iat])
                        if l1<lomaxp1:
                            for ilo in in1.nLO_at_ind[iat][l1]:
                                ulol = radf.ulo[isp,iat,l1,ilo]
                                #  iulol_ul = < ulo_{l1}| d/dr - cfx/r | u_{l2} > = int(ulo_{l1} du_l2/dr r^2 dr) - cfx*int(ulo_{l1} u_l2 r dr)
                                self.ilocals[('iulol_ul',il,isp,iat,l,ilo)] = rd.rint13_nr(ulol, upl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(ulol, uor, dh, npt, strc.r0[iat])
                                #  iulol_udl= < ulo_{l1}| d/dr - cfx/r |udot_{l2}> = int(ulo_{l1} d udot_l2/dr r^2 dr) - cfx*int(ulo_{l1} udot_l2 r dr)
                                self.ilocals[('iulol_udl',il,isp,iat,l,ilo)]= rd.rint13_nr(ulol, udpl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(ulol, udor, dh, npt, strc.r0[iat])
                        #  And now for local orbitals, only for l<= lomax
                        if l2<lomaxp1:
                            for jlo in in1.nLO_at_ind[iat][l2]:
                                ulopl = 2*radf.uslo[isp,iat,l2,jlo]
                                uloor = radf.ulo[isp,iat,l2,jlo]/rx
                                #  iul_ulol = < u_{l1}  | d/dr - cfx/r | ulo_{l2}> = int(u_{l1} d ulo_l2/dr r^2 dr) - cfx*int(u_{l1} ulo_l2 r dr)
                                self.ilocals[('iul_ulol',il,isp,iat,l,jlo)] = rd.rint13_nr(ul, ulopl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(ul, uloor, dh, npt, strc.r0[iat])
                                # iudl_ulol = <udot_{l1}| d/dr - cfx/r | ulo_{l2}> = int(udot_{l1} d ulo_l2/dr r^2 dr) - cfx*int(udot_{l1} ulo_l2 r dr)
                                self.ilocals[('iudl_ulol',il,isp,iat,l,jlo)] = rd.rint13_nr(udl, ulopl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(udl, uloor, dh, npt, strc.r0[iat])
                                for ilo in in1.nLO_at_ind[iat][l1]:
                                    ulol = radf.ulo[isp,iat,l1,ilo]
                                    # iulol_ulol = < ulo_{l1,ilo}| d/dr - cfx/r | ulo_{l2,jlo}> = int(ulo_{l1} d ulo_l2/dr r^2 dr) - cfx*int(ulo_{l1} ulo_l2 r dr)
                                    self.ilocals[('iulol_ulol',il,isp,iat,l,jlo,ilo)] = rd.rint13_nr(ulol, ulopl, dh, npt, strc.r0[iat]) - cfx * rd.rint13_nr(ulol, uloor, dh, npt, strc.r0[iat])


        # printing
        if Debug_Print:
            for isp in range(nspin):
                for iat in range(strc.nat):
                    rx, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
                    for l in range(in1.nt-1):
                        print('iul1ul  (%2d,%2d)=%10.7f' % (iat,l,self.iul_ul  [0,l,iat,isp]), ' iulul1  (%2d,%2d)=%10.7f' % (iat,l,self.iul_ul[1,l,iat,isp]), file=fout)
                        print('iul1udl (%2d,%2d)=%10.7f' % (iat,l,self.iul_udl [0,l,iat,isp]), ' iuludl1 (%2d,%2d)=%10.7f' % (iat,l,self.iul_udl[1,l,iat,isp]), file=fout)
                        print('iudl1ul (%2d,%2d)=%10.7f' % (iat,l,self.iudl_ul [0,l,iat,isp]), ' iudlul1 (%2d,%2d)=%10.7f' % (iat,l,self.iudl_ul[1,l,iat,isp]), file=fout)
                        print('iudl1udl(%2d,%2d)=%10.7f' % (iat,l,self.iudl_udl[0,l,iat,isp]), ' iudludl1(%2d,%2d)=%10.7f' % (iat,l,self.iudl_udl[1,l,iat,isp]), file=fout)
            # printing local orbitals
            to_print=[]
            for iky in list(self.ilocals.keys()):
                if iky[1]==0:
                    name = re.sub('_','1',iky[0])
                else:
                    name = re.sub('_','',iky[0])+'1'
                #print >> fout, '%-9s%s=%10.7f' % (name,str(tuple(iky[3:])),self.ilocals[iky])
                to_print.append( (name,tuple(iky[3:]),self.ilocals[iky]) )
            for itm in sorted(to_print,key=lambda c: c[1]):
                print('%-9s%s=%10.7f' % (itm[0],str(itm[1]),itm[2]), file=fout)
            
        # DICTIONARY
        #iul1ul    ,iulul1    = iul_ul[0,1]
        #iul1udl   ,iuludl1   = iul_udl[0,1]
        #iudl1ul   ,iudlul1   = iudl_ul[0,1]
        #iudl1udl  ,iudludl1  = iudl_udl[0,1]
        #iulol1ul  ,iulolul1  = iulol_ul[0,1] 
        #iulol1udl ,iuloludl1 = iulol_udl[0,1]
        #iul1ulol  ,iululol1  = iul_ulol[0,1]
        #iudl1ulol ,iudlulol1 = iudl_ulol[0,1]
        #iulol1ulol,iulolulol1= iulol_ulol[0,1]
    
    def Give_fortran_ilocals(self, isp,strc,in1):
        #for l in range(in1.nt-1):
        #    l1s = [l+1,l]
        #    l2s = [l,l+1]
        #    for il in [0,1]:
        #        l1, l2 = l1s[il], l2s[il]
        #        for jlo in in1.nLO_at_ind[iat][l2]:
        #            for ilo in in1.nLO_at_ind[iat][l1]:
        #                print 'l=',l, 'il=',il, 'jlo=',jlo, 'ilo=',ilo, '     l1=', l1, 'l2=', l2
        #
        #iat=0
        #print 'nLO_at=', in1.nLO_at[:,iat]
        #lomax = shape(in1.nLO_at)[0]-1
        #print 'lomax=', lomax
        #for l in range(in1.nt-1):
        #    for jlo in range(in1.nLO_at[l,iat]):
        #        for ilo in range(in1.nLO_at[l+1,iat]):
        #            print 'need l=', l, 'il=', 0, 'ilo=', ilo+1, 'jlo=', jlo+1
        #            print 'need l=', l, 'il=', 1, 'jlo=', jlo+1, 'ilo=', ilo+1
        kys = list(self.ilocals.keys())
        lomax, nlomax = 1, 1
        if kys:
            lomax = max([kys[i][4] for i in range(len(kys))])+1
            nlomax = max([kys[i][5] for i in range(len(kys))])
            #print 'nlomax=', nlomax, 'lomax=', lomax

        iulol_ul   = zeros((2,nlomax,lomax,strc.nat), order='F')
        iulol_udl  = zeros((2,nlomax,lomax,strc.nat), order='F')
        iul_ulol   = zeros((2,nlomax,lomax,strc.nat), order='F')
        iudl_ulol  = zeros((2,nlomax,lomax,strc.nat), order='F')
        iulol_ulol = zeros((2,nlomax,nlomax,lomax,strc.nat), order='F')
        for k in list(self.ilocals.keys()):
            if k[0]=='iul_ulol' and k[2]==isp:
                (il,_isp_,iat,l,jlo) = k[1:]
                iul_ulol[il,jlo-1,l,iat] = self.ilocals[k]
            elif k[0]=='iudl_ulol' and k[2]==isp:
                (il,_isp_,iat,l,jlo) = k[1:]
                iudl_ulol[il,jlo-1,l,iat] = self.ilocals[k]
            elif k[0]=='iulol_ul' and k[2]==isp:
                (il,_isp_,iat,l,jlo) = k[1:]
                iulol_ul[il,jlo-1,l,iat] = self.ilocals[k]
            elif k[0]=='iulol_udl' and k[2]==isp:
                (il,_isp_,iat,l,jlo) = k[1:]
                iulol_udl[il,jlo-1,l,iat] = self.ilocals[k]
            elif k[0]=='iulol_ulol' and k[2]==isp:
                (il,_isp_,iat,l,jlo,ilo) = k[1:]
                iulol_ulol[il,ilo-1,jlo-1,l,iat] = self.ilocals[k]
                #print 'setting il=', il, 'l=', l, 'ilo=', ilo, 'jlo=', jlo, ' = ', self.ilocals[k]
        return (iulol_ul, iulol_udl, iul_ulol, iudl_ulol, iulol_ulol)

            
