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
from datetime import date
from numpy import *
from numpy import linalg
from timeit import default_timer as timer
from scipy import optimize
import itertools
from functools import reduce
from numba import jit
#from numba.typed import List
from scipy import special
import subprocess, shutil


from cmn import Ry2H,H2eV,Ry2eV,H2Ry,Br2Ang
from inout import InOut
from kqmesh import KQmesh
import mcommon as mcmn
from kohnsham import KohnShamSystem
from planewaves import PlaneWaves
from matel2band import MatrixElements2Band
from productbasis import ProductBasis
from kweights import Kweights
import gwienfile as w2k

import radials as rd        # radial wave functions
import lapwcoeff as lapwc   # computes lapw coefficients alm,blm,clm
import gaunt as gn
import for_vxcnn as fvxcn   # matrix elements for Vxcn
import sphbes
from pylab import linalg as la
la_matmul = la.matmul
from pylab import *

class Spherical2CubicW90:
    def __init__(self):
        s2 = sqrt(2.)
        T2Cs = [[1]]
        #           pz           px                 py
        T2Cp = [ [0,1,0], [1/s2, 0, -1/s2], [-1j/s2, 0, -1j/s2]]
        #           z2         xz                 yz                 x2-y2             xy
        T2Cd = [[0,0,1,0,0],[0,1/s2,0,-1/s2,0],[0,1j/s2,0,1j/s2,0],[1/s2,0,0,0,1/s2],[1j/s2,0,0,0,-1j/s2]]
        #       fz3              fxz2                  fyz2                    fz(x2-y2)             fxyz                     fx(x2-3y2)             fy(3x2-y2)
        T2Cf = [[0,0,0,1,0,0,0],[0,0,1/s2,0,-1/s2,0,0],[0,0,1j/s2,0,1j/s2,0,0],[0,1/s2,0,0,0,1/s2,0],[0,1j/s2,0,0,0,-1j/s2,0],[1/s2,0,0,0,0,0,-1/s2],[1j/s2,0,0,0,0,0,1j/s2]]
        self.T = [ array(T2Cs).T, array(T2Cp).T, array(T2Cd).T, array(T2Cf).T]
        self.N = [['s'],['pz','px','py'],['dz2','dxz','dyz','dx2-y2','dxy'],['fz3','fxz2','fyz2','fz(x2-y2)','fxyz','fx(x2-3y2)','fy(3x2-y2)']]
        
def Path_distance(klist, k2cartes):
    return cumsum( [0]+[ linalg.norm(k2cartes@(klist[ik+1,:]-klist[ik,:]))/Br2Ang for ik in range(len(klist)-1)] )


def Find_which_Gs_from_pw_are_present(pw, heads, all_Ek, all_Gs, keep_los=False):
    #
    indgkir=[]
    for irk in range(len(all_Ek)):
        k, kname, _wgh_, _ios_, n0, nb = heads[irk]
        nG,n3 = shape(all_Gs[irk])
        ind = -ones(nG,dtype=int)
        for i in range(nG):
            iG = all_Gs[irk][i]          # reciprocal vector read from vector file
            ii = pw.ig0[tuple(iG)]   # what is the index of this G in my gvec (previously generated fixed reciprocal mesh)?
            if keep_los:
                ind[i]=ii
                #print('irk=%3d i=%3d indgkir=%3d' % (irk+1, i, ind[i]), 'iG=', iG )
            else:
                if (ii not in ind):
                    ind[i] = ii
                    #print('irk=%3d i=%3d indgkir=%3d' % (irk+1, i, ind[i]), 'iG=', iG )#, file=fout)
                #else:
                    #print('irk=%3d i=%3d indgkir=%3d' % (irk+1, i, ind[i]), 'iG=', iG, 'this is LO' )#, file=fout)
        indgkir.append(ind)
    return indgkir

    
def PrintM(A):
    ni,nj = shape(A)
    for i in range(ni):
        for j in range(nj):
            print('%6.3f '% A[i,j], end='')
        print()
    
def PrintMC(A, file=sys.stdout):
    ni,nj = shape(A)
    for i in range(ni):
        for j in range(nj):
            print('%10.5f %10.5f '% (A[i,j].real,A[i,j].imag), end='', file=file)
        print(file=file)
    

    
def GetRealSpaceMeshes(kqm,larmax=40):
    arbas = linalg.inv(kqm.k2icartes)
    ndiv = kqm.ndiv
    nall = ndiv[0]*ndiv[1]*ndiv[2]
    rind0 = zeros((nall,3), dtype=int)

    for ii,ir in enumerate(itertools.product(range(0,ndiv[0]),range(0,ndiv[1]),range(0,ndiv[2]))):
        rind0[ii] = ir
    rind = dot(rind0, arbas)
    return(rind)

#@jit(nopython=True)
def FindInArray(ang, angles):
    for ii,a in enumerate(angles):
        if linalg.norm(ang-a) < 1e-6:
            return ii
    return -1

#def Convert_iorb2text(orbs, S2C, anames, lmaxp1):
#    ndf = len(anames)
#    names = [[] for i in range(ndf*lmaxp1**2)]
#    for l in range(lmaxp1):
#        for idf in range(ndf):
#            #namesl = [anames[idf]+':'+s for s in S2C.N[l]]
#            namesl = [anames[idf]+':l='+str(l)+',mr='+str(mr+1) for mr in range((2*l+1))]
#            ist=idf+ndf*l**2
#            ind=idf+ndf*(l+1)**2
#            names[ist:ind:ndf] = namesl[:]
#    
#    names = [names[i] for i in orbs]
#    return names
#
#def Convert_iorb2text2(orbs, anames, lmaxp1):
#    ndf = len(anames)
#    names = [[] for i in range(ndf*lmaxp1**2)]
#    for l in range(lmaxp1):
#        for idf in range(ndf):
#            namesl = [str(idf+1)+':'+s for s in S2C.N[l]]
#            ist=idf+ndf*l**2
#            ind=idf+ndf*(l+1)**2
#            names[ist:ind:ndf] = namesl[:]
#    
#    names = [names[i] for i in orbs]
#    return names

class CWannier90:
    def __init__(self, nbs,nbe, case, strc_mult, strc_aname, kqm_ndiv, kqm_klist, kqm_LCM, kqm_k2icartes, _w90p_=None):
        self.nbs, self.nbe = nbs, nbe
        self.case = case
        self.kqm_ndiv = kqm_ndiv
        self.kqm_klist = kqm_klist
        self.kqm_LCM = kqm_LCM
        self.kqm_k2icartes = kqm_k2icartes
        
        self.w90p={}
        self.w90p['num_bands'] = self.nbe-self.nbs
        self.w90p['num_wann'] = self.nbe-self.nbs
        self.w90p['iprint'] = 1
        self.w90p['num_iter'] = 10000
        self.w90p['num_print_cycles'] = 100
        self.w90p['conv_window'] = 3
        self.w90p['dis_num_iter'] = 10000
        self.w90p['write_proj'] = True
        self.w90p['write_xyz'] = True
        self.w90p['translate_home_cell'] = True
        self.w90p['write_hr'] = True
        self.w90p['fermi_surface_plot'] = False
        self.w90p['dos'] = False
        self.w90p['bands_num_points'] = 50
        
        if _w90p_ is not None:
            self.w90p.update(_w90p_)
            
        self.S2C = Spherical2CubicW90()
        self.icartes2f = linalg.inv(self.kqm_k2icartes)
        
        self.aname = []
        for iat in range(len(strc_mult)):
            nam = strc_aname[iat].split()[0] # If space appears in struct file, such name is not working in wannier90, so remove anything after space.
            if strc_mult[iat]==1:
                self.aname.append( nam )
            else:
                self.aname += [ nam+str(ieq+1) for ieq in range(strc_mult[iat])]

    def Convert_iorb2text(self, orbs):
        lmaxp1 = 4
        ndf = len(self.aname)
        names = zeros( (lmaxp1**2,self.np_lo,ndf), dtype=(unicode_, 16))
        for l in range(lmaxp1):
            for m in range(-l,l+1):
                lm = l**2 + l + m
                for idf in range(ndf):
                    nm1 = self.aname[idf]+':l='+str(l)+',mr='+str(l+m+1)
                    names[lm,0,idf] = nm1
                    if self.np_lo>1 :
                        nm2 = self.aname[idf]+':l='+str(l)+',mr='+str(l+m+1) #+',r=2'
                        names[lm,1,idf] = nm2
        names = reshape(names, lmaxp1**2*self.np_lo*ndf)
        orb_names = [names[i] for i in orbs]
        return orb_names
                
    def Prepare_win_file(self, orbs, latgen_rbas, strc_vpos, strc_mult, strc_aname, k_path, semi_cartesian=True, _w90p_=None):
        if _w90p_ is not None:
            self.w90p.update(_w90p_)
        
        fow = open(self.case+'.win', 'w')
        for p,v in self.w90p.items():
            print('%-20s' % p+'=', v, file=fow)
        print('begin unit_cell_cart', file=fow)
        print('Bohr', file=fow)
        for j in range(3):
            print('%12.6f '*3 % tuple(latgen_rbas[:,j]), file=fow)
        print('end unit_cell_cart', file=fow)
        print('', file=fow)
        print('begin atoms_cart', file=fow)
        print('Bohr', file=fow)
        for idf in range(shape(strc_vpos)[1]):
            pos = dot(latgen_rbas,strc_vpos[:,idf])
            print(self.aname[idf], '%12.6f '*3 % tuple(pos), file=fow )
        print('end atoms_cart', file=fow)
        print('', file=fow)
        print('bands_plot = T', file=fow)
        
        print('begin kpoint_path', file=fow)
        for ik in range(len(k_path)-1):
            k1, k2 = k_path[ik][1], k_path[ik+1][1]
            if semi_cartesian:
                k1 = dot(self.icartes2f,array(k1))
                k2 = dot(self.icartes2f,array(k2))
            print('%-5s' % k_path[ik][0], '%10.5f '*3 % tuple(k1), '   %-5s' % k_path[ik+1][0], '%10.5f '*3 % tuple(k2), file=fow)
            
        print('end kpoint_path', file=fow)
        print('', file=fow)

        S2C = Spherical2CubicW90()
        orb_names = self.Convert_iorb2text(orbs)
        #orb_names = Convert_iorb2text2(orbs, self.aname, 4)
        print('guiding_centres = T', file=fow)
        print('begin projections', file=fow)
        for o in orb_names:
            print('  ', o, file=fow)
        print('end projections', file=fow)
        print('', file=fow)
        #k2c_1 = linalg.inv(self.kqm_k2icartes)
        print('mp_grid : %3d %3d %3d' % tuple(self.kqm_ndiv), file=fow)
        print('begin kpoints', file=fow)
        for ik in range(len(self.kqm_klist)):
            ki = dot(self.icartes2f, self.kqm_klist[ik,:]/self.kqm_LCM)
            print('%19.15f'*3 % tuple(ki), file=fow)
        print('end kpoints', file=fow)

    def CheckIsRestart(self):
        fow = open(self.case+'.win', 'r')
        lines = fow.readlines()
        fow = close()
        Found=False
        for line in lines:
            dat = line.split()
            if dat and dat[0]=='restart':
                Found=True
                break
        if not Found:
            fow = open(self.case+'.win', 'w')
            print('restart  = plot', file=fow)
            for line in lines:
                print(line, end='', file=fow)
            fow = close()
        return Found
        
        
    def Read_nnkp(self):
        filekkp = self.case+'.nnkp'
        lines = open(filekkp, 'r').readlines()

        for i,line in enumerate(lines):
            if line[:13]=='begin kpoints':
                i0 = i
                break
        nkp = int(lines[i0+1])
        
        for i,line in enumerate(lines):
            if line[:12]=='begin nnkpts':
                i0 = i
                break
        nntot = int(lines[i0+1]) # number of nearest neighbor k-points
        pair = zeros((nkp,nntot), dtype=int)
        pair_umklap = zeros((nkp,nntot,3), dtype=int)
        ii = i0+1
        for ik in range(nkp):
            for i in range(nntot):
                ii += 1
                dat0 = lines[ii].split()
                dat = [int(dat0[i]) for i in range(len(dat0))]
                if dat[0]!=ik+1:
                    print('ERROR1 in reading '+filekkp, 'line=', dat0, ik, i)
                    sys.exit(0)
                pair[ik,i] = dat[1]-1
                pair_umklap[ik,i,:] = dat[2:5]
                #print(ik+1, pair[ik,i], pair_umklap[ik,i])
        return (pair, pair_umklap)

    def Find_All_Possible_kpair_distances(self, pair, pair_umklap):
        nkp, nntot = shape(pair)
        self.idistance=zeros((nkp,nntot),dtype=int)
        self.distances=[]
        for ik in range(nkp):
            for i in range(nntot):
                jk = pair[ik,i]
                k1 = self.kqm_klist[ik,:]/self.kqm_LCM
                k2 = self.kqm_klist[jk,:]/self.kqm_LCM
                b = k2 - k1 + dot(self.kqm_k2icartes,pair_umklap[ik,i,:])
                aq = sqrt(dot(b,b))
                
                ip = where( abs(array(self.distances)-aq)<1e-6 )[0]
                if len(ip):
                    self.idistance[ik,i] = ip[0]
                else:
                    self.idistance[ik,i] = len(self.distances)
                    self.distances.append(aq)

    def Check_Irreducible_wedge(self, ks, kqm, fout):
        if (len(ks.all_As)==len(kqm.kirlist)):
            print('From vector-file we have irreducible wedge k-points only', file=fout)
            self.Irredk = True
        elif (len(ks.all_As)==len(kqm.klist)):
            print('From vector-file we have all k-points (not irreducible wedge)', file=fout)
            self.Irredk = False
        else:
            print('ERROR len(ks.all_As)=', len(ks.all_As), ' is not one of ', len(kqm.kirlist), 'or', len(ks.All_As))
            print('ERROR len(ks.all_As)=', len(ks.all_As), ' is not one of ', len(kqm.kirlist), 'or', len(ks.All_As), file=fout)
            sys.exit(1)
        
    def FindRelevantOrbitals(self, strc, latgen, ks, kqm, pw, in1, radf, fout, debug=False):
        
        self.Check_Irreducible_wedge(ks, kqm, fout)
        
        _nsp_,_nirk_,nbnd = shape(ks.Ebnd)
        if self.nbe >= nbnd:
            print('We do not have sufficient number of bands in vector file. Reduce number of bands for wannierization!', file=fout)
            print('requested: ', str(self.nbs)+':'+str(self.nbe), 'existing', nbnd, file=fout)
            sys.exit(0)
            
        lmaxp1 = min(4,in1.nt)
        lomaxp1 = shape(in1.nlo)[0]
        ndf = sum(strc.mult)
        if in1.nlomax > 0:
            # If we have local orbitals, we will increase np_lo to 2
            self.np_lo = 2
        else:
            self.np_lo = 1
        
        alfr = zeros((self.nbe-self.nbs,lmaxp1**2,self.np_lo,ndf), dtype=complex)
        asmr = zeros((lmaxp1**2*self.np_lo*ndf), dtype=float)
        self.largest_ilo = -ones((ndf,lomaxp1),dtype=int)

        nextra=0
        orbs=[]
        for irk in range(len(kqm.kirlist)):
            kil = array(kqm.kirlist[irk,:])/float(kqm.LCM)  # k in semi-cartesian form
            ik = kqm.k_ind[irk]   # index in all-kpoints, not irreducible
            if self.Irredk:
                Aeigk = array(ks.all_As[irk][self.nbs:self.nbe,:], dtype=complex)   # eigenvector from vector file
                alm,blm,clm = lapwc.gap2_set_lapwcoef(kil, ks.indgkir[irk], 1, True, ks.nv[irk], pw.gindex, radf.abcelo[0], strc.rmt, strc.vpos, strc.mult, radf.umt[0], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
            else:
                Aeigk = array(ks.all_As[ik][self.nbs:self.nbe,:], dtype=complex)   # eigenvector from vector file
                alm,blm,clm = lapwc.gap2_set_lapwcoef(kil, ks.indgkir[ik], 1, True, ks.nv[irk], pw.gindex, radf.abcelo[0], strc.rmt, strc.vpos, strc.mult, radf.umt[0], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)

            (ngi,nLOmax,ntnt,ndf) = shape(clm)
            (ngk,ntnt,ndf) = shape(alm)
            (nbmax,ngi2) = shape(Aeigk)
            # And now change alm,blm,clm to band basis, which we call alfa,beta,gama
            alfa = reshape( la_matmul(Aeigk, reshape(alm, (ngk,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
            if self.np_lo > 1:
                beta = reshape( la_matmul(Aeigk, reshape(blm, (ngi,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
                gama = reshape( la_matmul(Aeigk, reshape(clm, (ngi,ntnt*ndf*nLOmax)) ), (nbmax,nLOmax,ntnt,ndf) )
                idf2iat = concatenate([[iat]*strc.mult[iat] for iat in range(strc.nat)])
                for idf in range(ndf):
                    iat = idf2iat[idf]
                    for l in range(lomaxp1):
                        r = zeros(in1.nLO_at[l,iat])
                        for ilo in range(in1.lapw[l,iat],in1.nLO_at[l,iat]):
                            s2 = radf.umtlo[0,iat,l,ilo,2]  # <u_l|u_lo>
                            s3 = radf.umtlo[0,iat,l,ilo,3]  # <udot_l|u_lo>
                            alfa[:,l**2:(l+1)**2,idf] += s2 * gama[:,ilo,l**2:(l+1)**2,idf]  # 
                            cg = sqrt(1-s2**2)
                            cb = s3/cg
                            gama[:,ilo,l**2:(l+1)**2,idf] = cg*gama[:,ilo,l**2:(l+1)**2,idf] + cb*beta[:,l**2:(l+1)**2,idf]
                            if irk==0: r[ilo] = sum(abs(gama[:,ilo,l**2:(l+1)**2,idf])**2)
                        if irk==0 and in1.nLO_at[l,iat]:
                            print('idf=', idf, 'l=', l, 'ilo=', ilo, 'r=', r, file=fout)
                            self.largest_ilo[idf,l] = argmax(r)
                            print('largest_ilo[idf='+str(idf)+',l='+str(l)+']='+str(self.largest_ilo[idf,l]), file=fout)
                            
            for idf in range(ndf):
                for l in range(lmaxp1):
                    alfr[:,l**2:(l+1)**2,0,idf] = alfa[:,l**2:(l+1)**2,idf] @ self.S2C.T[l]
                    if self.largest_ilo[idf,l]>=0 :
                        ilo = self.largest_ilo[idf,l]
                        alfr[:,l**2:(l+1)**2,1,idf] = gama[:,ilo,l**2:(l+1)**2,idf] @ self.S2C.T[l]
                    
            alfr = reshape(alfr,(nbmax,lmaxp1**2*self.np_lo*ndf))
            asm = sum(abs(alfr)**2, axis=0)
            asmr += asm
            alfr = reshape(alfr,(nbmax,lmaxp1**2,self.np_lo,ndf))
            
        ind = sorted( range(len(asm)), key=lambda x: -asmr[x] )
        orbs = sorted(ind[:nbmax])

        print('orbitals to be used in projection to wanniers=', orbs, file=fout)
        orb_names = self.Convert_iorb2text(orbs)
        for i in range(len(orb_names)):
            print(i,orb_names[i], file=fout)
        return (orbs)


    def Compute_and_Save_Projection_amn(self, case, orbs, strc, latgen, ks, kqm, pw, in1, radf, fout, naive_projection=True, debug=False):
        
        if naive_projection:
            self.rind = GetRealSpaceMeshes(kqm)         # get all real-space vectors compatible with k-mesh
            krn = dot(self.rind, kqm.klist.T/kqm.LCM)   # r*k
            smatn = exp( -2*pi*1j*krn )/len(self.rind)  # smatn[ir,ik] = exp(-i*r*k)/N
            Idv = dot(conj(smatn).T, smatn) * len(smatn)
            diff = allclose( Idv, identity(len(smatn)) )
            if diff:
                print('exp(-ikr) is correct.', file=fout)
            else:
                print('ERROR : the real space mesh is wrong therefore \sum_r exp(ik1 r) * exp(-ik2 r) != delta(k1-k2)', filefout)
                sys.exit(0)
            self.Hr = zeros((len(smatn),len(orbs),len(orbs)), dtype=complex)
            
        fo_amn = open(case+'.amn', 'w')
        print('# code PyGW:wannier.py', date.today().strftime("%B %d, %Y"), file=fo_amn)
        print(self.nbe-self.nbs, len(kqm.klist), len(orbs), file=fo_amn)
        
        lmaxp1 = min(4,in1.nt)
        lomaxp1 = shape(in1.nlo)[0]
        ndf = sum(strc.mult)
        alfr = zeros((self.nbe-self.nbs,lmaxp1**2,self.np_lo,ndf), dtype=complex)

        print('The singular values of projection to bands <chi|psi>: ', file=fout)
        
        for ik in range(len(kqm.klist)):
            kil = array(kqm.klist[ik,:])/float(kqm.LCM)  # k in semi-cartesian form
            irk = kqm.kii_ind[ik]
            if self.Irredk:
                Aeigk = array(ks.all_As[irk][self.nbs:self.nbe,:], dtype=complex)   # eigenvector from vector file
                if kqm.k_ind[irk] != ik:                          # not irreducible
                    Aeigk *= exp( 2*pi*1j * ks.phase_arg[ik][:])  # adding phase : A *= exp( -2*pi*i*(k+G)*tau[isym] )
                alm,blm,clm = lapwc.gap2_set_lapwcoef(kil, ks.indgk[ik], 1, True, ks.nv[irk], pw.gindex, radf.abcelo[0], strc.rmt, strc.vpos, strc.mult, radf.umt[0], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
                
            else:
                Aeigk = array(ks.all_As[ik][self.nbs:self.nbe,:], dtype=complex)   # eigenvector from vector file
                alm,blm,clm = lapwc.gap2_set_lapwcoef(kil, ks.indgkir[ik], 1, True, ks.nv[ik], pw.gindex, radf.abcelo[0], strc.rmt, strc.vpos, strc.mult, radf.umt[0], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)            

            (ngi,nLOmax,ntnt,ndf) = shape(clm)
            (ngk,ntnt,ndf) = shape(alm)
            (nbmax,ngi2) = shape(Aeigk)
            # And now change alm,blm,clm to band basis, which we call alfa,beta,gama
            alfa = reshape( la_matmul(Aeigk, reshape(alm, (ngk,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
            # We will use two functions, the head and the orthogonalized local orbital, i.e.,
            #    |1> == |u_l> and |2>= (|u_lo>-|u_l><u_l|u_lo>)/sqrt(1-<u_l|u_lo>^2)
            # Note that <1|2>=0 and <2|2>=1
            if self.np_lo > 1:
                beta = reshape( la_matmul(Aeigk, reshape(blm, (ngk,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
                gama = reshape( la_matmul(Aeigk, reshape(clm, (ngk,ntnt*ndf*nLOmax)) ), (nbmax,nLOmax,ntnt,ndf) )
                idf2iat = concatenate([[iat]*strc.mult[iat] for iat in range(strc.nat)])
                for idf in range(ndf):
                    iat = idf2iat[idf]
                    for l in range(lomaxp1):
                        for ilo in range(in1.lapw[l,iat],in1.nLO_at[l,iat]):
                            s2 = radf.umtlo[0,iat,l,ilo,2]  # <u_l|u_lo>
                            s3 = radf.umtlo[0,iat,l,ilo,3]  # <udot_l|u_lo>
                            alfa[:,l**2:(l+1)**2,idf] += s2 * gama[:,ilo,l**2:(l+1)**2,idf]  # 
                            cg = sqrt(1-s2**2)
                            cb = s3/cg
                            gama[:,ilo,l**2:(l+1)**2,idf] = cg*gama[:,ilo,l**2:(l+1)**2,idf] + cb*beta[:,l**2:(l+1)**2,idf]
                            
            for idf in range(ndf):
                for l in range(lmaxp1):
                    alfr[:,l**2:(l+1)**2,0,idf] = alfa[:,l**2:(l+1)**2,idf] @ self.S2C.T[l]
                    if self.largest_ilo[idf,l]>=0 :
                        ilo = self.largest_ilo[idf,l]
                        alfr[:,l**2:(l+1)**2,1,idf] = gama[:,ilo,l**2:(l+1)**2,idf] @ self.S2C.T[l]
                        
            alfr = reshape(alfr,(nbmax,lmaxp1**2*self.np_lo*ndf))
            psi_chi = conj(alfr[:,orbs])  # psi_chi[nbnd,norbs]
            alfr = reshape(alfr,(nbmax,lmaxp1**2,self.np_lo,ndf))
            
            for n in range(len(orbs)):
                for m in range(nbmax):
                    print('%4d%4d %5d %25.12f%25.12f' % (m+1, n+1, ik+1, psi_chi[m,n].real, psi_chi[m,n].imag), file=fo_amn)

            u, s, vh = linalg.svd(psi_chi, full_matrices=True)  # uv[nbnd,norb]
            print('SVD: ik=%3d'%ik, 'k=(%5.3f,%5.3f,%5.3f)' % tuple(self.icartes2f@kil), 's=', list(s), file=fout)
            
            if naive_projection:
                uv = dot(u, vh[:self.nbe-self.nbs,:]) # uv[nbnd,norb]
                #
                eks = ks.Ebnd[0,irk,self.nbs:self.nbe] #
                e_uv = dot(diag(eks),uv)
                Hc = dot(conj(uv).T,e_uv) # Hc[norb,norb]
                #
                for ir in range(shape(smatn)[0]):
                    self.Hr[ir,:,:] += smatn[ir,ik]*Hc[:,:]

        fo_amn.close()
        
        if naive_projection and debug:
            # checking that Hr gives the same Hk
            Hk = tensordot( len(smatn)*smatn.conj().T, self.Hr, axes=(1,0))  # Hk[ik,:,:] = \sum_ir conj(smatn[ir,ik])*Hr[ir,:,:]
            for ik in range(len(Hk)):
                irk = kqm.kii_ind[ik]
                eks = ks.Ebnd[0,irk,self.nbs:self.nbe]
                ene = linalg.eigvalsh(Hk[ik,:,:])
                print('ik=', ik, 'diff=', sum(abs(ene-eks)), file=fout)
    
    def Naive_Projection_to_Existing_k_path(self, band_energy_file, strc, ks, pw, k2cartes, fout, EF=0):
        (klist2, wegh, ebnd2, hsrws, knames) = w2k.Read_energy_file(band_energy_file, strc, fout, give_kname=True)
        band_vector_file = re.sub('energy', 'vector', band_energy_file)
        (heads, all_Gs, all_As, all_Ek) = w2k.Read_vector_file2(band_vector_file, strc.nat, fout, vectortype=float)
        indgkir = Find_which_Gs_from_pw_are_present(pw, heads, all_Ek, all_Gs, True)
        
        krt = dot(self.rind, array(klist2).T)   # r*k
        smat = exp( 2*pi*1j*krt )               # smat[ir,ik] = exp(i*r*k)
        
        Hk = tensordot( smat, self.Hr, axes=(0,0))   # Hk[ik,:,:] = \sum_ir smat[ir,ik]*Hr[ir,:,:]
        
        s_eks = zeros((len(klist2), shape(Hk)[1]))
        print('k-points calculating naive projection', file=fout)
        for ik in range(len(klist2)):
            kl = klist2[ik]
            ene = linalg.eigvalsh(Hk[ik,:,:])
            s_eks[ik,:] = ene
            print('ik=%3d'%ik, 'k=(%5.3f,%5.3f,%5.3f)' % tuple(self.icartes2f@kl), file=fout)

        eb2 = zeros((len(klist2), self.nbe-self.nbs))
        eb2[:,:] = NaN
        for ik in range(len(klist2)):
            _nbe_ = min(self.nbe,len(ebnd2[ik]))
            for ib in range(self.nbs,_nbe_):
                #print('ik=', ik, 'ib=', ib, 'nbs=', nbs, 'nbe=', nbe, '_nbe_=', _nbe_, 'len(ebnd2[ik])=', len(ebnd2[ik]))
                eb2[ik,ib-self.nbs] = ebnd2[ik][ib]*Ry2H-EF
        
        SaveBandPlot('bands_naive_projection.dat', s_eks, klist2, k2cartes, knames)
        SaveBandPlot('bands_original.dat', eb2, klist2, k2cartes, knames)

    def Save_Eigenvalues(self, case, ks_Ebnd_isp, kqm):
        fo_eig = open(case+'.eig', 'w')
        for ik in range(len(kqm.klist)):
            if self.Irredk:
                irk = kqm.kii_ind[ik]
            else:
                irk = ik
            for i,ib in enumerate(range(self.nbs,self.nbe)):
                #print(i+1,ik+1, ks_Ebnd_isp[irk,ib]*H2eV, file=fo_eig)
                print(i+1,ik+1, ks_Ebnd_isp[irk,ib], file=fo_eig)
        fo_eig.close()
        

    def Compute_and_Save_BandOverlap_Mmn(self, pair, pair_umklap, case, orbs, strc, latgen, ks, kqm, pw, in1, radf, fout, DMFT1=False, naive_projection=True, debug=False):
        
        def FindInArray(ang, angles):
            for ii,a in enumerate(angles):
                if linalg.norm(ang-a) < 1e-6:
                    return ii
            return -1
        
        nspin,isp=1,0  # need to do further work for magnetic type calculation
        maxlp1 = min(in1.nt,7)
        # We are creating a product-like basis, which is serving the plane wave expansion e^{-i*b*r} = 4*pi*i^l * j_l(|b|*r) Y_{lm}^*(-b) Y_{lm}(r), here l is big_l, and |b| is distance
        big_l = array([repeat( range(maxlp1),  len(self.distances)) for iat in range(strc.nat)])    # index for l [0,0,0,1,1,1,....] each l is repeated as many times as there are distances.
        big_d = array([concatenate([ list(range(len(self.distances))) for i in range(maxlp1)]) for iat in range(strc.nat)]) # index for distances
        
        print('big_l for using GW projection <u_big_l|psi^*_{k-q}psi_k>', file=fout)
        for iat in range(strc.nat):
            for irm in range(len(big_l[iat])):
                print(irm, 'iat=', iat, 'l=', big_l[iat,irm], 'id=', big_d[iat,irm], file=fout)
        
        j_l_br = [[ [] for irm in range(len(big_l[iat]))] for iat in range(strc.nat)]
        for iat in range(strc.nat):
            rp, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
            for irm,L in enumerate(big_l[iat,:]):
                ab = self.distances[big_d[iat,irm]]        # this is |b|
                ur = special.spherical_jn(L,ab*rp)    # j_l(|b|r)
                j_l_br[iat][irm] = ur              # saving j_l(|b|r)
        
        lomaxp1 = shape(in1.nlo)[0]
        num_radial=[]
        for iat in range(strc.nat):
            tnum_radial = 2*in1.nt
            for l in range(lomaxp1):
                tnum_radial += len(in1.nLO_at_ind[iat][l])
            num_radial.append(tnum_radial)
        # Computing all necessary radial integrals <j_lb(br)|u_l2 u_l1>
        s3r = zeros( (len(big_l[0]),amax(num_radial),amax(num_radial),strc.nat), order='F' )
        for iat in range(strc.nat):
            rp, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
            rwf_all = []
            lrw_all = []
            # ul functions
            for l in range(in1.nt):
                rwf_all.append( (radf.ul[isp,iat,l,:npt],radf.us[isp,iat,l,:npt]) )
                lrw_all.append((l,1))
            # energy derivative ul_dot functions
            for l in range(in1.nt):
                rwf_all.append( (radf.udot[isp,iat,l,:npt],radf.usdot[isp,iat,l,:npt]) )
                lrw_all.append((l,2))
            # LO local orbitals
            for l in range(lomaxp1):
                for ilo in in1.nLO_at_ind[iat][l]:
                    rwf_all.append( (radf.ulo[isp,iat,l,ilo,:npt],radf.uslo[isp,iat,l,ilo,:npt]) )
                    lrw_all.append((l,3))
            # now performing the integration
            for ir2 in range(len(lrw_all)):
                l2, it2 = lrw_all[ir2]
                a2, b2 = rwf_all[ir2]
                for ir1 in range(ir2+1):
                    l1, it1 = lrw_all[ir1]
                    a1, b1 = rwf_all[ir1]
                    a3 = a1[:] * a2[:] #/ rp[:]
                    b3 = b1[:] * b2[:] #/ rp[:]
                    for irm in range(len(big_l[iat])):  # over all product functions
                        L = big_l[iat,irm]         # L of the product function
                        if L < abs(l1-l2) or L > (l1+l2): continue # triangular inequality violated
                        ## s3r == < u-product-basis_{irm}| u_{a1,l1} u_{a2,l2} > 
                        rint = rd.rint13g(strc.rel, j_l_br[iat][irm], j_l_br[iat][irm], a3, b3, dh, npt, strc.r0[iat])
                        #print('ir2=', ir2, 'ir1=', ir1, 'irm=', irm, 'int=', rint)
                        s3r[irm,ir1,ir2,iat] = rint
                        s3r[irm,ir2,ir1,iat] = rint
            if True:
                orb_info = [' core',' u_l ','u_dot',' lo  ']
                print((' '*5)+'Integrals <j_L(br)| u_(l1)*u_(l2)> for atom  %10s' % (strc.aname[iat],), file=fout)
                print((' '*13)+'N   L   l1  u_   l2 u_        <v | u*u>', file=fout)
                for irm in range(len(big_l[iat])):  # over all product functions
                    L = big_l[iat,irm]             # L of the product function
                    for ir2 in range(len(lrw_all)):
                        l2, it2 = lrw_all[ir2]
                        for ir1 in range(ir2+1):
                            l1, it1 = lrw_all[ir1]
                            if abs(s3r[irm,ir1,ir2,iat])>1e-10:
                                print(('%4d%4d'+(' '*2)+('%4d'*3)+' %s%4d %s%19.11e') % (ir1,ir2,irm,L,l1,orb_info[it1],l2,orb_info[it2],s3r[irm,ir1,ir2,iat]), file=fout)

        nmix = array([ maxlp1*len(self.distances) for iat in range(strc.nat)], dtype=int)
        loctmatsize  = sum([sum([(2*L+1)*strc.mult[iat] for L in big_l[iat]]) for iat in range(strc.nat)])
        ncore = zeros(strc.nat, dtype=int)  # we do not care about core here, so just set to zero
        cgcoef = gn.cmp_all_gaunt(in1.nt)   # gaunt coefficients
        
        cfac = []  # precompute 4*pi*(-i)^l, but we repeat the entry (2*lb+1) times, because of degeneracy with respect to m
        for lb in range(maxlp1):
            cfac += [4*pi*(-1j)**lb] * (2*lb+1)
        cfac = array(cfac)

        (nnkp, nntot) = shape(pair)
        
        # It turns out there are only a few possible angles Y_lm(b), for which Y_lm needs to be evaluates
        # We first find all possible angles of vector b.
        b_angles = []
        b_pair = zeros(shape(pair), dtype=int)
        for ik in range(len(kqm.klist)):
            for i in range(nntot):
                jk = pair[ik,i]
                k1 = kqm.klist[ik,:]/kqm.LCM
                k2 = kqm.klist[jk,:]/kqm.LCM
                b = k2-k1+ dot(kqm.k2icartes,pair_umklap[ik,i,:])
                ab = linalg.norm(b)
                if ab!=0:
                    b_ang = b/ab
                else:
                    b_ang = b
                ic = FindInArray( b_ang, b_angles)
                if ic<0:
                    b_pair[ik,i] = len(b_angles)
                    b_angles.append(b_ang)
                else:
                    b_pair[ik,i] = ic
        
        print('Found the following angles for b=k2-k1 vector:', file=fout)
        for i in range(len(b_angles)):
            print(i, b_angles[i], file=fout)
        
        idf = -1
        imm = -1
        ndf = sum(strc.mult)
        Ylmb = zeros((ndf,len(b_angles), maxlp1**2), dtype=complex)
        for iat in range(strc.nat):
            for ieq in range(strc.mult[iat]):
                idf += 1
                Ttot = strc.rotloc[iat,:,:] @ latgen.trotij[idf,:,:].T @ kqm.k2cartes[:,:] # All transformations that we need to evaluate Y_lm(b)
                for i,b in enumerate(b_angles):
                    Ylmb[idf,i,:] = sphbes.ylm( dot(Ttot, -b), maxlp1-1)*cfac  # 4*pi(-i)^l * Y_lm(-b), which is e^{-i*b*r}/Y^*_{lm}(r)
                    
        mpwipw = zeros((1,len(pw.ipwint)),dtype=complex) # will contain Integrate[e^{iG*r},{r in interstitials}]
        mpwipw[0,:] = pw.ipwint # mpwipw[0,ipw] = <0|G_{ipw}>_{int}=Integrate[e^{i*(G_{ipw}r)},{r in interstitial}]


        fo_mmn = open(case+'.mmn', 'w')
        print('# code PyGW:wannier.py', date.today().strftime("%B %d, %Y"), file=fo_mmn)
        print(self.nbe-self.nbs, len(kqm.klist), nntot, file=fo_mmn)
        
        V_rmt = 0
        for iat in range(strc.nat):
            for ieq in range(strc.mult[iat]):
                V_rmt += 4*pi*strc.rmt[iat]**3/3
        print('V_rmt/V=', V_rmt/latgen.Vol, 'V_I/V=', 1-V_rmt/latgen.Vol, 'V_rmt=', V_rmt, 'V=', latgen.Vol, file=fout)
        
        idf = -1
        for ik in range(len(kqm.klist)):
            kil = kqm.klist[ik,:]/kqm.LCM  # k1==kil in semi-cartesian form
            irk = kqm.kii_ind[ik]          # the index to the correspodning irreducible k-point
            if self.Irredk:
                # First create alpha, beta, gamma for k
                Aeigk = array(ks.all_As[irk][self.nbs:self.nbe,:], dtype=complex)   # eigenvector from vector file
                if not DMFT1:
                    if kqm.k_ind[irk] != ik:                       # not irreducible
                        Aeigk *= exp( 2*pi*1j * ks.phase_arg[ik][:])  # adding phase: e^{i*tau[isym]*(k_irr+G_irr)}, where tau[isym] is the part of the group operation from irreducible to reducible k-point
                    alm,blm,clm = lapwc.gap2_set_lapwcoef(kil, ks.indgk[ik], 1, True, ks.nv[irk], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
                else:
                    isym = kqm.iksym[ik]
                    timat_ik, tau_ik = strc.timat[isym,:,:].T, strc.tau[isym,:]
                    kirr = array(kqm.kirlist[irk,:])/float(kqm.LCM)
                    alm,blm,clm = lapwc.dmft1_set_lapwcoef(False, 1, True, kil, kirr, timat_ik, tau_ik, ks.indgkir[irk], ks.nv[irk], pw.gindex, radf.abcelo[0], strc.rmt, strc.vpos, strc.mult, radf.umt[0], strc.rotloc, latgen.rotij, latgen.tauij, latgen.Vol,  kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
            else:
                Aeigk = array(ks.all_As[ik][self.nbs:self.nbe,:], dtype=complex)   # eigenvector from vector file
                alm,blm,clm = lapwc.gap2_set_lapwcoef(kil, ks.indgkir[ik], 1, True, ks.nv[ik], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
                
            (ngi,nLOmax,ntnt,ndf) = shape(clm)
            (ngi,ntnt,ndf) = shape(alm)
            (nbmax,ngi2) = shape(Aeigk)
            # And now change alm,blm,clm to band basis, which we call alfa,beta,gama
            alfa = reshape( la_matmul(Aeigk, reshape(alm, (ngi,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
            beta = reshape( la_matmul(Aeigk, reshape(blm, (ngi,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
            if in1.nlomax > 0:
                gama = reshape( la_matmul(Aeigk, reshape(clm, (ngi,ntnt*ndf*nLOmax)) ), (nbmax,nLOmax,ntnt,ndf) )
            else:
                gama = zeros((1,1,1,1),dtype=complex,order='F') # can not have zero size. 

            
            if DMFT1 and kqm.k_ind[irk] != ik:                       # not irreducible
                # For interstitials we need to transform now, because we did not transform the eigenvectors before.
                Aeigk *= exp( 2*pi*1j * ks.phase_arg[ik][:])  # phase for reducible: e^{i*tau[isym]*(k_irr+G_irr)}, where tau[isym] is the part of the group operation from irreducible to reducible k-point
            
            if debug:
                print('alfa,beta,gama=', file=fout)
                for ie in range(shape(alfa)[0]):
                    for lm in range(shape(alfa)[1]):
                        print('ie=%3d lm=%3d alfa=%14.10f%14.10f beta=%14.10f%14.10f' % (ie+1,lm+1,alfa[ie,lm,0].real, alfa[ie,lm,0].imag, beta[ie,lm,0].real, beta[ie,lm,0].imag), file=fout)
            
            for idk in range(nntot):
                jk = pair[ik,idk]
                # M_{m,n}   = < psi_{m,k} | e^{-i b r} |psi_{n,k+b}>
                # M_{m,n}^* = < psi_{n,k+b}| e^{i b r} |psi_{m,k}>
                # M_{m,n}^* = < psi_{n,k-q}| e^{-i q r}|psi_{m,k}> where q=-b. The last form is equivalent to our definition of M matrix element in GW.
                # k2 = k1 + b - G_umklap => b = k2-k1+G_umklapp
                kjl = kqm.klist[jk,:]/kqm.LCM  # k2==kjl=k+b in semi-cartesian form
                G_umklap = array(dot(kqm.k2icartes,pair_umklap[ik,idk,:]), dtype=int)
                bl = kjl-kil + G_umklap
                aq = linalg.norm(b)
                jrk = kqm.kii_ind[jk]
                
                t0 = timer()
                if self.Irredk:
                    # And next create alpha, beta, gamma for k+q
                    Aeigq = array( conj( ks.all_As[jrk][self.nbs:self.nbe,:] ), dtype=complex)  # eigenvector from vector file
                    if not DMFT1:
                        if kqm.k_ind[jrk] != jk:                            # the k-q-point is reducible, eigenvector needs additional phase
                            Aeigq *= exp( -2*pi*1j * ks.phase_arg[jk][:] )  # adding phase e^{-i*tau[isym]*(k_irr+G_irr)}
                        alm,blm,clm = lapwc.gap2_set_lapwcoef(kjl, ks.indgk[jk], 2, True, ks.nv[jrk], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
                    else:
                        kjrr = array(kqm.kirlist[jrk,:])/float(kqm.LCM) 
                        isym = kqm.iksym[jk]
                        timat_ik, tau_ik = strc.timat[isym].T, strc.tau[isym,:]
                        alm,blm,clm = lapwc.dmft1_set_lapwcoef(False, 2, True, kjl, kjrr, timat_ik, tau_ik, ks.indgkir[jrk], ks.nv[jrk], pw.gindex, radf.abcelo[0], strc.rmt, strc.vpos, strc.mult, radf.umt[0], strc.rotloc, latgen.rotij, latgen.tauij, latgen.Vol,  kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
                else:
                    Aeigq = array( conj( ks.all_As[jk][self.nbs:self.nbe,:] ), dtype=complex)  # eigenvector from vector file
                    alm,blm,clm = lapwc.gap2_set_lapwcoef(kjl, ks.indgkir[jk], 2, True, ks.nv[jk], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
                
                (ngj,nLOmax,ntnt,ndf) = shape(clm)
                (ngj,ntnt,ndf) = shape(alm)
                (nbmax,ngj2) = shape(Aeigq)

                # And now change alm,blm,clm to band basis, which we call alfa,beta,gama
                alfp = reshape( la_matmul(Aeigq, reshape(alm, (ngj,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
                betp = reshape( la_matmul(Aeigq, reshape(blm, (ngj,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
                if in1.nlomax > 0:
                    gamp = reshape( la_matmul(Aeigq, reshape(clm, (ngj,ntnt*ndf*nLOmax)) ), (nbmax,nLOmax,ntnt,ndf) )
                else:
                    gamp = zeros((1,1,1,1),dtype=complex,order='F') # can not have zero size
                
                if debug:
                    print('alfp,betp,gamp=', file=fout)
                    for ie in range(shape(alfp)[0]):
                        for lm in range(shape(alfp)[1]):
                            print('ie=%3d lm=%3d alfp=%14.10f%14.10f betp=%14.10f%14.10f' % (ie+1,lm+1,alfp[ie,lm,0].real, alfp[ie,lm,0].imag, betp[ie,lm,0].real, betp[ie,lm,0].imag), file=fout)
                            
                t1 = timer()
                #t_times[0] += t1-t0
                
                ### The muffin-tin part : mmat[ie1,ie2,im] = < u^{product}_{im,lb} | psi^*_{ie2,k-q} psi_{ie1,k} > e^{-iq.r_atom} where ie1=[nst,nend] and ie2=[mst,mend]
                # mmat_mt[ie1,ie2,im] = <j_l(|b|r)|psi^*_{ie2,k+b} psi_{ie1,k} > e^{i*b.r_atom}
                big_L = big_l.T
                mmat_mt = fvxcn.calc_minm_mt(-bl,0,nbmax,0,nbmax, alfa,beta,gama,alfp,betp,gamp, s3r, strc.vpos,strc.mult, nmix,big_L, in1.nLO_at,ncore,cgcoef,in1.lmax,loctmatsize)
                
                idf = -1
                imm=0
                psi_psi = zeros((nbmax,nbmax), dtype=complex)
                for iat in range(strc.nat):
                  for ieq in range(strc.mult[iat]):
                      idf += 1
                      for irm in range(nmix[iat]):
                          lb = big_l[iat,irm] #
                          dimm = 2*lb+1       # lme-lms
                          if big_d[iat,irm]==self.idistance[ik,idk]:
                              lms, lme = lb**2, (lb+1)**2
                              # Now adding 4*pi(-i)^l * Y_lm(-b) so that we have : psi_psi = 4*pi(-i)^l * Y_lm(-b) <j_l(|b|r)Y_lm(r)|psi^*_{ie2,k+b} psi_{ie1,k} > e^{i*b.r_atom}
                              # Note that 4*pi*(-i)^l Y_lm(-b) Y_lm(r)^* = 4*pi*i^l*Y_lm(b) * Y^*_lm(r) = e^{i*b*r}
                              # which is psi_psi = < e^{-i*b*(r+r_atom)} | psi^*_{ie2,k+b} psi_{ie1,k} >
                              psi_psi += dot(mmat_mt[:,:,imm:imm+dimm], Ylmb[idf,idk,lms:lme])
                          imm += dimm
                
                if DMFT1 and kqm.k_ind[jrk] != jk:                             # the k-q-point is reducible, eigenvector needs additional phase
                    Aeigq *= exp( -2*pi*1j * ks.phase_arg[jk][:] )
                          
                ## The interstitial part : mmat[ie1,ie2,im]*sqrt(Vol) = < 1 |psi^*_{ie2} psi_{ie1} > = < e^{i*(G_{im}=0)*r}|psi_{ie2,k+q}^* |psi_{ie1,k}>_{Interstitials}
                ## mmat[ie1,ie2] = Aeigq[ie2,nvj]^*  mpwipw(nvj,nvi)  Aeigk.T(nvi,ie1)
                i_g0 = pw.convert_ig0_2_array() # Instead of dictionary pw.ig0, which converts integer-G_vector into index in fixed basis, we will use fortran compatible integer array
                indggq = array(range(len(pw.gindex)),dtype=int)
                if self.Irredk:
                    nvi, nvj = ks.nv[irk], ks.nv[jrk]
                    mmat_it = fvxcn.calc_minm_is(0,nbmax,0,nbmax,Aeigk,Aeigq,G_umklap,mpwipw,nvi,nvj,ks.indgk[ik],ks.indgk[jk],indggq,pw.gindex,i_g0,latgen.Vol)
                else:
                    nvi, nvj = ks.nv[ik], ks.nv[jk]
                    mmat_it = fvxcn.calc_minm_is(0,nbmax,0,nbmax,Aeigk,Aeigq,G_umklap,mpwipw,nvi,nvj,ks.indgkir[ik],ks.indgkir[jk],indggq,pw.gindex,i_g0,latgen.Vol)
                ## Combine MT and interstitial part together
                psi_psi1 = mmat_it[:,:,0]*sqrt(latgen.Vol)
                psi_psi3 = psi_psi + psi_psi1
                
                M_m_n = psi_psi3.conj()
                print(ik+1,jk+1, '%4d'*3 % tuple(pair_umklap[ik,idk,:]), file=fo_mmn)
                for n in range(len(M_m_n)):
                    for m in range(len(M_m_n)):
                        print('%18.14f %18.14f' % (M_m_n[m,n].real, M_m_n[m,n].imag), file=fo_mmn)

                print('ik=', ik, 'idk=', idk, 'jk=', jk, 'irk=', irk, 'iik=', kqm.k_ind[irk],'!=', ik, 'jjk=', kqm.k_ind[jrk],'!=', jk, 'b=', bl, 'G_umklap=', G_umklap, '<psi|psi>=', file=fout)
                PrintMC(psi_psi3,file=fout)
                if debug:
                    print(file=fout)
                    PrintMC(psi_psi,file=fout)
                    print(file=fout)
                    PrintMC(psi_psi1,file=fout)


    def ReadHamiltonianAfter(self):
        fi = open(self.case+'_hr.dat', 'r')
        lines = fi.readlines()
        num_wann = int(lines[1])
        nrpts = int(lines[2])
        degWs=[]
        ipos = 3+int(nrpts/15)+1
        for iline in range(3,ipos):
            degWs += [int(n) for n in lines[iline].split()]
        Ham = zeros((nrpts,num_wann,num_wann), dtype=complex)
        Rs = zeros((nrpts,3),dtype=int)
        for i in range(nrpts):
            for n in range(num_wann):
                for m in range(num_wann):
                    dat = lines[ipos].split()
                    R = [int(dat[i]) for i in range(3)]
                    #print('R=', R)
                    mi,ni = int(dat[3]), int(dat[4])
                    t = float(dat[5])+float(dat[6])*1j
                    if n==m==0:
                        Rs[i,:] = R
                    Ham[i,m,n] = t
                    if (ni!=n+1 or mi!=m+1):
                        print('ERROR in reading Hamiltonian '+self.case+'_hr.dat')
                        sys.exit(0)
                    #print(i, n, m, lines[ipos], end='')
                    ipos += 1
        return(Rs,Ham,degWs)

    def ReadCentersAfter(self):
        fi = open(self.case+'_centres.xyz', 'r')
        lines = fi.readlines()
        nc = int(lines[0])
        wRc = zeros((nc,3))
        for i in range(2,2+nc):
            xyz = [float(xyz) for xyz in lines[i].split()[1:4]]
            wRc[i-2,:] = xyz
            #print(i-2, xyz)
        return wRc
    
    def ReadBandsAfter(self, case):
        # check how many bands we have
        fi = open(case+'_band.dat', 'r')
        Nb = Nempty = sum(list(line.isspace() for line in fi))
        fi.close()
        
        klines = open(case+'_band.labelinfo.dat','r').readlines()
        kind=[]
        for line in klines:
            w = line.split()
            name, ik = w[0], int(w[1])
            kind.append([ik-1,name])
        
        dinp = loadtxt(case+'_band.dat')
        Npp = int(shape(dinp)[0]/Nb)
        dinp = reshape(dinp, (Nb,Npp,2))
        
        x_k = dinp[0,:,0]
        Ene_k = dinp[:,:,1].T
        return(x_k, Ene_k, kind)
    
    def PlotBands(self, x_k, Ene_k, kind):
        for ib in range(shape(Ene_k)[1]):
            plot(x_k, Ene_k[:,ib])
        
        y0, y1 = ylim()
        for ik,name in kind:
            plot([x_k[ik], x_k[ik]], [y0,y1], 'k:', lw=0.3)
            
        xticks([x_k[ik] for ik,name in kind], labels=[name for ik,name in kind])
        ylim([y0,y1])
        xlim([x_k[0], x_k[-1]])
        show()
        
def PlotBands(filename):
    fi = open(filename, 'r')
    data = fi.readline()
    fi.close()
    #leg = eval(data[1:].strip())
    s = data[1:].strip()
    s=re.sub('leg=', '', s)
    leg=eval(s)
    #print('leg=', leg)
    data = loadtxt(filename).T
    x_k = data[0]
    for ib in range(1,len(data)):
        plot(data[0], data[ib])
    y0, y1 = ylim()
    for ik,name in leg.items():
        plot([x_k[ik], x_k[ik]], [y0,y1], 'k:', lw=0.3)
    xticks([x_k[ik] for ik,name in leg.items()], labels=[name for ik,name in leg.items()])
    ylim([y0,y1])
    xlim([x_k[0], x_k[-1]])
    show()

def PrintBands(filename, x_k, e_k, kind):
    def PrintLegend(kind):
        leg = '{'
        for ik,kname in kind:
            leg += str(ik)+':"'+kname+'",'
        leg += '}'
        return leg
    
    fgd = open(filename, 'w')
    (nk,nb) = shape(e_k)
    print('# leg='+PrintLegend(kind), file=fgd)
    for ik in range(nk):
        print('%10.6f  ' % (x_k[ik],), ('%14.8f '*nb) % tuple(e_k[ik,:]), file=fgd)
    fgd.close()
    
def SaveBandPlot(filename, bands, klist2, k2cartes, knames):
    def PrintLegend(knames):
        leg = '{'
        for ik in range(len(knames)):
            name = knames[ik].strip()
            if name:
                leg += str(ik)+':"'+name+'",'
        leg += '}'
        return leg
    
    nk, nb = shape(bands)
    fgw = open(filename, 'w')
    # k-path distance
    xc = Path_distance( array(klist2), k2cartes )
    print('# leg=' + PrintLegend(knames), file=fgw)
    for ik in range(len(klist2)):
        print('%10.6f  ' % (xc[ik],), ('%14.8f '*nb) % tuple(bands[ik,:]*H2eV), file=fgw)
    fgw.close()

    
    
def Wannierize(nbs,nbe, k_path, semi_cartesian, cmpEF=True, band_energy_file=None):
    
    w90_exe = shutil.which('wannier90.x')
    if w90_exe is None:
        print('ERROR: Expecting wannier90.x in path, but could not find it')
        sys.exit(0)
    io = InOut("gw.inp", "pypp.out", mrank==master)
    case = io.case
    fout = io.out
    strc = w2k.Struct(case, fout)
    latgen = w2k.Latgen(strc, fout)
    latgen.Symoper(strc, fout)
    
    
    kqm = KQmesh(io.nkdivs, io.k0shift, strc, latgen, fout)
    ks = KohnShamSystem(io.case, strc, io.nspin, fout)
    in1 = w2k.In1File(case, strc, fout)#, io.lomax)
    ks.Set_nv(in1.nlo_tot)

    pw = PlaneWaves(ks.hsrws, io.kmr, io.pwm, case, strc, in1, latgen, kqm, False, fout)
    ks.VectorFileRead(io.case, strc, latgen, kqm, pw, fout, in1)
    (Elapw, Elo) = w2k.get_linearization_energies(io.case, in1, strc, io.nspin, fout)
    in1.Add_Linearization_Energy(Elapw, Elo)
    Vr = w2k.Read_Radial_Potential(io.case, strc.nat, io.nspin, strc.nrpt, fout)
    radf = w2k.RadialFunctions(in1,strc,ks.Elapw,ks.Elo,Vr,io.nspin,fout)
    del Vr
    radf.get_ABC(in1, strc, fout)

    EF=0
    if cmpEF:
        kqm.tetra(latgen, strc, fout)
        wcore = w2k.CoreStates(io.case, strc, io.nspin, fout)
        (EF, Eg, evbm, ecbm, eDos) = mcmn.calc_Fermi(ks.Ebnd[0], kqm.atet, kqm.wtet, wcore.nval, ks.nspin)
        ks.Ebnd -= EF
        if Eg >= 0:
            print('\n'+'-'*32+'\nFermi: Insulating, KS E_Fermi[eV]=%-12.6f Gap[eV]=%-12.6f  EVBM[eV]=%-12.6f  ECBM[eV]=%-12.6f' % (EF*H2eV, Eg*H2eV, evbm*H2eV, ecbm*H2eV), file=fout)
        else:
            print('\n'+'-'*32+'\nFermi: Metallic, KS E_Fermi[eV]=%-12.6f  DOS[E_f]=%-12.6f' % (EF*H2eV, eDos), file=fout)
        
    w90 = CWannier90(nbs,nbe,io.case, strc.mult, strc.aname, kqm.ndiv, kqm.klist, kqm.LCM, kqm.k2icartes)
    orbs = w90.FindRelevantOrbitals(strc, latgen, ks, kqm, pw, in1, radf, fout)
    w90.Prepare_win_file(orbs, latgen.rbas, strc.vpos, strc.mult, strc.aname, k_path, semi_cartesian)

    # execute wannier90.x -pp case.win
    process = subprocess.Popen([w90_exe, '-pp', io.case+'.win'], stdout=fout)
    process.wait()
    
    w90.Compute_and_Save_Projection_amn(io.case, orbs, strc, latgen, ks, kqm, pw, in1, radf, fout, naive_projection=True)
    
    if band_energy_file:
        w90.Naive_Projection_to_Existing_k_path(band_energy_file, strc, ks, pw, kqm.k2cartes, fout, EF)

    w90.Save_Eigenvalues(io.case, ks.Ebnd[0], kqm)
    
    (pair, pair_umklap) = w90.Read_nnkp()
    w90.Find_All_Possible_kpair_distances(pair, pair_umklap)
    w90.Compute_and_Save_BandOverlap_Mmn(pair, pair_umklap, io.case, orbs, strc, latgen, ks, kqm, pw, in1, radf, fout, DMFT1=False, naive_projection=True, debug=False)
    
    process = subprocess.Popen([w90_exe, io.case+'.win'], stdout=fout, stderr=subprocess.PIPE, universal_newlines=True)
    process.wait()
    output, errors = process.communicate()
    if errors: print('ERROR:', errors)
    
    (x_k, Ene_k, kind) = w90.ReadBandsAfter(io.case)
    w90.PlotBands(x_k, Ene_k, kind, k_path)

    
if __name__ == '__main__':
    
    band_energy_file = 'Na_bcc.energy_band'
    #band_EFermi = -0.0004467687
    nbs, nbe = 4,20
    semi_cartesian=True
    k_path=[['G', (0,0,0)],['N', (0,0.5,0.5)],['H', (0,0,1)],['G', (0,0,0)],['P',(-0.5,0.5,0.5)]]
    
    #Wannierize(nbs,nbe, k_path, semi_cartesian, cmpEF=True, band_energy_file=band_energy_file)
    
    
    band_energy_file = 'Si.energy_band'
    #band_EFermi = 0.3828824575
    nbs, nbe = 0,8
    ## fractional
    ##semi_cartesian=False
    ##k_path=[ ['W',(0.25,0.5,0.75)], ['L',(0.5,0.5,0.5)], ['La',(0.25,0.25,0.25)], ['G',(0,0,0)], ['D',(0,0.25,0.25)], ['X',(0.0,0.5,0.5)], ['Z',(0.125,0.5,0.625)], ['W',(0.25,0.5,0.75)],['K',(0.375,0.375,0.75)]]
    ##semi_cartesian
    semi_cartesian=True
    k_path=[ ['W', (1.0,0.5,0.0)],  ['L',(0.5,0.5,0.5)], ['La',(0.25,0.25,0.25)], ['G',(0,0,0)], ['D',(0.5,0.0,0.0)], ['X',(1.0,0.0,0.0)], ['Z',(1.0,0.25,0.0)],    ['W',(1.0,0.5,0.0)], ['K',(0.75,0.75,0)]]


    
    #Wannierize(nbs,nbe, k_path, semi_cartesian, cmpEF=True, band_energy_file=band_energy_file)

    nbs,nbe=0,15
    io = InOut("gw.inp", "debug.out", True)
    case = io.case
    fout = io.out
    strc = w2k.Struct(case, fout)
    latgen = w2k.Latgen(strc, fout)
    latgen.Symoper(strc, fout)
    kqm = KQmesh(io.nkdivs, io.k0shift, strc, latgen, fout)    
    
    w90 = CWannier90(nbs,nbe,io.case, strc.mult, strc.aname, kqm.ndiv, kqm.klist, kqm.LCM, kqm.k2icartes)
    (Rs,Ham,degWs) = w90.ReadHamiltonianAfter()
    wRc = w90.ReadCentersAfter()
    
    #tuple(latgen_rbas[:,j]),
    (nrpts,nw1,nw ) = shape(Ham)
    

    ks=[]
    for ik in range(len(k_path)-1):
        k1, k2 = k_path[ik][1], k_path[ik+1][1]
        if semi_cartesian:
            k1 = dot(w90.icartes2f,array(k1))
            k2 = dot(w90.icartes2f,array(k2))
        x = linspace(0,1,30)
        for i in range(len(x)):
            ks.append(k1 + (k2-k1)*x[i])
    ks = array(ks)
    print('ks=', ks)

    phase = zeros((len(ks),len(Rs)), dtype=complex)
    for ir,R in enumerate(Rs):
        kR = dot(ks,R)
        phase[:,ir] = exp(2*pi*1j*kR) #*degWs[ir]

    bands = zeros((nw,len(ks)))
    for ik in range(len(ks)):
        Hk = zeros((nw,nw),dtype=complex)
        for ir,R in enumerate(Rs):
            Hk += phase[ik,ir]*Ham[ir,:,:]
        ene = linalg.eigvalsh(Hk)
        bands[:,ik] = ene
    for i in range(nw):
        plot(bands[i,:])
    show()
