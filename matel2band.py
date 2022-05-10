#from scipy import *
from numpy import array
from scipy import special
from scipy import linalg
from timeit import default_timer as timer
import itertools
from pylab import linalg as la
la_matmul = la.matmul

from cmn import *
import mcommon as mcmn

import for_vxcnn as fvxcn   # matrix elements for Vxcn
import lapwcoeff as lapwc   # computes lapw coefficients alm,blm,clm
import for_kpts as fkp
import for_Coulomb as fCl
import radials as rd        # radial wave functions
import for_q_0 as f_q_0

DMFT1_like = False      # This forces the reducible k-points to be handled like in dmft1 code. Somehow, this has difficulty for more than one atom per unit cell.
Real_sqrt_olap = True   # This is a good way of ensuring that basis in the interstitials is as close to plane-waves as possible, but orthogonal
Real_sqrt_Vcoul = False # This similarly ensures that sqrt(V_c) is as close as possible to V_c in terms of phase. But it can make the matrix slightly bigger.
# The Coulomb interaction <psi_i|V_C|psi_j>==V_ij is a positive definite matrix, which has eigensystem V_ij == Vmat*lmbda*Vmat^H
# We need sqrt(V), which can have the form sqrt(V) == Vmat*sqrt(lmbda), or, it can have the form sqrt(V) == Vmat*sqrt(lmbda)*Vmat^H. Both matrices produce sqrt(V)*sqrt(V)^H = V.
# The matrix  Vmat*sqrt(lmbda) can be made smaller, because a few lmbdas are very small. This is done when Real_sqrt_Vcoul=False.
# If Real_sqrt_Vcoul=True, we use larger matrix sqrt(V) == Vmat*sqrt(lmbda)*Vmat^H
ForceReal = False       # This forces Coulomb interaction to be truncated to real part only, and is not exact


class MatrixElements2Band:
    DEBUG = False
    def __init__(self, io, pw, strc, latgen):
        # Instead of dictionary pw.ig0, which converts integer-G_vector into index in fixed basis, we will use fortran compatible integer array
        self.i_g0 = pw.convert_ig0_2_array() # Instead of dictionary pw.ig0, which converts integer-G_vector into index in fixed basis, we will use fortran compatible integer array
        
        alat = array([strc.a, strc.b, strc.c])
        self.calc_tildeg(2*(io.lmbmax+1))
        self.eta = self.optimal_eta(latgen.rbas, latgen.br2)
        self.r_cf = 2 * self.r_cutoff(io.stctol, self.eta, 10, alat)
        self.g_cf = 2 * self.g_cutoff(io.stctol, self.eta, 10, latgen.pia, latgen.Vol)

    
    def Vxc(self, strc, in1, latgen, kqm, ks, radf, pw, pb, fout):
        t_lapwcoef, t_lapack, t_muffint, t_interst = 0, 0, 0, 0
        print(' calc vxcnn', file=fout)
        if FORT:
            #  interstitial integral Integrate[ (iK - iG)\vr, {\vr in interstitial}] where K is averaged over all members of the star
            #  Here K is read from V_{xc} and G is from the fixed basis with length smaller than pw.npw2apw
            istpw = fvxcn.intstipw(ks.ksxc, pw.gindex, strc.timat, strc.tau, strc.vpos, pw.vmt, strc.rmt, strc.mult, kqm.k2cartes, pw.npw2apw)
        else:
            #  We will compute :
            #  iK_sym[isym,j,ik] = dot(ks.ksxc[ik,:], timat[isym][:,j]) = dot(timat[isym].T[j,:], ks.ksxc.T[:,ik]) 
            imat  = zeros((strc.Nsym,3,3),dtype=int)
            for isym in range(strc.Nsym):
                imat[isym,:,:] = strc.timat[isym,:,:].T
            imat = reshape(imat,(strc.Nsym*3,3))
            iK_symat = reshape( la_matmul(imat, ks.ksxc.T), (strc.Nsym,3,len(ks.ksxc)) ) # iK_symat[isym,:,ik]
            iK_tsymat = zeros( (strc.Nsym,len(ks.ksxc),3), dtype=int )                       # iK_tsymat[isym,ik,3]
            for isym in range(strc.Nsym):
                iK_tsymat[isym,:,:] = iK_symat[isym,:,:].T
            #
            # phasemat[ik,isym] = dot(ks.ksxc[ik,:],strc.tau[isym,:])*2*pi  = dot(ks.ksxc[ik,:],strc.tau.T[:,isym])*2*pi 
            phasemat = la_matmul(ks.ksxc, strc.tau.T)*(2*pi*1j)
            phimat = exp(-phasemat)       # phasemat[ik,isym], phimat[ik,isym]
            #
            istpw = zeros( (pw.npw2apw, len(ks.ksxc) ), dtype=complex, order='F')
            for ik in range(len(ks.ksxc)): # over all G vectors in Vxc
                for ig in range(pw.npw2apw): # over all G vectors we are interested in
                    istpw[ig,ik] = sum([fvxcn.int1ipw(iK_tsymat[isym,ik,:]-pw.gindex[ig,:], kqm.k2cartes, pw.vmt, strc.rmt, strc.vpos, strc.mult) * phimat[ik,isym] for isym in range(strc.Nsym)])/float(strc.Nsym)
                #to_print = istpw[ik,:10].real
                #print '%3d' % (ik,), '%12.7f'*10 % tuple(to_print)
            #for ik in range(len(ks.ksxc)):
            #    print '%3d' % (ik,), '%12.7f'*10 % tuple(istpw[:10,ik].real)

        #i_g0 = pw.convert_ig0_2_array()
        #nbmax = min([shape(ks.all_As[irk])[0] for irk in range(len(ks.all_As))])
        nbmax = ks.nbgw  # the band cutoff for this calculation
        Vxct = zeros((len(kqm.kirlist),nbmax,nbmax),dtype=complex)
        isp=0
        for irk in range(len(kqm.kirlist)):
            kl = array(kqm.kirlist[irk,:])/float(kqm.LCM)  # k in semi-cartesian form
            ik = kqm.k_ind[irk]   # index in all-kpoints, not irreducible
            tm10 = timer()
            # Get alm,blm,clm coefficients
            if DMFT1_like:
                isym = kqm.iksym[ik]
                timat_ik, tau_ik = strc.timat[isym].T, strc.tau[isym,:]
                alm,blm,clm = lapwc.dmft1_set_lapwcoef(False,1, True, kl, kl, timat_ik,tau_ik, ks.indgkir[irk], ks.nv[irk], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.rotij, latgen.tauij, latgen.Vol,  kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
            else:    
                alm,blm,clm = lapwc.gap2_set_lapwcoef(kl, ks.indgk[ik], 1, True, ks.nv[irk], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
            tm11 = timer()
            
            (ngk,ntnt,ndf) = shape(alm)
            (ngk,nLOmax,ntnt,ndf) = shape(clm)
            # eigen-vector
            Aeig = array(ks.all_As[irk][:nbmax,:],dtype=complex)
            if False:
                print('alm,blm')
                for idf in range(ndf):
                    for ig in range(ngk):
                        for ilm in range(ntnt):
                            print('irk=%2d idf=%2d i=%3d j=%3d alm=%14.9f%14.9f blm=%14.9f%14.9f' % (irk+1, idf+1, ig+1, ilm+1, alm[ig,ilm,idf].real, alm[ig,ilm,idf].imag, blm[ig,ilm,idf].real, blm[ig,ilm,idf].imag))
                print('eigenvector', shape(ks.all_As[irk]))
                for ib in range(nbmax):
                    for j in range(ngk):
                        print('irk=%2d ib=%3d j=%3d A=%14.9f%14.9f' % (irk+1,ib+1,j+1,Aeig[ib,j].real,Aeig[ib,j].imag))
            
            # And now change alm,blm,clm to band basis, which we call alfa,beta,gama
            alfa = reshape( la_matmul(Aeig, reshape(alm, (ngk,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
            beta = reshape( la_matmul(Aeig, reshape(blm, (ngk,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
            if in1.nlomax > 0:
                gama = reshape( la_matmul(Aeig, reshape(clm, (ngk,ntnt*ndf*nLOmax)) ), (nbmax,nLOmax,ntnt,ndf) )
            else:
                gama = zeros((nbmax,nLOmax,ntnt,ndf),dtype=complex,order='F')
            
            tm12 = timer()
            t_lapwcoef += tm11-tm10
            t_lapack   += tm12-tm11
            
            # The muffin-thin part of <psi_{k}|V_{xc}|psi_{k}>
            Vxcmt = fvxcn.mt_vxcnn(irk,ks.uxcu,alfa,beta,gama,pb.cgcoef,ks.lmxc_f,ks.lxcm_f,ks.ibgw,ks.nbgw,strc.mult,strc.iatnr,in1.nLO_at,in1.lmax,in1.nt)
            tm13 = timer()
            
            if self.DEBUG:
                print('alfabeta')
                for idf in range(ndf):
                    for l in range(in1.lmax):
                        for m in range(-l,l+1):
                            lm = l*l + l + m 
                            for ie in range(nbmax):
                                print(('%s%2d '+'%s%3d '*2+ '%s%14.9f%14.9f '*3) % ('irk=',irk+1,'lm=',lm+1,'ie=',ie+1, 'alf=', alfa[ie,lm,idf].real, alfa[ie,lm,idf].imag, 'bet=', beta[ie,lm,idf].real, beta[ie,lm,idf].imag, 'gam=', gama[ie,0,lm,idf].real, gama[ie,0,lm,idf].imag))
            
            # This part calculates index array jimp[iv1,iv2] which finds index of two reciprocal vectors G1-G2
            # in fixed basis (pw.gindex), where  G1 and G2 are reciprocal vectors from Hamiltonian
            # corresponding to a particular irreducible k-point.
            nvk = ks.nv[irk]
            if FORT:
                jipw = fvxcn.two_gs_to_one(pw.gindex,ks.indgk[ik][:nvk],self.i_g0)
            else:
                jipw = -ones((nvk,nvk),dtype=int)  # jipw is index for difference between two G points from vector file
                for i,(iv1,iv2) in enumerate(itertools.product(list(range(nvk)),list(range(nvk)))):
                    iGc1 = pw.gindex[ks.indgk[ik][iv1]]
                    iGc2 = pw.gindex[ks.indgk[ik][iv2]]
                    diG = tuple(iGc1-iGc2)
                    if diG in pw.ig0:
                        jipw[iv1,iv2] = pw.ig0[diG]
                
            # Now we calculate interstitail integral for the difference G1-G2 (with G1 and G2 from Hamiltonian basis)
            # and K from V_{xc}:
            #       inti[1,2] = Integrate[ (iK - i(G1-G2))\vr, {\vr in interstitial}]  = <G1|e^{iK\vr}|G2>
            Aeigc = Aeig[:,:nvk] # this is the eigenvector withouth local orbitals.
            # interstitial exchange-correlation
            Vxci = zeros((nbmax,nbmax),dtype=complex)
            for ikxc in range(len(ks.ksxc)):
                if FORT:
                    inti = fvxcn.get_inti(jipw, istpw[:,ikxc])
                else:
                    inti = zeros((nvk,nvk),dtype=complex, order='F')
                    for i,(iv1,iv2) in enumerate(itertools.product(list(range(nvk)),list(range(nvk)))):
                        if jipw[iv1,iv2] >= 0:
                            inti[iv1,iv2] = istpw[jipw[iv1,iv2],ikxc]
                
                # Vxc_{ij} = \sum_{G1,G2,K} (Aeig_{i,G1})* V_{xc,K}<G1| e^{iK\vr}|G2> Aeig_{j,G2}
                tmat1 = la_matmul( conj(Aeigc), inti ) * ks.Vxcs[ikxc]
                Vxci += la_matmul( tmat1, Aeigc.T)
                
            tm14 = timer()
            t_muffint  += tm13-tm12
            t_interst += tm14-tm13
            
            ns,ne = ks.ibgw,ks.nbgw
            Vxct[irk,ns:ne,ns:ne] = Vxcmt[ns:ne,ns:ne] + Vxci[ns:ne,ns:ne]
            if self.DEBUG:
                for i in range(len(Vxcmt)):
                    print('%s%3d %s%14.9f%14.9f' % ('ie=', i+1, 'vxcmt=', Vxcmt[i,i].real, Vxcmt[i,i].imag))
                for ie in range(nbmax):
                    print('%s%3d %s%3d %s%14.9f %14.9f' % ('irk=', irk, 'ie=', ie, 'Vxci=', Vxci[ie,ie].real, Vxci[ie,ie].imag))

        print('## Vxc:      t(lapwcoef)           =%14.9f' % t_lapwcoef, file=fout)
        print('## Vxc:      t(lapack)             =%14.9f' % t_lapack, file=fout)
        print('## Vxc:      t(muffin)             =%14.9f' % t_muffint, file=fout)
        print('## Vxc:      t(interst)            =%14.9f' % t_interst, file=fout)
        
        
        for irk in range(len(kqm.kirlist)):
            for i in range(nbmax):
                print('%s%3d %s%3d %s%14.9f %14.9f' % ('irk=', irk+1, 'ie=', i+1, 'Vxc=', Vxct[irk,i,i].real, Vxct[irk,i,i].imag), file=fout)
        return Vxct

    def optimal_eta(self, rbas, br2):
        """ This function calculates the optimal value of eta for the lattice
            summations needed to obtain the structure constants.
        """
        lrbs = [ linalg.norm(rbas[i,:]) for i in range(3)]
        lgbs = [ linalg.norm(br2 [:,i]) for i in range(3)]
        return sqrt(2*min(lrbs)/min(lgbs))

    def calc_tildeg(self, lmax):
        """ Calculates $\tilde{g}_{lm,l'm'}$ according to equation \ref{tildea},
             \begin{equation}\label{calctilg}
               \tilde{g}_{lm,l'm'}=\sqrt{4\pi}(-1)^{l}\sqrt{\frac{(l+l'+m+m')!(l+l'-m-m')!}%
                {(2l+1)(2l'+1)[2(l+l')+1]!(l+m)!(l-m)!(l'+m')!(l'-m')!}}
              \end{equation}
             needed for the calculation of the structure constants, for l and l' = 0
        ...  lmax and stores them in memory.
        """

        self.tilg = zeros( int((lmax+1)*(lmax+2)*(lmax+3)*(3*lmax+2)/12) )
        i=0
        for l1 in range(lmax+1):
            p0 = (-1)**l1 * 8*pi*sqrt(pi)
            for l2 in range(l1+1):
                denom = (2*l1+1)*(2*l2+1)*(2*(l1+l2)+1)
                p1 = p0 / sqrt(denom)                
                for m1 in range(-l1,l1+1):
                    for m2 in range(l2+1):
                        jm1 = l1+m1
                        jm2 = l2+m2
                        jm3 = l1-m1
                        jm4 = l2-m2
                        combj = special.binom( jm1+jm2, jm1 ) * special.binom( jm3+jm4, jm3 )
                        self.tilg[i] = p1*sqrt(combj)
                        i+=1

        #for i in range(len(self.tilg)):
        #    print '%s%5d %s%20.10f' % ('i=', i, 'tildg=', self.tilg[i])
    
    def r_cutoff(self, tol, eta, lambdamax, alat):
        """ Estimates the cutoff radius of the sums in real space for the calculation of the structure constants by the solving the equation:
       \begin{equation}
       \mathfrak{E}_{R,\lambda}^{\textrm{tol}}=\left\{%
       \begin{array}{ll}
       \frac{4\pi}{(\lambda -2)\Gamma(\lambda+\tfrac{1}{2})}%
       \left(\frac{\Gamma[\tfrac{\lambda}{2}+\tfrac{3}{2},\left(\tfrac{R_c}{\eta}\right)^2]}%
       {\eta^{\lambda-2}}-\frac{\Gamma[\lambda+\tfrac{1}{2},\left(\tfrac{R_c}{\eta}\right)^2]}%
       {R_c^{\lambda-2}}\right)&\lambda \neq 2\\
       \frac{4\pi}{\Gamma(\tfrac{5}{2})}\left[\tfrac{\eta}{R_c}%
       \Gamma[3,\left(\tfrac{R_c}{\eta}\right)^2]-\Gamma[\tfrac{5}{2},\left(\tfrac{R_c}{\eta}\right)^2]\right]&
       \lambda=2\\
       \end{array}
       \right.
       \end{equation}
        and taking the maximum value of $R_c$ obtained for $\lambda = 1...$ \verb lambdamax.
        """
        rnot = max(alat)
        rct = 50 * ones(lambdamax+1)
        eps = zeros(lambdamax+1)
        which_l = list(range(lambdamax+1))
        which_l.remove(2)
        four_pi = 4*pi
        ls = arange(lambdamax+1)           # all possible l's
        ls[2] = 0 # jus so that we do not divide by 0
        etal = 4*pi/(eta**(ls-2) * (ls-2)) # 4*pi/(eta**(l-2)*(l-2))
        ls[2] = 2
        gmms = special.gamma(ls+0.5)       # 
        gmns = special.gamma((ls+1)/2.)    #
        for i in range(1,101):
            x = i/2.0
            x2 = x**2
            gaml32 = special.gammaincc( 3,   x2 ) * 2
            gmm = gmms[2] # special.gamma( 2.5 )
            gaml12 = special.gammaincc( 2.5, x2 ) * gmm
            prefac = four_pi/gmm
            eps[2]= abs( prefac * ( gaml32 / x - gaml12 ) )
            if (eps[2] < tol) and (x < rct[2]) :
                rct[2] = x
            #print '%s%3d %s%10.4f %s%3d %s%12.4f %s%12.4f %s%12.4f %s%12.4f %s%12.9f %s%12.9f' % ('i=',i,'x=',x,'l1=',2,'g32=',gaml32,'g12=',gaml12,'gm=',gmms[2],'pref=',prefac,'eps=',eps[2],'rct=',rct[2])
            for l1 in which_l:
                gaml32 = special.gammaincc( (l1+1)/2., x2 ) * gmns[l1]
                gaml12 = special.gammaincc(  l1 + 0.5, x2 ) * gmms[l1]
                eps[l1] = abs( etal[l1] * ( gaml32 - gaml12/x**(l1-2) ) / gmms[l1] )
                if (eps[l1] < tol) and (x < rct[l1]) :
                    rct[l1] = x
                
                #print '%s%3d %s%10.4f %s%3d %s%12.4f %s%12.4f %s%12.4f %s%12.4f %s%12.9f %s%12.9f' % ('i=',i,'x=',x,'l1=',l1,'g32=',gaml32,'g12=',gaml12,'gm=',gmms[l1],'pref=',prefac,'eps=',eps[l1],'rct=',rct[l1])
        return max(rct) * eta
    
    def g_cutoff(self, tol, eta, lambdamax, pia, Vol):
        """ Estimates the cutoff radius of the sums in reciprocal space for the
         calculation of the structure constants by the solving the equation:
         \begin{equation}
         \mathfrak{E}_{G,\lambda}^{\textrm{tol}}=\frac{8(\pi)^{\frac{5}{2}}}{\Omega%
         \Gamma(\lambda+\tfrac{1}{2})\eta^{\lambda+1}}%
         \Gamma\left[\tfrac{\lambda+1}{2},\left(\tfrac{\eta G_c}{2}\right)^2\right]
         \end{equation}
          and taking the maximum value of $G_c$ obtained for $\lambda = 1...$ \verb lambdamax.
        """
        rnot = max(pia)
        rct = 50 * ones(lambdamax+1)
        eps = zeros(lambdamax+1)
        ls = arange(lambdamax+1)
        gmms = special.gamma(ls+0.5)       # 
        prefac = 8 * pi**2.5 /( Vol * eta**(ls+1) * gmms )
        gmns = special.gamma((ls+1)/2.)    #
        for i in range(1,101):
            x = i/2.0
            for l1 in range(lambdamax+1):
                gaml32 = special.gammaincc( (l1+1)/2., x**2 ) * gmns[l1]
                eps[l1] = abs( prefac[l1]*gaml32 )
                if (eps[l1] < tol) and (x < rct[l1]) :
                    rct[l1] = x
        return max(rct)*2/eta

    def frcut_coul(self, rcut_coul, alat, ndivs, Vol):
       if rcut_coul < 0.0:    # truncated/screened Coulomb interaction 
           if int(rcut_coul) == -2:
               return max(alat[:]*ndivs[:])/2
           else:
               vtot = Vol*ndivs[0]*ndivs[1]*ndivs[2]
               return (vtot/(4*pi/3))**(1/3.)
           print('set default rcut_coul to ', rcut_coul, file=fout)
       else:
           return rcut_coul

    def Zero_Momentum_Coulomb(self, kmax, mbsize, ortho, strc, latgen, pb, fout):
        gmax = 10 * pb.kmr * kmax
        ng = [ int(gmax*latgen.pia[i])+1 for i in range(3)]
        np_tmp, ngindx = fkp.count_number_pw(ng,kmax,gmax,latgen.br2)
        glen, G_c, gindex = fkp.generate_pw(ng,ngindx,gmax,latgen.pia,latgen.br2,ortho,strc.lattice[1:3]=='CXZ',True)
        
        ngs, gind4 = fCl.pw_get_short_length(glen)
        print('gmax=', gmax, 'ng=', ng, 'ngindx=', ngindx, 'len(gindex)=', len(gindex), 'ngs=', ngs, file=fout)
        
        glen0,phase = fCl.pw_get_short_and_phase(ngs,gind4,gindex,glen,strc.vpos,strc.mult)
        if False: #@@??
            print('phase in python')
            for ig in range(len(glen0)):
                print('%5d ' % (ig,), end='')
                for idf in range(len(pb.atm)):
                    for jdf in range(len(pb.atm)): # over atoms again
                        print('%10.4f %10.4f   ' % (phase[idf,jdf,ig].real, phase[idf,jdf,ig].imag), end='')
                print()
            print('phase python end')
            
        sing = {}
        irms = [[] for iat in range(strc.nat)]
        for iat in range(strc.nat):
            #npt = strc.nrpt[iat]
            #dh  = log(strc.rmt[iat]/strc.r0[iat])/(npt - 1)      # logarithmic step for the radial mesh
            #dd = exp(dh)
            #rx = strc.r0[iat]*dd**range(npt)
            rx, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
            
            sinf = zeros((ngs,npt))
            for ig in range(ngs):   # now iterating only over G's of different length
                # each |G| length appears only once
                sinf[ig,:] = sin( glen0[ig] * rx )  # sinf[ir] = sin(G*r_i)
                
            for irm in range(len(pb.big_l[iat])):  # over all product functions
                L = pb.big_l[iat][irm]             # L of the product function
                if L==0:                           # only L==0 is important for q==0
                    au, bu = pb.ul_product[iat][irm], pb.us_product[iat][irm]
                    # now iterating only over G's of different length                            
                    sing[(iat,irm)] = [ rd.rint13g(strc.rel, au, bu, sinf[ig], sinf[ig], dh, npt, strc.r0[iat]) for ig in range(ngs) ]
                    irms[iat].append(irm) # which product basis have L==0

        const = 16*pi**2/latgen.Vol
        G_2_4 = const/glen0**4  # const/G^4
        sinsing={}
        for (iat,irm) in list(sing.keys()):
            for (jat,jrm) in list(sing.keys()):
                sinsing[(iat,jat,irm,jrm)] = G_2_4[:] * sing[(iat,irm)][:] * sing[(jat,jrm)][:]  # <u_i|sin(G*r)><u_j|sin(G*r)>*const/G^4
        
        Vmat = zeros((mbsize,mbsize), dtype=complex)
        for idf in range(len(pb.atm)): # over atoms
            iat = pb.atm[idf]          # atom in product basis
            for irm in irms[iat]:      # product basis index 
                im = pb.iProd_basis[(idf,irm,0,0)]  # product basis index for l=0,m=0, this atom
                for jdf in range(len(pb.atm)): # over atoms again
                    jat = pb.atm[jdf]
                    for jrm in irms[jat]:
                        jm = pb.iProd_basis[(jdf,jrm,0,0)]
                        vmat = sum(phase[idf,jdf,:] * sinsing[(iat,jat,irm,jrm)][:]) # e^{i*G(R1-R2)} * <u_{R1,i}|sin(G*r)><u_{R2,j}|sin(G*r)>*const/G^4
                        Vmat[im,jm] = vmat
                        #print('%4d %4d %20.15f %20.15f' % (im,jm, vmat.real,vmat.imag), [idf,irm,jdf,jrm] )   #@@??

        return Vmat

    def Calc_Olap_PW_Product(self, iat, gqlen, Lmax, rx, dh, strc, pb):
        # calcjlam
        # First we compute spherical bessel functions for all plane waves up to large cutoff ( indgq[iq,:] ). Different lengths are storeed in gqlen, which we are using here.
        npt = len(rx)
        Kqr = outer(gqlen, rx)                           # Kq[ig]*rx[ir]
        jln = reshape( fCl.spher_bessel(Lmax, Kqr.flatten()), (Lmax+1,len(gqlen),npt) )    # all spherical bessels: jln[L,iq,ir] = j_L( |G+q|*r )
        for ir in range(npt):
            jln[:,:,ir] *= rx[ir]     # jln[L,iq,ir] = j_L( |K+g|*r ) * r
        jlam = zeros( ( len(pb.big_l[iat]), len(gqlen) ) )   # <j_L((q+K)r)|u_{mix,L}>=Int[j_L((q+G)r) u_{mix,L}/r r^2,r]
        for irm in range(len(pb.big_l[iat])):  # over all product functions
            L = pb.big_l[iat][irm]             # L of the product function
            au, bu = pb.ul_product[iat][irm], pb.us_product[iat][irm]
            for ig in range(len(gqlen)):
                jlam[irm,ig] = rd.rint13g(strc.rel, au, bu, jln[L,ig], jln[L,ig], dh, npt, strc.r0[iat])  # <j_L((q+G)r)|u_{mix,L}>=Int[j_L((q+G)r) u_{mix,L}/r r^2,r]
        return jlam

    def OrthogonalizedBasisInterstitials(self, iq, pw):
        ### mpwipw
        ### diagipw
        if False:
            print('pw.indgq')
            for ipw in range(pw.ngq[iq]):
                idG = pw.gindex[pw.indgq[iq,ipw],:]
                print('%4d%4d' % (ipw+1,pw.indgq[iq,ipw]+1), ' ', ('%3d'*3) % tuple(idG), '  %14.10f%14.10f' % (pw.ipwint[ipw].real,pw.ipwint[ipw].imag))
            print('pw.indgq_done')
        
        if FORT:
            # Computes overlap between plane waves : olap[jpw,ipw] = <G_{jpw}|G_{ipw}>_{interstitials}/V_{cell}
            olap = fCl.cmp_pw_olap2(pw.indgq[iq], pw.ipwint, pw.gindex, self.i_g0, pw.ngq[iq], pw.ngq[iq])
        else:
            # Computes overlap between plane waves : olap[jpw,ipw] = <G_{jpw}|G_{ipw}>_{interstitials}/V_{cell}
            olap = zeros( (pw.ngq[iq],pw.ngq[iq]), dtype=complex)
            for ipw in range(pw.ngq[iq]):
               olap[ipw,ipw] = pw.ipwint[0]
               for jpw in range(ipw+1,pw.ngq[iq]):
                   idG = pw.gindex[pw.indgq[iq,ipw],:] - pw.gindex[pw.indgq[iq,jpw],:] # G_{ipw}-G_{jpw}
                   idg = pw.ig0[tuple(idG)]                                            # index of G_{ipw}-G_{jpw}
                   olap[jpw,ipw] =      pw.ipwint[idg]                                 # olap(jpw,ipw) = <G_{jpw}|G_{ipw}>=Integrate[e^{(G_{ipw}-G_{jpw})r},{r in interstitials}]
                   olap[ipw,jpw] = conj(pw.ipwint[idg])                                # olap(ipw,jpw) = <G_{ipw}|G_{jpw}>
                   #print '%4d%4d' % (pw.indgq[iq,ipw]+1,pw.indgq[iq,jpw]+1), ('%3d'*3) % tuple(idG), '%4d' % (idg+1,)

        if sum(abs(olap.imag)) < 1e-7:  # overlap is real
            olap = array(olap.real,dtype=float)
        
        if False:
            print('olap_PW')
            for ipw in range(pw.ngq[iq]):
                for jpw in range(pw.ngq[iq]):
                    print('%3d %3d %17.12f%17.12f%17.12f%17.12f' % (ipw+1,jpw+1,olap[ipw,jpw].real,-olap[ipw,jpw].imag,olap[jpw,ipw].real,-olap[jpw,ipw].imag))
            
        # diagonalizing overlap basis for interstitial
        epsipw, sgi = linalg.eigh(olap)   # sgi[pw.ngq[iq],pw.ngq[iq]]
        
        if False:
            print('epsipw:')
            for i in range(len(epsipw)):
                print('%2d %s%16.12f ' % (i,'e=',epsipw[i]))
                #print ('%10.6f '*len(epsipw)) % tuple(sgi[:,i].real)
        

        if False: ##### ATTENTION : ONLY FOR DEBUGGING
            dat = loadtxt('si_eigvec_fort.dat')
            sgi_fort = dat[:,0::2] + dat[:,1::2]*1j
            sgi = sgi_fort
            
        # Finally, taking 1/sqrt(olap) 
        if Real_sqrt_olap:
            # Now we will try to compute real 1/sqrt(O) = U 1/sqrt(eig) U^H
            sgiH = la_matmul(sgi, dot(diag(1/sqrt(abs(epsipw))), conj(sgi.T)) )
        else:
            #sgi = ImproveDegEigenvectors(epsipw, sgi)
            sgiH = conj(sgi.T)
            for i in range(len(epsipw)):
                sgiH[i,:] *= 1/sqrt(abs(epsipw[i]))

        if False:
            print('sgi=')
            for ipw in range(pw.ngq[iq]):
                for jpw in range(pw.ngq[iq]):
                    print('%3d %3d %17.12f%17.12f%17.12f%17.12f' % (ipw+1,jpw+1,sgiH[ipw,jpw].real,-sgiH[ipw,jpw].imag,sgiH[jpw,ipw].real,-sgiH[jpw,ipw].imag))
        
        if FORT:
            # coul_mpwipw
            # note that this matrix of overlap tmat[i,j]=<G_i|G_j> is non-diagonal, and is of shape tmat[ngq,ngq_barc], where ngq_barc>ngq.
            # Here ngq is the size of the basis in the interstitials, while ngq_barc is some high-G cutoff.
            tmat = fCl.cmp_pw_olap2(pw.indgq[iq], pw.ipwint, pw.gindex, self.i_g0, pw.ngq[iq], pw.ngq_barc[iq]) 
        else:
            tmat = zeros((pw.ngq[iq],pw.ngq_barc[iq]),dtype=complex)
            for ipw in range(pw.ngq_barc[iq]):
              for jpw in range(pw.ngq[iq]):
                idG = pw.gindex[pw.indgq[iq,ipw],:] - pw.gindex[pw.indgq[iq,jpw],:]  # G_{ipw}-G_{jpw}
                idg = pw.ig0[tuple(idG)]                                             # index of G_{ipw}-G_{jpw}
                tmat[jpw,ipw] = pw.ipwint[idg]                                       # tmat[jpw,ipw] = <G_{jpw}|G_{ipw}>_{int}=Integrate[e^{i*(G_{ipw}-G_{jpw}r)},{r in interstitial}]
                #print ipw, jpw, idG, idg, tmat[ipw,jpw]
        # mpwipw[i,j] = <G_i,G_j>
        # where G_j has much larger dimension that G_i, because it needs to allow the possibility of G_j to be any combination of reciprocal vectors from KS-vector file.
        mpwipw = dot(sgiH,tmat)   # mpwipw[pw.ngq[iq],pw.ngq_barc[iq]]

        if False:
            print('mpwipw')
            for jpw in range(pw.ngq[iq]):
                for ipw in range(pw.ngq_barc[iq]):
                    print('%3d %4d %16.10f %16.10f' % (jpw+1,ipw+1,mpwipw[jpw,ipw].real,mpwipw[jpw,ipw].imag))
            print('end_mpwipw')
        return mpwipw
    
    def fwi0(self, latgen, pb):
        loctmatsize = len(pb.Prod_basis)
        wi0 = zeros(loctmatsize)
        fct = sqrt(4*pi/latgen.Vol) 
        for idf in range(len(pb.atm)):
            iat = pb.atm[idf]
            for irm in range(len(pb.big_l[iat])):  # over all product functions
                L = pb.big_l[iat][irm]             # L of the product function
                if L==0:                           # only L==0 is important for q==0
                    im = pb.iProd_basis[(idf,irm,0,0)]
                    wi0[im] = fct * pb.rtl[iat,irm]  # 4*pi/Vol*< r^0 | u_{im,at} > because rtl=< r^L | u_{im,at} >
                    #print 'idf=', idf, 'irm=', irm, 'L=', L, 'im=', im, 'wi0=', wi0[im]
        return wi0
                
    def Coulomb_from_PW(self, iq, io, strc, in1, latgen, kqm, ks, radf, pw, pb, fout):
        """ Calculating Coulomb by  
                <u_{product} | K+q > 4*pi/(K+q)^2 < K+q| u_{product}> 
            where  u_{product} is the product basis defined in enire space, i.e.,
            both in MT and interstitials. 
            In the MT the plane wave is expaned 
                  <r|K+q> = 4*pi*(i)^l j_l(|q+K|*r) Y_{lm}(\vr) Y_{lm}(q+K)
            and the bessel functions are integrated in MT. 

            Also calculates mpwipw[G,K] == 1/sqrt(O) * <e^{iG*r}| e^{i*K*r}>
            Maybe we should compute mpwiw outside this, because it is used to define 
            othogonalized basis in the interstitials
        """
        ortho = (latgen.ortho or strc.lattice[1:3]=='CXZ')
        alat = array([strc.a, strc.b, strc.c])

        vq = kqm.qlistc[iq,:]/float(kqm.LCMq)
        
        if False:  ##### ATTENTION ONLY FOR DEBUGGING?????
            dat = array(loadtxt('indgq_fort.dat').T, dtype=int)
            pw.indgq[iq,:pw.ngq_barc[iq]] = dat[1,:]
            #for ig in range(pw.ngq[iq]):
            #    print ig, pw.indgq[iq,ig], dat[1,ig]
        
        gqlen = array([ pw.gqlen[iq, pw.G_unique[iq,i]] for i in range(pw.ngqlen[iq]) ])  # lengths of unique |q+G|
        Lmax =  max( [ max(pb.big_l[iat]) for iat in range(strc.nat)])                    # maximum possible value of L in product basis
        
        loctmatsize = len(pb.Prod_basis)
        mbsize =  loctmatsize + pw.ngq[iq]
        im_start = zeros(strc.nat+1, dtype=int)                                           # first index for product basis on each atom
        idf = 0
        for iat in range(strc.nat):
            im_start[iat] = pb.iProd_basis[(idf,0,0,0)]                                   # first index of product basis on each atom
            idf += strc.mult[iat]
        im_start[strc.nat] = loctmatsize                                                  # the last index for product basis on the last atom
        # mpwmix : <u_{product_basis_everywhere}| e^{i*K*r}>
        mpwmix = zeros((mbsize,pw.ngq_barc[iq]), dtype=complex)
        for iat in range(strc.nat):
            rx, dh, npt = strc.radial_mesh(iat)
            jlam = self.Calc_Olap_PW_Product(iat, gqlen, Lmax, rx, dh, strc, pb)
            istr, iend = im_start[iat], im_start[iat+1]
            # Now computing matrix elements <e^{(q+G)\vr}|V_{coul}|u_{irm}>  = 4*pi/|q+G|^2 e^{-i(q+G)_R }<q+G|u_{irm}>
            mpwmix[istr:iend,:] = fCl.mixed_coulomb(vq,iat+1,False,jlam,pw.gindex,pw.indgq[iq],pw.gqlen[iq],gqlen,pw.G_unique[iq],pw.ngq_barc[iq],iend-istr,pb.big_l[iat],kqm.k2cartes,strc.rotloc,latgen.trotij,strc.mult,strc.vpos,latgen.Vol)
        mpwmix = conj(mpwmix)  # <u_{product_basis}| K>
        self.mpwipw = self.OrthogonalizedBasisInterstitials(iq, pw)  # 1/sqrt(Olap)*<G|K>
        mpwmix[loctmatsize:mbsize,:] = self.mpwipw
        
        #if (vq==0): mpwmix[:,0] += wi0[:]. But this is not used anyway. So should be irrelevant.
        
        gqlen = pw.gqlen[iq,:pw.ngq_barc[iq]]
        if sum(abs(vq)) < 1e-7: gqlen[0] = 1e100
        Q2 = gqlen**2
        Vq = 4*pi/Q2

        Vmm = conj(mpwmix)
        Vmm *= Vq          # Vmm[i,j] = conj(mpwmix[i,j]) * Vq[j]
        Vmat = dot(mpwmix, Vmm.T)


        self.ev, self.Vmat = linalg.eigh(Vmat)
        
        #self.wi0 = zeros(mbsize)
        #self.wi0[:loctmatsize] = me.wi0(latgen, pb)
        #self.wi0[loctmatsize:mbsize] = mpwipw[:pw.ngq[iq]]
        #print 'wi0'
        #for i in range(len(wi0)):
        #    print >> fout, '%3d %16.9f%16.9f' % (i, wi0[i].real, wi0[i].imag)
        print('mpwmix loctmatsize=', loctmatsize, file=fout)
        for j in range(pw.ngq_barc[iq]):
            for i in range(mbsize):
                print('%3d %3d %16.9f%16.9f' % (j+1, i+1, mpwmix[i,j].real, mpwmix[i,j].imag), file=fout)
        #print >> fout, '** barc eigenvalues **'
        #for i in range(len(ev)):
        #    print >> fout, '%5d%18.9f' % (i+1, ev[-1-i])
        
    def Coulomb(self, iq, io, strc, in1, latgen, kqm, ks, radf, pw, pb, fout):
        """ Calculating Coulomb in the interstitail and mixed term, we use
                <u_{product} | K+q > 4*pi/(K+q)^2 < K+q| u_{product}> 
            where  u_{product} is the product basis defined in enire space.
            In the MT part we use the real space expression for the Coulomb,
            which is cutoff-free, but it requires two center Laplace expansion.
            Also needs Ewalds summation and real space integration.

            Also calculates mpwipw[G,K] == 1/sqrt(O) * <e^{iG*r}| e^{i*K*r}>
            Maybe we should compute mpwiw outside this, because it is used to define 
            othogonalized basis in the interstitials
        """
        t_sph_bess, t_mixed, t_MT = 0,0,0
        
        ortho = (latgen.ortho or strc.lattice[1:3]=='CXZ')
        alat = array([strc.a, strc.b, strc.c])
        
        tm9 = timer()
        #rcut_coul = self.frcut_coul(io.rcut_coul, alat, kqm.ndiv, latgen.Vol)
        
        vq = kqm.qlistc[iq,:]/float(kqm.LCMq)
        
        ndf = sum(strc.mult)
        lam_max = 4*(io.lmbmax+1)
        loctmatsize = len(pb.Prod_basis)
        mbsize =  loctmatsize + pw.ngq[iq]
        kmax = in1.rkmax/min(strc.rmt)
        print('mbsize=', mbsize, 'loctmatsize=', loctmatsize, 'kmax=', kmax, file=fout)
        
        if iq==0:
            Vmat = self.Zero_Momentum_Coulomb(kmax, mbsize, ortho, strc, latgen, pb, fout)
        else:
            Vmat = zeros((mbsize,mbsize), dtype=complex)
        
        tm10 = timer()
        self.mpwipw = self.OrthogonalizedBasisInterstitials(iq, pw)
        
        tm16 = timer()
        cq = dot(kqm.k2cartes, vq)
        sgm = fCl.ewald_summation(cq, lam_max, strc.vpos, latgen.br2, latgen.rbas, alat, latgen.Vol, ortho, self.r_cf, self.g_cf, self.eta) # coul_strcnst
        tm17 = timer()
        
        gqlen = array([ pw.gqlen[iq, pw.G_unique[iq,i]] for i in range(pw.ngqlen[iq]) ])  # lengths of unique |q+G|
        Lmax =  max( [ max(pb.big_l[iat]) for iat in range(strc.nat)])                    # maximum possible value of L in product basis
        nmix = array([ len(pb.big_l[iat]) for iat in range(strc.nat) ], dtype=int)        # nmix is the length of product basis on each atom
        max_nmix = max( nmix )
        im_start = zeros(strc.nat+1, dtype=int)                                           # first index for product basis on each atom
        big_l = zeros( (max_nmix,strc.nat), dtype=int, order='F' )                        # big_l in fortran array form
        idf = 0
        for iat in range(strc.nat):
            im_start[iat] = pb.iProd_basis[(idf,0,0,0)]                                   # first index of product basis on each atom
            nm = len(pb.big_l[iat])                                                       # length of the atomic product basis on this atom
            big_l[:nm,iat] = pb.big_l[iat][:]                                             # saving big_l to firtran-like array
            idf += strc.mult[iat]                                                         # what is the index of atom in array of all atoms, including equivalent
        im_start[strc.nat] = loctmatsize                                                  # the last index for product basis on the last atom
        
        #tm18 = timer()
        rtlij = zeros((max_nmix,max_nmix,strc.nat,strc.nat),order='F')
        for iat in range(strc.nat):
            nim = len(pb.big_l[iat])
            for jat in range(strc.nat):
                njm = len(pb.big_l[jat])
                rtlij[:njm,:nim,jat,iat] = outer(pb.rtl[jat,:njm] , pb.rtl[iat,:nim])   # bug jul.7 2020
        
        #tm19 = timer()
        Vmatit = zeros((loctmatsize,pw.ngq_barc[iq]), dtype=complex)
        
        for iat in range(strc.nat):
            rx, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
            
            tm20 = timer()
            ## First we compute spherical bessel functions for all plane waves up to large cutoff ( indgq[iq,:] ).
            # Different lengths are storeed in gqlen, which we are using here.
            # Next we compute matrix elements with the atomic centered product basis functions
            jlam = self.Calc_Olap_PW_Product(iat, gqlen, Lmax, rx, dh, strc, pb)
            tm21 = timer()
            
            istr, iend = im_start[iat], im_start[iat+1]
            
            # Now computing matrix elements <e^{(q+G)\vr}|V_{coul}|u_{irm}>  = 4*pi/|q+G|^2 e^{-i(q+G)_R }<q+G|u_{irm}>
            Vmatit[istr:iend,:] = fCl.mixed_coulomb(vq,iat+1,True,jlam,pw.gindex,pw.indgq[iq],pw.gqlen[iq],gqlen,pw.G_unique[iq],pw.ngq_barc[iq],iend-istr,pb.big_l[iat],kqm.k2cartes,strc.rotloc,latgen.trotij,strc.mult,strc.vpos,latgen.Vol)
            tm22 = timer()
            # Here is the muffin-thin part
            Vmat[:loctmatsize,:loctmatsize] += fCl.mt_coulomb(vq,iat+1,loctmatsize,big_l,nmix,im_start,self.tilg,pb.rrint,pb.djmm,rtlij,sgm,strc.mult,lam_max)
            tm23 = timer()
            
            t_sph_bess += tm21-tm20
            t_mixed += tm22-tm21
            t_MT += tm23-tm22

        if False: #@@??
            print('vmat')
            for im in range(loctmatsize):
                for jm in range(loctmatsize):
                    if abs(Vmat[im,jm]) > 1e-7 :
                        if abs(Vmat[im,jm].imag) > 1e-7 : 
                            print('%4d %4d %17.12f %17.12f' % (im+1, jm+1, Vmat[im,jm].real, Vmat[im,jm].imag))
                        else:
                            print('%4d %4d %17.12f' % (im+1, jm+1, Vmat[im,jm].real))
        #tm24 = timer()
        # This transforms into the plane-wave-product basis
        mat2 = dot(self.mpwipw, Vmatit.T)      # mat2[ pw.ngq[iq], loctmatsize]
        # Setting mixed- contribution to final storage
        Vmat[loctmatsize:mbsize,:loctmatsize] = mat2[:,:]
        Vmat[:loctmatsize,loctmatsize:mbsize] = conj(mat2.T)
        #tm25 = timer()
        
        # Now diagonal plane-wave part, i.e, plane-wave-product basis: mpwipw[i,j] * Vq[j] * mpwipw.H[j,k]
        gqlen = pw.gqlen[iq,:pw.ngq_barc[iq]]
        if sum(abs(vq)) < 1e-7: gqlen[0] = 1e100
        
        Q2 = gqlen**2
        Vq = 4*pi/Q2
        # Vmm[i,k] = mpwipw[i,j] * Vq[j] * mpwipw.H[j,k]
        Vmm = zeros((pw.ngq[iq],pw.ngq_barc[iq]), dtype=complex)
        for k in range(pw.ngq[iq]):
            Vmm[k,:] = Vq[:]*conj(self.mpwipw[k,:])
        m2 = dot(self.mpwipw, Vmm.T)
        Vmat[loctmatsize:mbsize,loctmatsize:mbsize] = m2[:,:]
        self.loctmatsize = loctmatsize
            
        tm11 = timer()
        print('## Coulomb: t(optimal_etas_cutoffs)=%14.9f' % (tm10-tm9,), file=fout)
        print('## Coulomb: t(Ewald)               =%14.9f' % (tm17-tm16,), file=fout)
        print('## Coulomb: t(spherical_bessel)    =%14.9f' % t_sph_bess, file=fout)
        print('## Coulomb: t(Muffin-thin)         =%14.9f' % t_MT, file=fout)
        print('## Coulomb: t(mixed_MT_PW)         =%14.9f' % t_mixed, file=fout)

        self.ev, self.Vmat = linalg.eigh(Vmat)

        if False: ### ATTENTION  Only for debugging
            dat = loadtxt('CoulEigvec.dat')
            Vm = dat[:,0::2] + dat[:,1::2]*1j
            self.Vmat[:,:] = Vm[:,:]  # This should make Coulomb diagonalized matrix exactly the same as in fortran
        
        if iq==0:
            wi0 = zeros(mbsize, dtype=complex)
            wi0[:loctmatsize] = self.fwi0(latgen, pb)             # <Chi_product| 1 >
            wi0[loctmatsize:mbsize] = self.mpwipw[:pw.ngq[iq],0]  # <G_orthohonal_basis|G=0>
            wi0new = dot(wi0, conj(self.Vmat))                    # V^\dagger < Chi_product| 1> = overlap of singular vector with unity in space
            
            self.immax = argmax(abs(wi0new))                      # which singular eigenvector V_{l,:} has largest overlap with unity?
            print('- Maximum singular eigenvector **', self.immax, abs(wi0new[self.immax]), self.ev[self.immax], file=fout)
            
            alpha = (latgen.Vol/(6*pi**2))**(1./3.)
            ankp = kqm.ndiv[0]*kqm.ndiv[1]*kqm.ndiv[2]
            ifst = 1
            expQ = exp(-alpha*Q2[ifst:])
            sf1 = sum(expQ/gqlen[ifst:])/ankp
            sf2 = sum(expQ/Q2[ifst:])/ankp
            for iq in range(1,len(kqm.qlist)):
                Q1 = pw.gqlen[iq,:pw.ngq_barc[iq]]
                Q2 = Q1**2
                expQ = exp(-alpha*Q2)
                f1 = sum(expQ/Q1)/ankp
                f2 = sum(expQ/Q2)/ankp
                sf1 += f1
                sf2 += f2
            intf  = latgen.Vol/(4*pi**2)
            intf1 = intf/alpha
            intf2 = intf*sqrt(pi/alpha)
            self.singc1 = intf1-sf1   # see Eq.B6 (page 364) in gap2 paper (Eq.B3..Eq.B6)
            self.singc2 = intf2-sf2   # also Ref.61 in gap2 paper
            
        if False:
            print('Vmat=', file=fout)
            for im in range(mbsize):
                for jm in range(mbsize):
                    if (Vmat[im,jm] != 0 ):
                        if (im < loctmatsize and jm < loctmatsize):
                            (idf,irm,iL,iM) = pb.Prod_basis[im]
                            (jdf,jrm,jL,jM) = pb.Prod_basis[jm]
                            print(('%3d '*2+'%2d'*3+'%3d'+','+'%2d'*3+'%3d'+' %14.9f%14.9f') % (im+1,jm+1,idf+1,iL,iM,irm+1, jdf+1,jL,jM,jrm+1, Vmat[im,jm].real, Vmat[im,jm].imag), file=fout)
                        elif (im < loctmatsize):
                            (idf,irm,iL,iM) = pb.Prod_basis[im]
                            ii = pw.indgq[iq,jm-loctmatsize]
                            jG = pw.gindex[ii,:]
                            print(('%3d '*2+'%2d'*3+'%3d'+','+'   '+'%2d'*3+' %14.9f%14.9f') % (im+1,jm+1,idf+1,iL,iM,irm+1, jG[0],jG[1],jG[2], Vmat[im,jm].real, Vmat[im,jm].imag), file=fout)
                        elif (jm < loctmatsize):
                            (jdf,jrm,jL,jM) = pb.Prod_basis[jm]
                            ii = pw.indgq[iq,im-loctmatsize]
                            iG = pw.gindex[ii,:]
                            print(('%3d '*2+'%2d'*3+'   '+','+'%2d'*3+'%3d'+' %14.9f%14.9f') % (im+1,jm+1,iG[0],iG[1],iG[2], jdf+1,jL,jM,jrm+1, Vmat[im,jm].real, Vmat[im,jm].imag), file=fout)
                        else:
                            ii = pw.indgq[iq,im-loctmatsize]
                            iG = pw.gindex[ii,:]
                            ii = pw.indgq[iq,jm-loctmatsize]
                            jG = pw.gindex[ii,:]
                            print(('%3d '*2+'%2d'*3+'   '+','+'   '+'%2d'*3+' %14.9f%14.9f') % (im+1,jm+1,iG[0],iG[1],iG[2], jG[0],jG[1],jG[2], Vmat[im,jm].real, Vmat[im,jm].imag), file=fout)
        #print >> fout, '** barc eigenvalues **'
        #for i in range(len(self.ev)):
        #    print >> fout, '%5d%18.9f' % (i+1, self.ev[-1-i])

    def Coul_setev(self, iq, fout, iop_coul_x, evtol=1e-8):
        # evtol -- eigenvalues bigger than this cutoff will be kept
        mbsize = len(self.ev)
        if Real_sqrt_Vcoul:
            matsize = mbsize
            self.barcev = self.ev
            # Here we literaly compute sqrt(V_coulomb) using eigenvectors and eigenvalues
            # The hope is that such procedure leads to more stable and real values of V_coulomb, which do not have arbitrary complex phase due to diagonalization
            self.barcvm = la_matmul(self.Vmat, dot(diag(sqrt(abs(self.ev))), conj(self.Vmat.T)) )
            #print 'Imaginary part of Vcoul=', sum(abs(self.barcvm.imag))/(mbsize*mbsize)
            
            print("  - Old/New basis set size =", mbsize, matsize, file=fout)
            print('** barc eigenvalues **', file=fout)
            for im in range(matsize-1,-1,-1):
                print('%3d %14.8f' % (matsize-im, self.barcev[im]),'v=', ('%13.8f'*10) % tuple(self.barcvm[:10,im].real), file=fout)
        else:
            #print('evs=', self.ev[0], self.ev[-1], 'evtol=', evtol)#@@!!
            im_kept = self.ev > evtol
            if iq==0:
                if iop_coul_x == 0:      # even if the eigenvalue is larger than cutooff, we will still remove it
                    im_kept[self.immax] = False
                else:                       # now if io.iop_coul_x !=0 then we will always keep this eigenvector
                    im_kept[self.immax] = True
            
            self.barcev = self.ev[im_kept]                              # using mask im_kept to pick the data
            self.barcvm = self.Vmat[:,im_kept] * sqrt(abs(self.barcev))    # using mask im_kept to pick the data
            self.barcev = self.barcev[::-1]  # we revert the order of eigenvalues
            self.barcvm = self.barcvm[:,::-1] # and also eigenvectors
            matsize = len(self.barcev)
            
            print("  - Old/New basis set size =", mbsize, matsize, file=fout)
            print('** barc eigenvalues **', file=fout)
            for im in range(matsize):
                print('%3d %14.8f' % (im+1, self.barcev[im]),'v=', ('%13.8f'*10) % tuple(self.barcvm[:10,im].real), file=fout)

            
    def calc_minm(self, ik, iq, band_limits, mode, strc, in1, latgen, kqm, ks, radf, pw, pb, core, rest, t_times, fout, PRINT=False):
        #print 'mode=', mode, 'ik=', ik, 'band_limits=', band_limits
        (isp, indggq, nmix, max_nmix, big_l, ql) = rest
        (mbsize,matsize) = shape(self.barcvm)
        (nst, nend, mst, mend, cst, cend) = band_limits
        #(nst, nend, mst, mend, cst, cend) = (ks.ibgw, ks.nbgw, 0, nomx+1, 0, ks.ncg_x)
        irk = kqm.kii_ind[ik]
        jk = kqm.kqid[ik,iq]  # index of k-q
        jrk = kqm.kii_ind[jk]
        kpts = (ik,jk,iq)
        
        t0 = timer()
        # First create alpha, beta, gamma for k
        Aeigk = array(ks.all_As[irk], dtype=complex)   # eigenvector from vector file
        kil = array(kqm.klist[ik,:])/float(kqm.LCM)  # k in semi-cartesian form

        if DMFT1_like:
            kirr = array(kqm.kirlist[irk,:])/float(kqm.LCM)  # k in semi-cartesian form
            isym = kqm.iksym[ik]
            timat_ik, tau_ik = strc.timat[isym].T, strc.tau[isym,:]
            alm,blm,clm = lapwc.dmft1_set_lapwcoef(False, 1, True, kil, kirr,timat_ik,tau_ik, ks.indgkir[irk], ks.nv[irk], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.rotij, latgen.tauij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
        else:
            if kqm.k_ind[irk] != ik:                       # not irreducible
                Aeigk *= exp( 2*pi*1j * ks.phase_arg[ik][:])  # adding phase : zzk[1:ngk,ib] = phase[1:ngk] * zzk[1:ngk,ib]   # WARNING: change 2022
            alm,blm,clm = lapwc.gap2_set_lapwcoef(kil, ks.indgk[ik], 1, True, ks.nv[irk], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
            
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

        # And next create alpha, beta, gamma for k+q
        Aeigq = array( conj( ks.all_As[jrk] ), dtype=complex)  # eigenvector from vector file
        kjl = array(kqm.klist[jk,:])/float(kqm.LCM)          # k+q in semi-cartesian form
        if DMFT1_like:
            kjrr = array(kqm.kirlist[jrk,:])/float(kqm.LCM)
            isym = kqm.iksym[jk]
            timat_ik, tau_ik = strc.timat[isym].T, strc.tau[isym,:]
            alm,blm,clm = lapwc.dmft1_set_lapwcoef(False, 2, True, kjl, kjrr,timat_ik,tau_ik, ks.indgkir[jrk], ks.nv[jrk], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.rotij, latgen.tauij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
        else:
            if kqm.k_ind[jrk] != jk:                             # the k-q-point is reducible, eigenvector needs additional phase
                Aeigq *= exp( -2*pi*1j * ks.phase_arg[jk][:] )
            alm,blm,clm = lapwc.gap2_set_lapwcoef(kjl, ks.indgk[jk], 2, True, ks.nv[jrk], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)

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
        
        abc_lapw = (alfa,beta,gama,alfp,betp,gamp)
        Aeigs = (Aeigk,Aeigq)
        
        if False:
            print('clm')
            idf = -1
            for iat in range(strc.nat):
                for ieq in range(strc.mult[iat]):
                    idf = idf + 1
                    for l2 in range(shape(in1.nlo)[0]):
                        for m2 in range(-l2,l2+1):
                            l2m2=l2*l2+l2+m2
                            for ilo in range(nLOmax): #range(in1.nLO_at[l+1,iat]):
                                for i in range(ngj):
                                    if (abs(clm[i,ilo,l2m2,idf]) > 1e-8):
                                        print('idf=%2d l2m2=%3d ilo=%2d i=%4d clm=%16.10f%16.10f' % (idf+1, l2m2+1, ilo+1, i+1, clm[i,ilo,l2m2,idf].real, clm[i,ilo,l2m2,idf].imag ))
            print('gamp', 'nbmax=', nbmax)
            idf = -1
            for iat in range(strc.nat):
                for ieq in range(strc.mult[iat]):
                    idf = idf + 1
                    for l2 in range(shape(in1.nlo)[0]):
                        for m2 in range(-l2,l2+1):
                            l2m2=l2*l2+l2+m2
                            for ilo in range(nLOmax): #range(in1.nLO_at[l+1,iat]):
                                for ie2 in range(nbmax):
                                    if (abs(gamp[ie2,ilo,l2m2,idf]) > 1e-8 ):
                                        print('gamp: %4d%4d%4d%4d  %16.10f%16.10f' % (idf+1,l2m2+1,ilo+1,ie2+1, gamp[ie2,ilo,l2m2,idf].real, gamp[ie2,ilo,l2m2,idf].imag))
        if False:
            print('alfa,beta,gama=')
            for ie in range(shape(alfa)[0]):
                for lm in range(shape(alfa)[1]):
                    print('ie=%3d lm=%3d alfa=%14.10f%14.10f beta=%14.10f%14.10f gama=%14.10f%14.10f' % (ie+1,lm+1,alfa[ie,lm,0].real, alfa[ie,lm,0].imag, beta[ie,lm,0].real, beta[ie,lm,0].imag, gama[ie,0,lm,0].real, gama[ie,0,lm,0].imag))
            print('alfp,betp,gamp=')
            for ie in range(shape(alfp)[0]):
                for lm in range(shape(alfp)[1]):
                    print('ie=%3d lm=%3d alfa=%14.10f%14.10f beta=%14.10f%14.10f gama=%14.10f%14.10f' % (ie+1,lm+1,alfp[ie,lm,0].real, alfp[ie,lm,0].imag, betp[ie,lm,0].real, betp[ie,lm,0].imag, gamp[ie,0,lm,0].real, gamp[ie,0,lm,0].imag))
            
                    
        t1 = timer()
        t_times[0] += t1-t0

        # Next computing overlap of two Kohn-Sham orbitals and product basis
        s3r = pb.s3r[:,:,:,:,isp]
        ## The muffin-tin part : mmat[ie1,ie2,im] = < u^{product}_{im,lb} | psi^*_{ie2,k-q} psi_{ie1,k} > e^{-iq.r_atom}
        #    where ie1=[nst,nend] and ie2=[mst,mend]
        mmat_mt = fvxcn.calc_minm_mt(ql,nst,nend,mst,mend, alfa,beta,gama,alfp,betp,gamp,s3r,strc.vpos,strc.mult,nmix,big_l,in1.nLO_at,pb.ncore,pb.cgcoef,in1.lmax,self.loctmatsize)


        if DMFT1_like:
            # For interstitials we need to transform now, because we did not transform the eigenvectors before.
            if kqm.k_ind[irk] != ik:                       # not irreducible
                Aeigk *= exp( 2*pi*1j * ks.phase_arg[ik][:])  # adding phase : zzk[1:ngk,ib] = phase[1:ngk] * zzk[1:ngk,ib]   # WARNING: change 2022
            if kqm.k_ind[jrk] != jk:                             # the k-q-point is reducible, eigenvector needs additional phase
                Aeigq *= exp( -2*pi*1j * ks.phase_arg[jk][:] )
                
        ## The interstitial part : mmat(ie1,ie2,im) = 1/sqrt(O)*< e^{i*iGp*r}|psi_{ie2}^* |psi_{ie1}>_{Interstitials}
        iumklap = array(round_( kqm.klist[ik]/float(kqm.LCM) - kqm.klist[jk]/float(kqm.LCM) - ql ),dtype=int)
        nvi, nvj = ks.nv[irk], ks.nv[jrk]
        # The interstitial part : mmat[ie1,ie2,im] = < u^{product}_{im,lb} | psi^*_{ie2,k-q} psi_{ie1,k} > = 1/sqrt(O)*< e^{i*iG_{im}*r}|psi_{ie2,k+q}^* |psi_{ie1,k}>_{Interstitials}
        #    where ie1=[nst,nend] and ie2=[mst,mend]
        mmat_it = fvxcn.calc_minm_is(nst,nend,mst,mend,Aeigk,Aeigq,iumklap,self.mpwipw,nvi,nvj,ks.indgk[ik],ks.indgk[jk],indggq,pw.gindex,self.i_g0,latgen.Vol)
        # Combine MT and interstitial part together
        mmat = concatenate( (mmat_mt, mmat_it), axis=-1 )
        t2 = timer()
        # Now compute also product of sqrt(Vcoulomb)*mmat
        minm = la_matmul( mmat, conj(self.barcvm) )
        #print 'mmat_mt=', isfortran(mmat_mt), 'mmat_it=', isfortran(mmat_it), 'mmat=', isfortran(mmat), 'minm=', isfortran(minm), 'barcvm=', isfortran(self.barcvm)
        t3 = timer()
        t_times[1] += t2-t1
        t_times[2] += t3-t2

        minc = None
        if cend > 0:  # at least one core state, hence compute overlap of Kohn-Sham+core orbitals and product basis
            t1 = timer()
            if mode=='selfe':
                # computing <Product_Basis| psi_{i1,k} psi_{icore}^*> with i1=[nst,nend] and icore=[cst,cend], where i1 is occupied
                mmat_c = fvxcn.calc_minc(kil,nst,nend,cst,cend,alfa,beta,gama,s3r,core.corind,strc.vpos,strc.mult,nmix,big_l,in1.nLO_at,pb.ncore,pb.cgcoef,in1.lmax,self.loctmatsize)
            else:
                # computing <Product_Basis| psi_{icore} psi_{i2,k-q}^* > with i2=[mst,mend] and icore=[cst,cent], where i2 is empty
                mmat_c = fvxcn.calc_minc2(kjl,cst,cend,mst,mend,alfp,betp,gamp,s3r,core.corind,strc.vpos,strc.mult,nmix,big_l,in1.nLO_at,pb.ncore,pb.cgcoef,in1.lmax,self.loctmatsize)
                
            t2 = timer()
            # Now compute also                                                   #selfe:      minc = \sum_{i<loctmatsize} mmat_c[:nend-nstart,:cend-cstart,i]*barcvm[i, :eigen_size]
            minc = la_matmul( mmat_c, conj(self.barcvm[:self.loctmatsize,:]) )   #not(selfe): minc = \sum_{i<loctmatsize} mmat_c[:cend-cstart,:mend-mstart,i]*barcvm[i, :eigen_size]
            #minc = dot( mmat_c, conj(self.barcvm[:self.loctmatsize,:]) )
            
            t3 = timer()
            t_times[3] += t2-t1
            t_times[4] += t3-t2
            
            if False and mode == 'polarization':
                (nb1,nb2,ngq) = shape(mmat_it)
                print('mmat=')
                for im in range(self.loctmatsize):
                    for ie1 in range(nend-nst):
                        for ie2 in range(mend-mst):
                            if abs(mmat_mt[ie1,ie2,im]) > 1e-8:
                                print('%4d %4d %4d %20.14f %20.14f' % (im+1, ie1+nst+1, ie2+mst+1, mmat_mt[ie1,ie2,im].real, mmat_mt[ie1,ie2,im].imag))
                for im in range(ngq):
                    for ie1 in range(nend-nst):
                        for ie2 in range(mend-mst):
                            if abs(mmat_it[ie1,ie2,im]) > 1e-8:
                                print('%4d %4d %4d %20.14f %20.14f' % (im+self.loctmatsize+1, ie1+nst+1, ie2+mst+1, mmat_it[ie1,ie2,im].real, mmat_mt[ie1,ie2,im].imag))
                
                print('mmat_c')
                for im in range(self.loctmatsize):
                    for ic in range(cend-cst):
                        for ie2 in range(mend-mst):
                            if (abs(mmat_c[ic,ie2,im]) > 1e-8):
                                print('%4d %4d %4d %20.14f %20.14f' % (im+1, ic+1, ie2+mst+1, mmat_c[ic,ie2,im].real, mmat_c[ic,ie2,im].imag))
                print('mmat_c_end')
                
            
        #if PRINT: # Conclusion: you will need to check mmat, which is compatible, and can not check minm, because it is not unique
        #    print >> fout, 'mmat=', 'ik=', ik
        #    for imix in range(shape(mmat)[2]):
        #        for ie1 in range(shape(mmat)[0]):
        #            for ie2 in range(shape(mmat)[1]):
        #                print >> fout, '%3d %3d %3d   %12.8f%12.8f' % (imix+1, ie1+1, ie2+1, mmat[ie1,ie2,imix].real, mmat[ie1,ie2,imix].imag)

            
        return (minm, minc)
    
    def calc_selfx(self, sigx, iq, strc, in1, latgen, kqm, ks, radf, pw, pb, core, kw, io, fout):
        isp = 0
        
        indggq = pw.inverse_indgq(iq)  # index for |q+G|<cutoff. If we go through all G_i points of the large mesh, indggq[i] will give the index in smaller mesh optimized for |q+G|
        # nomx+1 is the number of occupied valence bands, nomx is the last valence band
        nomx, numin = ks.nomax_numin
        # Here we need both all occuped valence bands (nomx+1) and also all the core states. Hence nbands = nomx+1 + ncore
        
        nirkp = len(kqm.weight)
        (mbsize,matsize) = shape(self.barcvm)
        nmix = array([ len(pb.big_l[iat]) for iat in range(strc.nat) ], dtype=int)        # nmix is the length of product basis on each atom
        max_nmix = max( nmix )
        big_l = zeros( (max_nmix,strc.nat), dtype=int, order='F' )                        # big_l in fortran array form
        for iat in range(strc.nat):
            big_l[:len(pb.big_l[iat]),iat] = pb.big_l[iat][:]                             # saving big_l to fortran-like array
        
        ql = kqm.qlistc[iq,:]/float(kqm.LCMq)
        
        print(file=fout)
        #selfx = zeros((nirkp,ks.nbgw-ks.ibgw))
        t_selfx = 0
        t_times = zeros(5)
        for irk in range(nirkp):
            kl = array(kqm.kirlist[irk,:])/float(kqm.LCM)  # k in semi-cartesian form
            ik = kqm.k_ind[irk]   # index in all-kpoints, not irreducible
            jk = kqm.kqid[ik,iq]  # index of k-q point index in reducible mesh
            jrk = kqm.kii_ind[jk] # k-q but its irreducible equivalent.
            kpts = (ik,jk,iq)
            
            band_limits = (ks.ibgw, ks.nbgw, 0, nomx+1, 0, ks.ncg_x)

            rest = (isp, indggq, nmix, max_nmix, big_l, ql)
            minm, minc = self.calc_minm(ik, iq, band_limits, 'selfe', strc, in1, latgen, kqm, ks, radf, pw, pb, core, rest, t_times, fout)
            (nst, nend, mst, mend, cst, cend)  = band_limits
            
            t2 = timer()
            kwgh = kw.kiw[jrk, mst:mend] # tetrahedral weights for occupied valence bands.
            # Note that this is like 1./all-k-points. This is the weight at k-q point, but we are
            # summing over all q-points (not just irreducible), hence we divide by weight of all-k-points
            MatrixSelfEnergy = io.MatrixSelfEnergy
            if MatrixSelfEnergy:
                (nb1,nb2,nim) = shape(minm) # nb1-user input between 0:nbgw-ibgw, and nb2 are occupied bands
                # We need sx[ie1,ie3] = -\sum_{ie2,im} minm[ie1,ie2,im]*minm[ie3,ie2,im].conj() * kwgh[ie2]
                #   which can be simplified to
                # sx[ie1,ie3] = -\sum_{ie2,im} (sqrt(kwgh[ie2])*minm[ie1,ie2,im])* (minm[ie3,ie2,im]*sqrt(kwgh[ie2])).conj()
                #   hence we modify minm[ie1,ie2,im] <= minm[ie1,ie2,im]*sqrt(kwgh[ie2])
                #   so that we have matrix product
                # sx[ie1,ie3] = -\sum_{ie2,im} minm[ie1,ie2 im] * minm.H[ie2 im, ie3]
                for ie2 in range(nb2):
                    minm[:,ie2,:] *= sqrt(abs(kwgh[ie2])) # occupied bands have weight associated with f(e_{k+q})*w_{k+q}. In order to use matrix product below, we 
                minm = reshape(minm, (nb1,nb2*nim))
                sx = -la_matmul(minm, minm.conj().T)
            else:
                #   ms[ie1,ie2] = sum_{im} minm[ie1,ie2,im] * conj(minm[ie1,ie2,im])
                ms = sum(minm * conj(minm), axis=2).real
                #   msw[ie1] = sum_{ie2} ms[ie1,ie2]*kwgh[ie2]
                sx = -dot(ms, kwgh)
            
            t3 = timer()
            t_selfx += t3-t2

            if cend > 0:  # at least one core state, hence compute overlap of Kohn-Sham+core orbitals and product basis
                t2 = timer()
                wjk = kwgh[0] # for the lowest band, should be 1/ankp
                if MatrixSelfEnergy:
                    #   sx[ie1,ie3] -= wjk * sum_{im,ie2} minc[ie1,ie2,im] * minc[ie3,ie2,im].conj()
                    (nb1,nbc,nim) = shape(minc)
                    minc = reshape(minc,(nb1,nbc*nim))
                    ms = la_matmul(minc, minc.conj().T)
                    sx -= ms*wjk
                else:
                    #   sx[ie1] -= wjk * sum_{im,ie2} minc[ie1,ie2,im] * minc[ie1,ie2,im].conj()
                    ms = sum(minc * minc.conj(), axis=2).real  # ms[ie1,ie2] = \sum_im minc[ie1,ie2,im]*conj(minc[ie1,ie2,im])
                    sx -= sum(ms, axis=1)*wjk                  # sigmax -= wk * \sum_ei2 ms[ie1,ie2]
                    
                t3 = timer()
                t_selfx += t3-t2
            
            if iq==0:  # correction for q==0
                ### See notes. This comes from the fact that M^{G=0}(i1,i2)=delta(i1,i2)/sqrt(Vol), therefore the singular part of
                ### - M*(4pi/q^2)*M is -4pi/Vol delta(i1,i2)*delta(i3,i2) *f(e_{k+q}) * 1/q^2
                ### The 1/q^2 is removed in the discrete sum, hence we need to add self.singc2 as a correction.
                ### The prefactor is  -4pi/Vol delta(i1,i2)*delta(i3,i2) *f(e_{k+q})*singc2
                cst = 4*pi/latgen.Vol*self.singc2*len(kqm.qlist)
                dsigx = -cst * kw.kiw[jrk, nst:nomx+1] # tetrahedral weights for occupied valence bands, which are essentially 1/len(kqm.qlist) for occupied bands.
                if MatrixSelfEnergy:
                    for i in range(nomx-nst+1):
                        sx[i,i] += dsigx[i]
                else:
                    sx[:nomx+1-nst] += dsigx

            #print('sx.imag=', sum(abs(sx.imag)))
            sigx[irk] += sx.real
            
            if False:
                print('selfx:minm', file=fout)
                for ie1 in range(nend-nst):
                    for ie2 in range(mend-mst):
                        print('%3d %3d' % (ie1, ie2), '%12.8f'*matsize % tuple(minm[ie1,ie2,:].real), file=fout)
                        print('%3d %3d' % (ie1, ie2), '%12.8f'*matsize % tuple(minm[ie1,ie2,:].imag), file=fout)
                print('minc', file=fout)
                if cend > 0 :
                    for ie1 in range(nend-nst):
                        for ie2 in range(cend-cst):
                            print('%3d %3d' % (ie1, ie2), '%12.8f'*matsize % tuple(minm[ie1,ie2,:].real), file=fout)
                            print('%3d %3d' % (ie1, ie2), '%12.8f'*matsize % tuple(minm[ie1,ie2,:].imag), file=fout)

            if MatrixSelfEnergy:
                for i in range(shape(sx)[0]):
                    print('dSigx[iq=%3d,irk=%3d,ie1=%3d,ie3=%3d]=%16.12f' % (iq,irk, i+ks.ibgw, i+ks.ibgw, sx[i,i].real), file=fout)
                for i in range(shape(sx)[0]):
                    for j in range(i,shape(sx)[1]):
                        if i!=j:
                            ratio = (abs(sx[i,j])+abs(sx[j,i]))/(abs(sx[i,i])+abs(sx[j,j])) 
                            if ratio>io.sigma_off_ratio:
                                print('dSigx[iq=%3d,irk=%3d,ie1=%3d,ie3=%3d]=%16.12f' % (iq,irk, i+ks.ibgw, j+ks.ibgw, sx[i,j].real), file=fout)
            else:
                for i in range(len(sx)):
                    print('dSigx[iq=%3d,irk=%3d,ie=%3d]=%16.12f' % (iq,irk, i+ks.ibgw, sx[i]), file=fout)
                
        print('## selfx  : t(prep_minm) [iq=%-3d]  =%14.9f' % (iq,t_times[0]), file=fout)
        print('## selfx  : t(minm)      [iq=%-3d]  =%14.9f' % (iq,t_times[1]), file=fout)
        print('## selfx  : t(minm*sV)   [iq=%-3d]  =%14.9f' % (iq,t_times[2]), file=fout)
        print('## selfx  : t(minc)      [iq=%-3d]  =%14.9f' % (iq,t_times[3]), file=fout)
        print('## selfx  : t(minc*sV)   [iq=%-3d]  =%14.9f' % (iq,t_times[4]), file=fout)
        print('## selfx  : t(cmp_selfx) [iq=%-3d]  =%14.9f' % (iq,t_selfx), file=fout)
        return sigx
    
    def calc_head(self, strc, in1, latgen, kqm, ks, radf, pw, pb, core, kw, fr, kcw, io_iop_drude, io_eta_head, dUl, Ul, fout, PartialTetra=True):
        ### calcomat
        isp = 0
        iq = 0
        indggq = pw.inverse_indgq(iq)
        # nomx+1 is the number of occupied valence bands, nomx is the last valence band
        nomx, numin = ks.nomax_numin
        # Here we need both all occuped valence bands (nomx+1) and also all the core states. Hence nbands = nomx+1 + ncore
        
        nirkp = len(kqm.weight)
        (mbsize,matsize) = shape(self.barcvm)
        nmix = array([ len(pb.big_l[iat]) for iat in range(strc.nat) ], dtype=int)        # nmix is the length of product basis on each atom
        max_nmix = max( nmix )
        big_l = zeros( (max_nmix,strc.nat), dtype=int, order='F' )                        # big_l in fortran array form
        for iat in range(strc.nat):
            big_l[:len(pb.big_l[iat]),iat] = pb.big_l[iat][:]                             # saving big_l to fortran-like array

        ql = kqm.qlistc[iq,:]/float(kqm.LCMq)

        isp=0
        (iulol_ul, iulol_udl, iul_ulol, iudl_ulol, iulol_ulol) = ks.Give_fortran_ilocals(isp,strc,in1)
        iul_ul   = ks.iul_ul[:,:,:,isp]   #= zeros((2,in1.nt-1,strc.nat,nspin),order='F')
        iul_udl  = ks.iul_udl[:,:,:,isp]  #= zeros((2,in1.nt-1,strc.nat,nspin),order='F')
        iudl_ul  = ks.iudl_ul[:,:,:,isp]  #= zeros((2,in1.nt-1,strc.nat,nspin),order='F')
        iudl_udl = ks.iudl_udl[:,:,:,isp] #= zeros((2,in1.nt-1,strc.nat,nspin),order='F')
        
        nst,nend,mst,mend = 0,nomx+1,numin,ks.nbmaxpol
        ncg = len(core.corind)
        mmatvv = zeros((nirkp,mend-mst,nend-nst,3),dtype=complex)
        mmatcv = zeros((nirkp,mend-mst,ncg,3),dtype=complex)

        t_coeff = 0
        t_matvv, t_matcv = 0, 0
        for irk in range(nirkp):
            kl = array(kqm.kirlist[irk,:])/float(kqm.LCM)  # k in semi-cartesian form
            ik = kqm.k_ind[irk]   # index in all-kpoints, not irreducible
            jk = kqm.kqid[ik,iq]  # index of k-q
            jrk = kqm.kii_ind[jk]
            kpts = (ik,jk,iq)

            t0 = timer()
            # First create alpha, beta, gamma for k
            Aeigk = array(ks.all_As[irk], dtype=complex)   # eigenvector from vector file
            kil = array(kqm.klist[ik,:])/float(kqm.LCM)  # k in semi-cartesian form
            if DMFT1_like:
                isym = kqm.iksym[ik]
                timat_ik, tau_ik = strc.timat[isym].T, strc.tau[isym,:]
                alm,blm,clm = lapwc.dmft1_set_lapwcoef(False, 1, False, kil, kil, timat_ik,tau_ik, ks.indgkir[irk], ks.nv[irk], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.rotij, latgen.tauij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
            else:
                alm,blm,clm = lapwc.gap2_set_lapwcoef(kil, ks.indgk[ik], 1, False, ks.nv[irk], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
            
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
            
            t1 = timer()
            tw1,tw2 = 0, 0
            t_coeff += t1-t0
            for ie2 in range(mst,mend):
                t3 = timer()
                alfa_n2, beta_n2 = alfa[ie2,:,:],  beta[ie2,:,:]
                if in1.nlomax > 0:
                    gama_n2 = gama[ie2,:,:,:]
                else:
                    gama_n2 = gama[0,:,:,:]
                for ie1 in range(nst,nend):
                    #  < psi_{ie1}| (-i*\nabla) | psi_{ie2}>_MT
                    if False:
                        p_mt = f_q_0.calcmmatvv_mt(ie1+1,ie2+1,alfa,beta,gama,in1.nLO_at,strc.mult,iul_ul,iul_udl,iudl_ul,iudl_udl,iulol_ul,iulol_udl,iul_ulol,iudl_ulol,iulol_ulol)
                    else:
                        calfa_n1, cbeta_n1 = conj(alfa[ie1,:,:]), conj(beta[ie1,:,:])
                        if in1.nlomax > 0:
                            cgama_n1 = conj(gama[ie1,:,:,:])
                        else:
                            cgama_n1 = gama[0,:,:,:]
                        p_mt = f_q_0.calcmmatvv_mt2(calfa_n1,cbeta_n1,cgama_n1,alfa_n2,beta_n2,gama_n2,in1.nLO_at,strc.mult,iul_ul,iul_udl,iudl_ul,iudl_udl,iulol_ul,iulol_udl,iul_ulol,iudl_ulol,iulol_ulol)
                    
                    #  < psi_{ie1}| (-i*\nabla) | psi_{ie2}>_I = \sum_{G1,G2} A*_{G1,ie1} A_{G2,ie2} <G1+k| -i*\nabla |G2+k>_I =
                    p_in = f_q_0.calcmmatvv_is(ie1+1,ie2+1,kil, Aeigk, ks.indgk[ik], ks.nv[irk], pw.ipwint,pw.gindex,self.i_g0,kqm.k2cartes)
                    if (ie1 == ie2):
                        p12 = (p_mt+p_in).real
                    else:
                        #  < psi_{ie2}| (-i*\nabla) | psi_{ie1}>_MT
                        p_mth = f_q_0.calcmmatvv_mt(ie2+1,ie1+1,alfa,beta,gama,in1.nLO_at,strc.mult,iul_ul,iul_udl,iudl_ul,iudl_udl,iulol_ul,iulol_udl,iul_ulol,iudl_ulol,iulol_ulol)
                        #  < psi_{ie2}| (-i*\nabla) | psi_{ie1}>_I = \sum_{G1,G2} A*_{G1,ie2} A_{G2,ie1} <G1+k| -i*\nabla |G2+k>_I =
                        p_inh = f_q_0.calcmmatvv_is(ie2+1,ie1+1,kil, Aeigk, ks.indgk[ik], ks.nv[irk], pw.ipwint,pw.gindex,self.i_g0,kqm.k2cartes)
                        p12 = (p_mt+p_in+conj(p_mth+p_inh))/2.
                    mmatvv[irk,ie2-mst,ie1-nst,:] = p12[:]
                    p2 = dot(p12, conj(p12)).real/3.
                    #print '%3d %3d' % (ie2+1, ie1+1), ('%16.12f %16.12f  '*3+'  %16.12f') % (p12[0].real, p12[0].imag, p12[1].real, p12[1].imag, p12[2].real, p12[2].imag, p2)
                t4 = timer()
                t_matvv += t4-t3
                if ncg > 0:
                    #  ( < psi_{ie2}| (-i*\nabla) | u_core>_MT + < u_core | (-i*\nabla) | psi_{ie2}>^*_MT )/2
                    ic_start=0
                    for iat in range(strc.nat):
                        if not core.l_core[iat]: # this atom has no core
                            continue
                        ic_end = ic_start + len(core.l_core[iat])
                        iul_ucl  = ks.iul_ucl[isp][iat]
                        iudl_ucl = ks.iudl_ucl[isp][iat]
                        iucl_ul  = ks.iucl_ul[isp][iat]
                        iucl_udl = ks.iucl_udl[isp][iat]
                        if shape(ks.iulol_ucl[isp][iat])[1] > 1:
                            iulol_ucl = ks.iulol_ucl[isp][iat][:,1:,:]
                            iucl_ulol = ks.iucl_ulol[isp][iat][:,1:,:]
                        else:
                            # this is just a fix so that we do not give empty array to fortran
                            iulol_ucl = ks.iulol_ucl[isp][iat]
                            iucl_ulol = ks.iucl_ulol[isp][iat]

                        mmcv = f_q_0.calcmmatcv(iat+1,ie2+1,core.corind,alfa,beta,gama,iul_ucl,iudl_ucl,iucl_ul,iucl_udl,iulol_ucl,iucl_ulol,in1.nLO_at)
                        mmatcv[irk,ie2-mst,:,:] += mmcv
                        #mmatcv[irk,ie2-mst,ic_start:ic_end,:] += mmcv[ic_start:ic_end,:]
                        #for ic in range(ic_start,ic_end):
                        #    print 'mmatcv[irk=%3d,ie2=%3d,iat=%3d,ic=%3d] = %14.9f' % (irk+1, ie2+1, iat+1, ic+1, sum(mmatcv[irk,ie2-mst,ic,:]).real)
                        ic_start = ic_end
                    #if ncg != ic_end:
                    #    print 'ERROR : missmatch with core states ic_end='+str(ic_end)+' while ncg='+str(ncg)
                    #    sys.exit(1)
                    
                    for icg in range(ncg):
                        p12 = mmatcv[irk,ie2-mst,icg,:]
                        p2 = dot(p12, conj(p12)).real/3.
                        #print >> fout, '%3d %3d' % (ie2+1, icg+1), ('%16.12f %16.12f  '*3+'  %16.12f') % (p12[0].real, p12[0].imag, p12[1].real, p12[1].imag, p12[2].real, p12[2].imag, p2)
                        #print 'fmatcv[irk=%3d,ie2=%3d,iat=%3d,ic=%3d] = %14.9f' % (irk+1, ie2+1, iat+1, icg+1, sum(mmatcv[irk,ie2-mst,icg,:]).real)
                        
                t5 = timer()
                t_matcv += t5-t4
        print('## cal_head t(ABC coeff)           =%14.9f' % t_coeff, file=fout)
        print('## cal_head t(matvv)               =%14.9f' % t_matvv, file=fout)
        print('## cal_head t(matcv)               =%14.9f' % t_matcv, file=fout)

        ### calchead
        t6 = timer()
        # Dielectric constant at Gamma-point
        c0_head = 0.   # 4*pi/Vol * Nspin* \sum_{i,k} 1/3*<psi_i| p |psi_i>^2 * delta(e_{k,i}-EF)
        ankp = kqm.ndiv[0]*kqm.ndiv[1]*kqm.ndiv[2]
        fspin = 2. # because we do not  sum  over spins
        (nb1,nb2,nom_nl,nkp) = shape(kcw) # the last index could be iomega or svd l
        
        head = zeros(nom_nl, dtype=complex)
        for irk in range(nirkp):
            ik = kqm.k_ind[irk]   # index in all-kpoints, not irreducible
            nvbm, ncbm = ks.nomax_numin[0]+1, ks.nomax_numin[1]
            kwt = kqm.weight[irk]
            coef = 4.*pi*kwt*fspin/latgen.Vol
            if (ks.Eg <= 0): # metallic
                # correction for metallic state
                for ie in range(ncbm,nvbm):
                    # 4*pi/Vol * Nspin*kw * 1/3*sum_{i = cross_EF} <psi_i| p |psi_i>^2
                    pnmkq2 = vdot(mmatvv[irk,ie-mst,ie-nst,:],mmatvv[irk,ie-mst,ie-nst,:]).real/3.0
                    # print 'pnmkq2=', pnmkq2, 'kwfer=', kw.kwfer[irk,ie]
                    c0_head += coef * pnmkq2 * kw.kwfer[irk,ie]
            
            #print >> fout, 'irk=', irk, 'c0_head=', c0_head
            
            # core contribution
            for icg in range(ks.ncg_p):
                iat,idf,ic,l,m = core.corind[icg,:]
                Edif = ks.Ebnd[isp,irk,ncbm:ks.nbmaxpol] - core.eig_core[isp][iat][ic]
                dwe = 1-ankp*kw.kiw[irk,ncbm:ks.nbmaxpol]  # like f(-E_{jk,ie})
                mm = mmatcv[irk,(ncbm-mst):(ks.nbmaxpol-mst),icg,:]  # mm[irk,icg][ie,3] == < u_core| -i*\nabla | psi_{irk,ie}>
                mm2 = sum(mm*conj(mm), axis=1).real/3.               # mm2[ie] = 1/3. * sum_{i} |mm[ie,i=1:3]|^2=1/3(|<u_core|p_x|psi_{irk,ie}>|^2+|<u_core| p_y|psi_{irk,ie}>|^2+|<u_core| p_z|psi_{irk,ie}>|^2)
                termcv = mm2/Edif**2

                if PartialTetra:
                    for iw,omega in enumerate(fr.omega):
                        sm = sum(termcv * (-2.*dwe/ankp)*Edif/( omega**2 + Edif**2))
                        head[iw] -= coef*conj(sm)  # head[iom] -= 4*pi/Vol*Nspin \sum_{k,i,j} 1/3|<u_i|p|u_j>|^2/(E_{k,i}-E_{k,j})^2 * Polarization_weight[i,j,k,omega]
                else:
                    _kcw_ = kcw[icg,:(ks.nbmaxpol-ncbm),:,ik]  # Polarization_weight[icg,ik][ie,iom]
                    head -= coef*conj(dot(termcv,_kcw_))    # head[iom] -= 4*pi/Vol*Nspin \sum_{k,i,j} 1/3|<u_i|p|u_j>|^2/(E_{k,i}-E_{k,j})^2 * Polarization_weight[i,j,k,omega]
            
            #print('head[irk='+str(irk)+']=')
            #for iw in range(len(head)):
            #    print('%3d %16.13f  %16.13f' % (iw, head[iw].real, head[iw].imag))
            
            # valence contribution
            mm2 = sum(abs(mmatvv[irk,(ncbm-mst):(ks.nbmaxpol-mst),0:(nvbm),:])**2,axis=2)/3.
            termvv = zeros((ks.nbmaxpol-ncbm,nvbm))
            Edif = zeros((ks.nbmaxpol-ncbm,nvbm))
            for ie2 in range(ks.nbmaxpol-ncbm):
                for ie1 in range(nvbm):
                    edif = ks.Ebnd[isp,irk,ncbm+ie2]-ks.Ebnd[isp,irk,ie1]
                    Edif[ie2,ie1] = edif
                    if abs(edif) < 1e-5:
                        #print >> fout, 'WARNING in calc_head : degenerate CB and VB'
                        termvv[ie2,ie1] = 0
                    else:
                        termvv[ie2,ie1] = mm2[ie2,ie1]/edif**2
                        
            if PartialTetra:
                dwe = 1-ankp*kw.kiw[irk,ncbm:ks.nbmaxpol]  # ie2
                dwo = ankp*kw.kiw[irk,:nvbm]               # ie1
                # (0:ks.ibmin_tetra+1,0:ks.ibmax_tetra-ncbm), (ks.ibmin_tetra+1:nvdm,0:ks.ibmax_tetra-ncbm), (0:nvdm,ns2:ne2)
                # kcw(nomx-ks.ibmin_tetra, ks.ibmax_tetra-numin, nom, nkp)
                ns2,ne2 = ks.ibmax_tetra-ncbm, ks.nbmaxpol-ncbm
                ne1 = ks.ibmin_tetra+1


                #ee = Edif[ns2:ne2,:]
                #d1 = dwe[ns2:ne2]
                #eq = ee*dwo
                #dw = tensordot(dwe[ns2:ne2], dwo, axes=0)
                #dwee = dw * ee
                #print(shape(ee), shape(dw), shape(dwee))
                #
                #rr = (dwe[ns2:ne2]*Edif[ns2:ne2,:])*dwo[:]
                #print(shape(Edif[ns2:ne2,:]), shape(rr), shape(dwe[ns2:ne2]), shape(dwo))

                
                for iw,omega in enumerate(fr.omega):
                    for ie2 in range(ns2):
                        # no tetrahedron for bands at which empty bands are in the low energy window, but occupied bands are outside the window.
                        #kcw2 = (-2./ankp)*Edif[ie2,:ne1]/( omega**2 + Edif[ie2,:ne1]**2)
                        sm = sum(termvv[ie2,:ne1] * (-2.*dwe[ie2]*dwo[:ne1]/ankp)*Edif[ie2,:ne1]/( omega**2 + Edif[ie2,:ne1]**2))
                        head[iw] -= coef*sm
                        # tetrahedron here, since both are in the low energy window
                        sm = sum(termvv[ie2,ne1:nvbm] * kcw[:,ie2,iw,ik])
                        head[iw] -= coef*sm
                    # no tetrahedron since empty bands are outside the window
                    #kcw2 = (-2./ankp)*Edif[ns2:ne2,:]/( omega**2 + Edif[ns2:ne2,:]**2)
                    sm = sum(termvv[ns2:ne2,:] * (-2./ankp)*tensordot(dwe[ns2:ne2],dwo,axes=0)*Edif[ns2:ne2,:]/( omega**2 + Edif[ns2:ne2,:]**2))
                    head[iw] -= coef*sm
                    #sm = sum(termvv * (-2./ankp)*Edif/( omega**2 + Edif**2))
                    #head[iw] -= coef*sm
            else:
                for iom_il in range(nom_nl):
                    _kcw_ = kcw[ks.ncg_p:(ks.ncg_p+nvbm),:(ks.nbmaxpol-ncbm),iom_il,ik]  # Polarization_weight[ik,iom][ie1,ie2]
                    sm = sum(termvv * _kcw_.T) # sm = sum_{i,j} 1/3|<u_i|p|u_j>|^2/(E_{k,i}-E_{k,j})^2 * Polarization_weight[i,j,k,omega]
                    head[iom_il] -= coef*sm       # head -= 4*pi/Vol*Nspin \sum_{k,i,j} 1/3|<u_i|p|u_j>|^2/(E_{k,i}-E_{k,j})^2 * Polarization_weight[i,j,k,omega]
                    

            #print('head[irk='+str(irk)+']=')
            #for iw in range(len(head)):
            #    print('%3d %16.13f  %16.13f' % (iw, head[iw].real, head[iw].imag))
            #sys.exit(0)
        
        if ks.Eg <= 0 and io_iop_drude==1 :
            # Add plasmon contribution
            print(" Intraband contribution : Calc. plasmon freq. (eV):", sqrt(c0_head)*H2eV, file=fout)
            wpl2 = c0_head

            # on imaginary axis
            if dUl is not None:
                head += dot( wpl2/(fr.omega + io_eta_head)**2, dUl)
            else:
                head += wpl2/(fr.omega + io_eta_head)**2
            # on real axis
            #head -= wpl2/(fr.omega * (fr.omega + io_eta_head*1j))
        t7 = timer()
        print('## calcomat t(dielectric head)     =%14.9f' % (t7-t6), file=fout)

        if dUl is not None:
            head_om = dot(head,Ul)
            for iom in range(len(head_om)):
                print('%3d head= %16.10f%16.10f' % (iom+1, head_om[iom].real+1.0, head_om[iom].imag), file=fout)
        else:
            for iom_il in range(nom_nl):
                print('%3d head= %16.10f%16.10f' % (iom_il+1, head[iom_il].real+1.0, head[iom_il].imag), file=fout)
        # Note that in this implementation we need to add 1.0 to head in frequency domain
        return (head, mmatcv, mmatvv, mst)
    
    def calc_eps(self, iq, head_quantities, strc, in1, latgen, kqm, ks, radf, pw, pb, core, kw, fr, kcw, ddir, dUl, Ul, fout, PartialTetra=True):
        isp = 0
        fspin = 2.
        coefw = 2.*sqrt(pi/latgen.Vol)
        if iq==0:
            (head, mmatcv, mmatvv, mst) = head_quantities
            # Converting head from svd to frequency
            if Ul is not None:
                head = dot(head,Ul)
        
        ql = kqm.qlistc[iq,:]/float(kqm.LCMq)
        indggq = pw.inverse_indgq(iq)
        nmix = array([ len(pb.big_l[iat]) for iat in range(strc.nat) ], dtype=int)        # nmix is the length of product basis on each atom
        max_nmix = max( nmix )
        big_l = zeros( (max_nmix,strc.nat), dtype=int, order='F' )                        # big_l in fortran array form
        for iat in range(strc.nat):
            big_l[:len(pb.big_l[iat]),iat] = pb.big_l[iat][:]                             # saving big_l to fortran-like array
        
        (mbsize,matsize) = shape(self.barcvm)
        (nb1_kcw,nb2_kcw,nom_nil,nkp_kcw) = shape(kcw) # kcw(io,ie,nom,nkp)
        
        #len(fr.omega)
        eps   = zeros((matsize,matsize,nom_nil), dtype=complex, order='F')
        epsw1 = zeros((matsize,nom_nil), dtype=complex, order='F')
        #epsw2 = zeros((matsize,len(fr.omega)), dtype=complex) == conj(epsw1)
        
        t_times= zeros(10)
        for ik,irk in enumerate(kqm.kii_ind):
            coef = -fspin 
            jk = kqm.kqid[ik,iq]
            jrk = kqm.kii_ind[jk]
            nvbm, ncbm = ks.nomax_numin[0]+1, ks.nomax_numin[1]
            # set the local array for band energies  
            enk = zeros( ks.ncg_p + ks.nbmaxpol)
            for icg in range(ks.ncg_p):
                iat,idf,ic = core.corind[icg,:3]
                enk[icg] = core.eig_core[isp][iat][ic]
            enk[ks.ncg_p:(ks.ncg_p+ks.nbmaxpol)] = ks.Ebnd[isp,irk,:ks.nbmaxpol] 
            # find the index for the band whose energy is higher than eminpol  
            
            band_limits = (0, nvbm, ncbm, ks.nbmaxpol, 0, ks.ncg_p)
            rest = (isp, indggq, nmix, max_nmix, big_l, ql)
            minm0, minc0 = self.calc_minm(ik, iq, band_limits, 'polarization', strc, in1, latgen, kqm, ks, radf, pw, pb, core, rest, t_times, fout)
            if minc0 is not None:
                minm = concatenate( (minc0, minm0), axis=0 ) # minc0[core-states,unoccupied-bands,product-basis], minm0[occupied-bands,unoccupied-bands,product-basis]
            else:
                minm = minm0
            
            nb1,nb2,matsiz = shape(minm)
            Ndouble_bands = (ks.ncg_p+nvbm)*(ks.nbmaxpol-ncbm)
            
            if (matsiz != matsize):
                print('Error : matsize=', matsize, 'and matsiz=', matsiz)
            if False:
                print('shape(minm)=', shape(minm), 'shape(minc0)=', shape(minc0), 'shape(minm0)=', shape(minm0), file=fout)
                print('ik=', ik, 'calceps:minm', file=fout)
                for ie1 in range(nb1):
                    for ie2 in range(nb2):
                        for imix in range(matsiz):
                            if abs(minm[ie1,ie2,imix]) > 1e-8:
                                print('%3d %3d %3d %16.12f %16.12f' % (imix+1, ie1+1, ie2+ncbm+1, minm[ie1,ie2,imix].real, minm[ie1,ie2,imix].imag), file=fout)
                print('calceps:minm_end', file=fout)
                
            t10 = timer()
            if iq==0:
                # this could be done for irreducible k-points only, hence this can be optimized
                # Calculate pm(ie12) = pm(ie1,ie2) = p_{ie1,ie2}/(e_2-e_1)
                # Needed for the wings
                mmc = sum(mmatcv[irk,(ncbm-mst):(ks.nbmaxpol-mst),:,:],axis=2)*(coefw/sqrt(3.)) # ~<psi_{i1}|p^2|psi_{i2}>
                mmv = sum(mmatvv[irk,(ncbm-mst):(ks.nbmaxpol-mst),:,:],axis=2)*(coefw/sqrt(3.))
                pm = zeros((ks.ncg_p+nvbm)*(ks.nbmaxpol-ncbm), dtype=complex) # ~<psi_{i1}|p^2|psi_{i2}>/dE
                ie12=0
                for ie1 in range(ks.ncg_p):        # core-occupied
                    for ie2 in range(ncbm,ks.nbmaxpol): # empty
                        edif = enk[ie1] - enk[ie2+ks.ncg_p]
                        if abs(edif) > 1e-10:
                            pm[ie12] = mmc[ie2-ncbm,ie1]/edif
                        #print 'ic1=%3d ie2=%3d edif=%14.10f pm=%14.10f mcv=%14.10f' % (ie1+1,ie2+1,edif,pm[ie12].real, mmc[ie2-ncbm,ie1].real)
                        ie12 += 1
                for ie1 in range(nvbm):        # valence-occupied
                    for ie2 in range(ncbm,ks.nbmaxpol): # empty
                        edif = enk[ie1+ks.ncg_p] - enk[ie2+ks.ncg_p]
                        if abs(edif) > 1e-10:
                            pm[ie12] = mmv[ie2-ncbm,ie1]/edif
                        #print 'ie1=%3d ie2=%3d edif=%14.10f pm=%14.10f mvv=%14.10f' % (ie1+1,ie2+1,edif,pm[ie12].real, mmv[ie2-ncbm,ie1].real)
                        ie12 += 1
                
                if False:
                    print('ik=', ik+1, 'pm=', file=fout)
                    for ie in range(ie12):
                        if abs(pm[ie]) > 1e-8:
                            print('%4d  %18.14f%18.14f' % (ie+1, pm[ie].real, pm[ie].imag), file=fout)
            else:
                pm = zeros(1, dtype=complex)
                
            t11 = timer()
            t_times[5] += t11-t10
            
            if False:
                minm2 = reshape(minm, (nb1*nb2,matsiz))
                print('minm2=', file=fout)
                for ie12,(ie1,ie2) in enumerate(itertools.product(list(range(ks.ncg_p+nvbm)),list(range(ks.nbmaxpol-ncbm)))):
                    for im in range(matsiz):
                        if abs(minm2[ie12,im]) > 1e-8:
                            print('%5d %5d' % (ie12+1, im+1), '%18.14f %18.14f' % (minm2[ie12,im].real, minm2[ie12,im].imag), file=fout)
                sys.exit(0)
                
            if ForceReal:
                "This is dangerous: We are truncating minm matrix elements to real components only"
                print('Imaginary part of minm=', sum(abs(minm.imag))/(matsiz*nb1*nb2), ' is set to zero!')
                minm = array(minm.real, dtype=float)

            t12 = timer()
            tmat  = zeros( (matsiz,Ndouble_bands), dtype=minm.dtype)
            
            # minm[ie12,im] = < u^{product}_{im} | psi^*_{ie2} psi_{ie1} >
            minm2 = reshape(minm, (nb1*nb2,matsiz))

            # Here we tried to do SVD on the large matrix minm, hoping that there is some redundancy in the matrix. But it seems none is found.
            if False:
                u, s, vh = linalg.svd(minm2.T, full_matrices=True)
                s_kept = s[ abs(s/s[0]) > 1e-3 ]
                n = len(s_kept)
                print('matrix size=', shape(minm2), 'n_kept=', n ) #, 's=', ('%7.4f'*n) % tuple(s_kept))
                for i in range(len(s)):
                    print(i, s[i])
                sys.exit(0)
            if False:
                for i in range(shape(minm)[2]):
                    u,s,vh = linalg.svd(minm[:,:,i], full_matrices=True) #
                    s_kept = s[ abs(s/s[0]) > 1e-3 ]
                    n = len(s_kept)
                    print(i, 'shape(minm)=', shape(minm)[:2], 'n_kept=', n)# 's_kept=', s_kept)
                
            if PartialTetra:
                ankp = len(kqm.kii_ind)
                t13 = timer()
                ncg = ks.ncg_p
                No = ncg+nvbm                     # occupied
                Ne = ks.nbmaxpol-ncbm             # empty
                Noactual = ncg+ks.ibmin_tetra+1   # occupied done by tetrahedra
                Neactual = ks.nbmaxpol-ks.ibmax_tetra # empty done by tetrahedra
                enk = zeros(No)
                for icg in range(ncg):
                    iat, idf, ic = core.corind[icg][0:3]
                    enk[icg] = core.eig_core[isp][iat][ic]
                enk[ncg:No] = ks.Ebnd[isp,irk,:(No-ncg)]
                dwo = hstack( (ones(ncg), ankp*kw.kiw[irk,:(No-ncg)]) )# like f(E_{ik,io})
                enq = ks.Ebnd[isp,jrk,ncbm:ks.nbmaxpol]    # energy at k-q
                dwe = 1-ankp*kw.kiw[jrk,ncbm:ks.nbmaxpol]  # like f(-E_{jk,ie})
                
                t14 = timer()
                t_times[6] += t14-t13
                # The size of kwc is kcw(No-Nactual,Ne-Neactual,nom,nkp)
                fvxcn.compute_m_p_m_notetra2(eps,epsw1,kcw[:,:,:,ik],minm2,enk,enq,dwo,dwe,fr.omega,pm,iq,Noactual,Neactual,coef,ankp)
                #fvxcn.compute_m_p_m_notetra2_orig(eps,epsw1,kcw[:,:,:,ik],minm2,enk,enq,fr.omega,pm,iq,Noactual,Neactual,coef,ankp)
                t15 = timer()
                t_times[7] += t15-t14
                
            #elif False:
            #    ankp = len(kqm.kii_ind)
            #    t13 = timer()
            #    ncg = ks.ncg_p
            #    No = ncg+nvbm                     # occupied
            #    Ne = ks.nbmaxpol-ncbm             # empty
            #    Noactual = ncg+ks.ibmin_tetra+1   # occupied done by tetrahedra
            #    Neactual = ks.nbmaxpol-ks.ibmax_tetra # empty done by tetrahedra
            #    enk = zeros(No)
            #    for icg in range(ncg):
            #        iat, idf, ic = core.corind[icg][0:3]
            #        enk[icg] = core.eig_core[isp][iat][ic]
            #    enk[ncg:No] = ks.Ebnd[isp,irk,:(No-ncg)]
            #    enq = ks.Ebnd[isp,jrk,ncbm:ks.nbmaxpol]  # energy at k-q
            #    t14 = timer()
            #    t_times[6] += t14-t13
            #    # The size of kwc is kcw(No-Nactual,Ne-Neactual,nom,nkp)
            #    fvxcn.compute_m_p_m_notetra2_2(eps,epsw1,kcw[:,:,:,ik],minm2,enk,enq,fr.omega,pm,iq,Noactual,Neactual,coef,ankp)
            #    t15 = timer()
            #    t_times[7] += t15-t14
            #    
            else:
                
                ### Human readable form of what we do here:
                # No = ks.ncg_p+nvbm     # occupied bands, core+valence
                # Ne = ks.nbmaxpol-ncbm  # empty bands
                # io in range(No):
                #   ie in range(Ne):
                #      eps[alp,bet,iom] += -2 * \sum_{io,ie} minm2[io*Ne+ie,alp] * kcw[io,ie,iom,ik] * minm2.C[io*Ne+ie,bet]
                #      eps1[alp,iom]    += -2 * \sum_{io,ie} minm2[io*Ne+ie,alp] * kcw[io,ie,iom,ik] * pm[io*Ne+ie]
                ###
                cminm2 = conj(minm2)
                minm2Tc = minm2.T*coef
                t_times[6] += timer()-t12
                for iom_il in range(nom_nil):
                    t13 = timer()
                    # tmat = < u^{product}_{im} | psi^*_{ie2} psi_{ie1} > F[ie12,om]
                    # coef*minm2.T*kcw[:No,:Ne]
                    tmat = minm2Tc * kcw[:(ks.ncg_p+nvbm),:(ks.nbmaxpol-ncbm),iom_il,ik].ravel() #  tmat[matsiz,ie12] = transpose(minm2[ie12,matsiz])*_kcw_[ie12]
                    t14 = timer()
                    t_times[7] += t14-t13
                    # coef*minm2.T*kcw[:No,:Ne]*minm2^*
                    eps[:,:,iom_il] += la_matmul( tmat, cminm2 )    # eps += minm2.T*(coef*kcw)*conj(minm2)
                    
                    t15 = timer()
                    t_times[8] += t15-t14
                    if iq==0: 
                        wtmp = dot(tmat,pm)           # wtmp[:matsiz] = tmat[:matsiz,ie12]*pm[ie12]
                        epsw1[:,iom_il] += wtmp       #  \sum_ioie minm2.T[:,ioie]*coef*kcw[ioie]*pm[ioie]
                        #print >> fout, 'ik=%3d  iom=%3d  wtmp^2=%18.14f  epsw1=%18.14f' % (ik+1,iom_il+1, sum(abs(wtmp)**2), sum(abs(epsw1[:,iom_il])**2) )
                        
                    t16 = timer()
                    t_times[9] += t16-t15
        #if iq==0:
        #    print('t=', t_times[7])
        #    for ia in range(matsize):
        #        for ib in range(matsize):
        #            print('%3d %3d %20.12f %20.12f  %20.12f %20.12f' % (ia,ib, eps[ia,ib,0].real, eps[ia,ib,0].imag, eps[ia,ib,2].real, eps[ia,ib,2].imag))
        #    print()
        #    #for ia in range(matsize):
        #    #    print('%3d %20.12f %20.12f' % (ia, epsw1[ia,0].real, epsw1[ia,0].imag))
        #    sys.exit(0)
        
        if iq==0:
            if Ul is not None:
                # converting epsw1 to frequency, if in svd basis
                epsw1 = dot(epsw1,Ul)
            epsw2 = copy(conj(epsw1))
        
        epsd = diagonal(eps, axis1=0, axis2=1).T.copy()  # epsd[i,iom_il] = eps[i,i,iom_il]
        if Ul is not None:
            epsd = dot(epsd,Ul) # convert to frequency when in svd basis
        epw1=0
        for iom in range(len(fr.omega)):
            #epst = sum([eps[i,i,iom].real for i in range(matsiz)])
            epst = sum(epsd[:,iom].real)
            if iq==0:
                epw1 = sum(abs(epsw1[:,iom])**2)
            print('Tr(epsilon[%11.6f]) = %18.14f  epsw1^2=%18.14f' % (fr.omega[iom], epst, epw1), file=fout)
        
        t17 = timer()
        PRINT = False
        if Ul is not None:
            # converting the intire eps to frequency, when in svd basis
            eps = dot(eps,Ul)
        t18 = timer()
        
        emac = zeros((len(fr.omega),2),dtype=complex)
        Id = identity(shape(eps)[0])
        for iom in range(len(fr.omega)):
            # force hermicity, which should be obeyed
            eps[:,:,iom] = (eps[:,:,iom] + conj(eps[:,:,iom].T))/2.
            
            # above we just computed V*polarization, while epsilon = 1+VP
            eps[:,:,iom] += Id
            
            if False:
                eigs = linalg.eigvalsh(eps[:,:,iom])
                print('iom=%3d  eps Eigenvalues=' % (iom+1,), file=fout)
                for i in range(len(eigs)-1,-1,-1):
                    print('%3d %14.10f' % (i+1, eigs[i]), file=fout)
            # this is W/V = (1+VP)^{-1}
            eps[:,:,iom] = linalg.inv(eps[:,:,iom])   # 1/eps
            
            if iq==0:
                bw1 =      dot(eps[:,:,iom], epsw1[:,iom])
                w2b = conj(dot(eps[:,:,iom], epsw2[:,iom]))
                ww = dot(epsw2[:,iom],bw1)
                
                emac[iom,0] = 1.0+head[iom]-ww
                emac[iom,1] = 1.0+head[iom]

                head[iom] = 1/(1+head[iom]-ww)
                
                print('iom=%2d ww=%13.10f new head=%16.10f emac=[%16.10f,%16.10f]' % (iom+1,ww.real,head[iom].real,emac[iom,0].real, emac[iom,1].real), file=fout)
                
                epsw1[:,iom] = -head[iom]*bw1[:]
                epsw2[:,iom] = -head[iom]*w2b[:]
                
                eps[:,:,iom] += head[iom]*tensordot(bw1, w2b, axes = 0)

                head[iom] -= 1.0
            else:
                head, epsw1, epsw2 = None, None, None
            
            # Now we compute 1/(1+VP)-1
            eps[:,:,iom] -= Id
            
            wst = sum([eps[i,i,iom].real for i in range(matsiz)])
            print('Tr(1/eps[%3d]-1) = %18.14f' % (iom+1, wst), file=fout)
        t19 = timer()

        if dUl is not None:
            # converting back to svd basis, if svd basis is available
            eps   = dot(eps,dUl)
            if iq==0:
                epsw1 = dot(epsw1,dUl)
                epsw2 = dot(epsw2,dUl)
                head  = dot(head,dUl)
        
        t20 = timer()
        print('shape(minm)=', shape(minm), '=(nb1,nb2,matsiz)', file=fout)
        print('## calc_eps: t(prep_minm) [iq=%-3d] =%14.9f' % (iq,t_times[0]), file=fout)
        print('## calc_eps: t(minm)      [iq=%-3d] =%14.9f' % (iq,t_times[1]), file=fout)
        print('## calc_eps: t(minm*sV)   [iq=%-3d] =%14.9f' % (iq,t_times[2]), file=fout)
        print('## calc_eps: t(minc)      [iq=%-3d] =%14.9f' % (iq,t_times[3]), file=fout)
        print('## calc_eps: t(minc*sV)   [iq=%-3d] =%14.9f' % (iq,t_times[4]), file=fout) #.
        print('## calc_eps: t(wings)     [iq=%-3d] =%14.9f' % (iq,t_times[5]), file=fout)
        print('## calc_eps: t(minm2Tc)   [iq=%-3d] =%14.9f' % (iq,t_times[6]), file=fout)
        print('## calc_eps: t(minm*kcw)  [iq=%-3d] =%14.9f' % (iq,t_times[7]), file=fout)
        print('## calc_eps: t(tmat*minm) [iq=%-3d] =%14.9f' % (iq,t_times[8]), file=fout) #.
        print('## calc_eps: t(wings2)    [iq=%-3d] =%14.9f' % (iq,t_times[9]), file=fout)
        print('## calc_eps: t(svd2iom)   [iq=%-3d] =%14.9f' % (iq,t18-t17), file=fout)
        print('## calc_eps: t(inverse)   [iq=%-3d] =%14.9f' % (iq,t19-t18), file=fout)
        print('## calc_eps: t(iom2svd)   [iq=%-3d] =%14.9f' % (iq,t20-t19), file=fout)
        return (eps, epsw1, epsw2, head)

    def calc_selfc(self, sigc, iq, eps, epsw1, epsw2, head, strc, in1, latgen, kqm, ks, radf, pw, pb, core, kw, fr, ddir, dUl, Ul, fout, io, PRINT=True):
        ##
        ## You should save eps, epsw1, and epsw2, and then run calc_selfc for arbitrary k-points....
        ##
        isp = 0
        ql = kqm.qlistc[iq,:]/float(kqm.LCMq)
        indggq = pw.inverse_indgq(iq)
        nmix = array([ len(pb.big_l[iat]) for iat in range(strc.nat) ], dtype=int)        # nmix is the length of product basis on each atom
        max_nmix = max( nmix )
        big_l = zeros( (max_nmix,strc.nat), dtype=int, order='F' )                        # big_l in fortran array form
        for iat in range(strc.nat):
            big_l[:len(pb.big_l[iat]),iat] = pb.big_l[iat][:]                             # saving big_l to fortran-like array
        
        if iq==0:
            coefs2 = self.singc2*4*pi/latgen.Vol
            coefs1 = self.singc1*sqrt(4*pi/latgen.Vol)
        wkq = 1.0/len(kqm.qlist)
        
        nirkp = len(kqm.weight)
        
        if False and (dUl is None):
            for iom in range(len(fr.omega)):
                ee = linalg.eigvalsh(eps[:,:,iom])
                print('iom=%3d  eps Eigenvalues=' % (iom+1,), file=fout)
                for i in range(len(ee)):
                    print('%3d %14.10f' % (i+1,ee[i]), file=fout)
            (matsiz,matsize,nom) = shape(eps)
            print('eps=', file=fout)
            for iom in range(len(fr.omega)):
                for i in range(matsiz):
                    for j in range(matsize):
                        print('%4d %4d %4d %18.14f%18.14f' % (iom+1, i+ks.ibgw+1, j+1, eps[i,j,iom].real, eps[i,j,iom].imag), file=fout)

        
        (matsiz1,matsiz2,nom_nil) = shape(eps)
        
        #if io.MatrixSelfEnergy:
        #    sc_p = zeros( (nirkp, ks.nbgw-ks.ibgw, ks.nbgw-ks.ibgw, len(fr.omega) ), dtype=complex )
        #else:
        #    sc_p = zeros( (nirkp, ks.nbgw-ks.ibgw, len(fr.omega) ), dtype=complex )
        t_times = zeros(8)
        for irk in range(nirkp):
            kl = array(kqm.kirlist[irk,:])/float(kqm.LCM)  # k in semi-cartesian form
            ik = kqm.k_ind[irk]   # index in all-kpoints, not irreducible
            jk = kqm.kqid[ik,iq]  # index of k-q
            jrk = kqm.kii_ind[jk]
            kpts = (ik,jk,iq)
            
            band_limits = (ks.ibgw, ks.nbgw, 0, ks.nbands_c-ks.ncg_c, 0, ks.ncg_c)
            rest = (isp, indggq, nmix, max_nmix, big_l, ql)
            minm0, minc0 = self.calc_minm(ik, iq, band_limits, 'selfe', strc, in1, latgen, kqm, ks, radf, pw, pb, core, rest, t_times, fout)
            (nst, nend, mst, mend, cst, cend)  = band_limits  # (outside_band_min,outside_band_max, inside_valence_band_first, inside_valence_band_last, core_first, core_last)
            nmdim = (band_limits[1]-band_limits[0])*(band_limits[3]-band_limits[2]+band_limits[5]-band_limits[4]) # (ie1 * ie2) size
            if minc0 is not None:
                minm = concatenate( (minc0, minm0), axis=1 )  # putting core and valence part together shape(minim)=(nb1,nb2,matsiz), where nb1=nbgw-ibgw, nb2=core+valence
            else:
                minm = minm0
            nb1,nb2,matsiz = shape(minm) # (oustside_bads, inside_bands, product_basis_index_
            minm2 = reshape(minm, (nb1*nb2,matsiz)) # now minm2[nb1*nb2,matsiz]
            cminm2 = conj(minm2)


            t_i1 = timer()
            if io.MatrixSelfEnergy:
                zmwm = zeros((nom_nil,nb2,nb1,nb1), dtype=complex)
            else:
                mwm = zeros((nom_nil,nb2,nb1), dtype=complex, order='F')  # we will sum over internal bands nb2. If self-energy is allowed to be off-diagonal, we will need nb0,nb1,nb2
            for iom_il in range(nom_nil):
                ## mwm[iom,ib1,ib2] = wkq * \sum_{im1,im2} minm2[nb1*nb2,im1]^* eps[im1,im2,iom] minm2[nb1*nb2,im2]
                mw  = la_matmul(cminm2,eps[:,:,iom_il]) # mw[nb1*nb2,matsiz] = minm2*[nb1*nb2,matsiz] * eps[matsiz,matsiz]
                if io.MatrixSelfEnergy:
                    mw3 = reshape(mw, (nb1,nb2,matsiz))
                    minm3 = reshape(minm2, (nb1,nb2,matsiz))
                    for ie2 in range(nb2):
                        zmwm[iom_il,ie2,:,:] = la_matmul(minm3[:,ie2,:],mw3[:,ie2,:].T)*wkq
                else:
                    mwm[iom_il,:,:] = reshape(wkq*sum(mw * minm2, axis=-1),(nb1,nb2)).T   # sum(mw[nb1*nb2,matsiz] * minm[nb1*nb2,matsiz],axis=1)
                    
                if (iq==0):
                    if io.MatrixSelfEnergy:
                        # mwm[iom,ib2,ib1] += coefs2 head[iom] + coefs1 \sum_{im} minm[ib1,ib2,im] eps2[im,iom] + coefs1 \sum_{im} minm[ib1,ib2,im]^* epsw1[im,iom]
                        for i1 in range(ks.nbgw-ks.ibgw):
                            ie1 = i1 + ks.ibgw
                            ie2 = i1 + ks.ibgw + ks.ncg_c
                            d2 = coefs2*head[iom_il] + coefs1*dot(minm[i1,ie2,:],epsw2[:,iom_il]) + coefs1*vdot(minm[i1,ie2,:],epsw1[:,iom_il])
                            zmwm[iom_il,ie2,i1,i1] += d2
                    else:
                        # mwm[iom,ib2,ib1] += coefs2 head[iom] + coefs1 \sum_{im} minm[ib1,ib2,im] eps2[im,iom] + coefs1 \sum_{im} minm[ib1,ib2,im]^* epsw1[im,iom]
                        for ie1 in range(ks.ibgw, ks.nbgw):
                            ie2 = ie1 + ks.ncg_c
                            d2 = coefs2*head[iom_il] + coefs1*dot(minm[ie1-ks.ibgw,ie2,:],epsw2[:,iom_il]) + coefs1*vdot(minm[ie1-ks.ibgw,ie2,:],epsw1[:,iom_il])
                            mwm[iom_il,ie2,ie1-ks.ibgw] += d2

            which_indices=[(i1,i1) for i1 in range(nb1)]
            if io.MatrixSelfEnergy:
                mdiag = zeros(nb1)
                for i1 in range(nb1):
                    mdiag[i1] = sum(abs(zmwm[:,:,i1,i1]))
                for i1 in range(nb1):
                    for i3 in range(i1+1,nb1):
                        moff_diag = (sum(abs(zmwm[:,:,i1,i3]))+sum(abs(zmwm[:,:,i3,i1])))
                        ratio = moff_diag/(mdiag[i1]+mdiag[i3])
                        if ratio > io.sigma_off_ratio:
                            #print(i1,i3, moff_diag, mdiag[i1], mdiag[i3], ratio)
                            which_indices.append([i1,i3])
                            which_indices.append([i3,i1])
                
                mwm = zeros((nom_nil,nb2,len(which_indices)), dtype=complex, order='F') 
                for i,(i1,i3) in enumerate(which_indices):
                    mwm[:,:,i] = zmwm[:,:,i1,i3]

                #print(len(which_indices), which_indices)
            
            
            t_i2 = timer()
            t_times[5] += t_i2-t_i1
            
            if io.save_mwm:
                save(ddir+'/mwm.'+str(iq)+'.'+str(irk), mwm)
                if io.MatrixSelfEnergy:
                    save(ddir+'/wich.'+str(iq)+'.'+str(irk), which_indices)
                
            t_i3 = timer()
            t_times[6] += t_i3-t_i2

            sig = mcmn.Compute_selfc_inside(iq, irk, ks.Ebnd[isp], mwm, fr, kqm, ks.ncg_c, core, Ul, fout, which_indices, io.MatrixSelfEnergy)

            if io.MatrixSelfEnergy:
                for i,(i1,i3) in enumerate(which_indices):
                    sigc[irk,i1,i3,:] += sig[i,:]
            else:
                for i in range(nb1):
                    sigc[irk,i,:] += sig[i,:]
                
            t_i4 = timer()
            t_times[7] += t_i4-t_i3

            PRINT=True
            if PRINT:
                #for ie1 in range(ks.nbgw-ks.ibgw):
                #    ie = ie1+ks.ibgw
                #    iom=0
                #    s = sig[ie1,iom]
                #    print('dSigc[iq=%3d,irk=%3d,ie=%3d,iom=%3d]=%16.12f%16.12f' % (iq,irk,ie,iom,s.real,s.imag), file=fout)

                Ndiag = nb1
                if io.MatrixSelfEnergy:
                    Ndiag = count_nonzero([x[0]==x[1] for x in which_indices])  # how many diagonal components of the self-energy
                
                for i in range(Ndiag):
                    ie1=i+ks.ibgw
                    iom=0
                    print('dSigc[iq=%3d,irk=%3d,ie1=%3d,ie3=%3d]=%16.12f%16.12f' % (iq,irk,ie1,ie1,sig[i,iom].real,sig[i,iom].imag), file=fout)
                    
                for i,(i1,i3) in enumerate(which_indices[Ndiag::2]):
                    ie1=i1+ks.ibgw
                    ie3=i3+ks.ibgw
                    iom=0
                    print('dSigc[iq=%3d,irk=%3d,ie1=%3d,ie3=%3d]=%16.12f%16.12f' % (iq,irk,ie1,ie3,sig[Ndiag+2*i,iom].real,sig[Ndiag+2*i,iom].imag), file=fout)
            
        print('## calc_selfc: t(prep_minm)[iq=%-3d]=%14.9f' % (iq,t_times[0]), file=fout)
        print('## calc_selfc: t(minm)    [iq=%-3d] =%14.9f' % (iq,t_times[1]), file=fout)
        print('## calc_selfc: t(minm*sV) [iq=%-3d] =%14.9f' % (iq,t_times[2]), file=fout)
        print('## calc_selfc: t(minc)    [iq=%-3d] =%14.9f' % (iq,t_times[3]), file=fout)
        print('## calc_selfc: t(minc*sV) [iq=%-3d] =%14.9f' % (iq,t_times[4]), file=fout)
        print('## calc_selfc: t(mwm)     [iq=%-3d] =%14.9f' % (iq,t_times[5]), file=fout)
        print('## calc_selfc: t(enk      [iq=%-3d] =%14.9f' % (iq,t_times[6]), file=fout)
        print('## calc_selfc: t(convol)  [iq=%-3d] =%14.9f' % (iq,t_times[7]), file=fout)
        return sigc
