#from scipy import *
from numpy import *
from scipy import optimize
from numpy import linalg
from scipy import interpolate
import itertools

#from inout import *
import for_tetrahedra as ft
import fnc
import for_pade as fpade

from inout import InOut
from cmn import Ry2H,H2eV
#from pylab import *
from pylab import *

def Give2TanMesh(x0,L,Nw):
    def fun(x,x0,L,Nw):
        "x[0]=d, x[1]=w"
        d=x[0]
        w=x[1]
        #print 'd=', d, 'w=', w
        return array([L-w/tan(d), x0-w*tan(pi/(2*Nw)-d/Nw) ])
    tnw = tan(pi/(2*Nw))
    if x0 > L*0.25*Nw*tnw**2:
        x0 = L*0.25*Nw*tnw**2-1e-15
    xi=x0/L
    d0 = Nw/2.*(tnw-sqrt(tnw**2 - 4*xi/Nw))
    w0 = L*d0
    sol=optimize.root(fun, [d0,w0], args=(x0,L,Nw) )
    (d,w) = sol.x
    xt = linspace(0.0,1.0,2*Nw)*(pi-2*d) - pi/2 + d
    om = w*tan(xt)
    dom = (w*(pi-2*d)/(2.*Nw))/cos(xt)**2
    return om,dom


def calc_Fermi(Ebnd, kqm_atet, kqm_wtet, nval, nspin):
    eDos = 1.0
    if nspin==1 and (nval % 2 == 0):
        # just simply counting bands, and setting EF in the middle
        nvm   = int( nval/2.0 + 0.51 )
        evbm  = max( Ebnd[:,nvm-1] )
        ecbm  = min( Ebnd[:,nvm])
        EF  = 0.5*(evbm+ecbm)
        ocint = ft.idos(EF, Ebnd, kqm_atet, kqm_wtet)*2.0/nspin
        if ecbm >= evbm and abs(ocint-nval) < 1e-6:
            Eg = ecbm - evbm
            eDos = 0
    if eDos != 0: # the simple method did not succeed
        evbm = min(Ebnd.ravel()) # minimum of energy
        ecbm = max(Ebnd.ravel()) # maximum of energy
        ocmax = sum([ft.idos(ecbm, Ebnd, kqm_atet, kqm_wtet) for isp in range(nspin)])*2.0/nspin
        if ocmax <= nval:
            print('ERROR in fermi: not enough bands : %s%-10.4f %s%-10.2f %s%-10.2f' % ('emax=', ecbm, 'ocmax=', ocmax, 'nel= ', nval ))
            sys.exit(1)
        
        EF = optimize.brentq(lambda x: sum([ft.idos(x, Ebnd, kqm_atet, kqm_wtet) for isp in range(nspin)])*2.0/nspin-nval, evbm, ecbm)
        eDos = sum([ft.dostet(EF, Ebnd, kqm_atet, kqm_wtet) for isp in range(nspin)])*2.0/nspin
        
        # For insulator (including semiconductor, set fermi energy as the middle of gap
        if eDos < 1e-4:
            evbm = max( [x for x in Ebnd.ravel() if x < EF] )
            ecbm = min( [x for x in Ebnd.ravel() if x > EF] )
            EF = (evbm + ecbm)/2.
            Eg = ecbm - evbm
        else: 
            Eg = -eDos
            evbm, ecbm = EF, EF
    return (EF, Eg, evbm, ecbm, eDos)
 
def cart2int(klist,strc,latgen):
    """ Cartesian to integer representation of k-points"""
    if latgen.ortho:
        rbas = zeros((3,3))
        alat = array([strc.a, strc.b, strc.c])
        for j in range(3):
            rbas[j,:] = latgen.rbas[j,:]/alat[j] 
        iklist = dot(klist,rbas)
        return iklist
    else:
        return klist

def Compute_selfc_inside(iq, irk, bands, mwm, fr, kqm, ncg_c, core, Ul, fout, which_indices, MatrixSelfEnergy=False, PRINT=False):
    #def freqconvl(iom, enk, mwm, omega, womeg):
    #    """ We are computing frequency convolution
    #            sigma(iom) = -1/beta \sum_{iOm} mwm[iOm]/(iom-eps+iOm)
    #        Because mwm[-iOm]=mwm[iOm] and at T=0, we can rewrite
    #            sigma(iom) = 1/pi Integrate[ mwm[Om]*(eps-iom)/((eps-iom)^2 + Om^2) , {Om,0,Infinity} ]
    #        Finally, we notice that
    #                              Integrate[ 1/((eps-iom)^2 + Om^2) , {Om,0,Infinity} ] = pi*sign(eps)/(2*(eps-iom))
    #        therefore
    #            sigma(iom) = (eps-iom)/pi Integrate[ (mwm[Om]-mwm[iom])/((eps-iom)^2 + Om^2) , {Om,0,Infinity} ] + mwm[iom] * sign(eps)/2
    #    """
    #    if FORT:
    #        return fnc.fr_convolution(iom+1, enk, mwm, omega, womeg)
    #    else:
    #        eps_om = enk - omega[iom]*1j
    #        sc0 = sum( (mwm[:]-mwm[iom]) * womeg / (omega**2 + eps_om**2) )
    #        return eps_om*sc0/pi + mwm[iom]/pi * atan(omega[-1]/eps_om)
        
    (nom, nb2, nb1) = shape(mwm)
    omega, womeg = fr.omega, fr.womeg
    
    ik = kqm.k_ind[irk]   # index in all-kpoints, not irreducible
    jk = kqm.kqid[ik,iq]  # index of k+q
    jrk = kqm.kii_ind[jk] # index of k+q in irreducible list

    #print 'Compute_selfc_inside:shape(bands)=', shape(bands), 'jrk=', jrk, 'nb2=', nb2
    isp=0
    enk = zeros( nb2 )
    for icg in range(ncg_c):
        iat,idf,ic = core.corind[icg,:3]
        enk[icg] = core.eig_core[isp][iat][ic]
    enk[ncg_c:nb2] = bands[jrk,:(nb2-ncg_c)]   # careful, we need G(k+q,:) here, not G(k,:)
    
    if Ul is not None:
        (nl,nom) = shape(Ul)
        # Cnl[l,ie2,iom] = 1/pi*Integrate[ Ul[l,iOm]*(enk[ie2]-iom)/((enk[ie2]-iom)^2 + Om^2),{Om,0,Infinity}]
        Cnl = fnc.many_fr_convolutions(enk, Ul, omega, womeg) # frequency convolution of all svd functions
        Cnl = reshape(Cnl,(nl*nb2,nom))
        mwm2 = zeros( (nb1, nl*nb2), dtype=complex )
        for ie1 in range(nb1):
            mwm2[ie1,:] = reshape( mwm[:,:,ie1], nl*nb2 )
        sc_p = dot(mwm2,Cnl)
    else:
        sc_p = zeros( (nb1,nom), dtype=complex )
        Ndiag = nb1
        Nall = len(which_indices)
        if MatrixSelfEnergy:
            Ndiag = count_nonzero([x[0]==x[1] for x in which_indices])  # how many diagonal components of the self-energy
            
        if fr.iopfreq!=4 or len(fr.omega_precise)==len(fr.omega):
            # No interpolation needed
            for i in range(Ndiag): # over all diagonals
                ie1 = which_indices[i][0]
                #S = mwm[:,:,i].real # (ie1,ie1) This is because we combined +iOm and -iOm, which makes W(iOm) hermitian, hence on diagonal is 2*Real
                sc_p[i,:] = fnc.some_fr_convolutions(mwm[:,:,i].real, enk, fr.omega, fr.womeg, True)
                
                #savetxt('sgc.'+str(ie1), vstack((fr.omega,sc_p[i,:].real,sc_p[i,:].imag)).T)
            for i in range(Ndiag,Nall,2): # over off-diagonal components. We know that (i1,i3) is followed by (i3,i1)
                i1,i3 = which_indices[i]
                S = 0.5*(mwm[:,:,i]+mwm[:,:,i+1].conj())      # (i1,i3)
                sc_p[i,  :] = fnc.some_fr_convolutionsc(S, enk, fr.omega, fr.womeg, True)
                sc_p[i+1,:] = fnc.some_fr_convolutionsc(S.conj(), enk, fr.omega, fr.womeg, True)
                if False: # This term is actually zero because P is real
                    R = 0.5*(mwm[:,:,i]-mwm[:,:,i+1].conj())*1j   # (i1,i3)
                    sc_p[i,  :] +=  fnc.off_fr_convolutions(R, enk, fr.omega, fr.womeg, False) # This term is actually zero because P is real
                    sc_p[i+1,  :] +=  fnc.off_fr_convolutions(R.conj(), enk, fr.omega, fr.womeg, False) # This term is actually zero because P is real
                
                #savetxt('sgc.'+str(i1)+'.'+str(i3), vstack((fr.omega,sc_p[i,:].real,sc_p[i,:].imag, sc_p[i+1,:].real,sc_p[i+1,:].imag)).T)
        else:
            # Here we are requesting the precise mesh for convolution, which has io.iopMultiple>1 - times denser mesh
            for_asymptotic = (omega**2+1.0)
            for_asymptotic_precise = (fr.omega_precise**2+1.0)
            zeros_nb2 = zeros(nb2, dtype=complex)
            mwmp = zeros( (len(fr.omega_precise),nb2), dtype=float, order='F')
            for i in range(Ndiag):
                ie1 = which_indices[i][0]
                for ie2 in range(nb2):
                    # Interpolating W on more dense mesh for frequency convolution
                    # Note that we interpolate W*(om^2+1), which goes to constant at large om, and is easier to interpolate properly
                    mwmr = interpolate.CubicSpline(omega, for_asymptotic*mwm[:,ie2,i].real, bc_type=((2,0),(1,0)), extrapolate=True)
                    mwmp[:,ie2] = mwmr(fr.omega_precise)/for_asymptotic_precise
                sc_p[i,:] = fnc.some_fr_convolutions3(mwmp.real, mwm[:,:,i].real, enk, fr.omega_precise, fr.womeg_precise, omega, True)

                #savetxt('gxx.'+str(ie1), vstack((fr.omega,sc_p[i,:].real,sc_p[i,:].imag)).T)
                
            mwmp = zeros( (len(fr.omega_precise),nb2), dtype=complex, order='F')
            for i in range(Ndiag,Nall,2): # over off-diagonal components. We know that (i1,i3) is followed by (i3,i1)
                i1,i3 = which_indices[i]
                Sc = 0.5*(mwm[:,:,i]+mwm[:,:,i+1].conj()) # this is S(i1,i3)+S(i3,i1)^* 
                for ie2 in range(nb2):
                    #S = 0.5*(mwm[:,ie2,i]+mwm[:,ie2,i+1].conj()) # this is S(i1,i3)+S(i3,i1)^*
                    mwmr = interpolate.CubicSpline(omega, for_asymptotic*Sc[:,ie2].real, bc_type=((2,0),(1,0)), extrapolate=True)
                    mwmi = interpolate.CubicSpline(omega, for_asymptotic*Sc[:,ie2].imag, bc_type=((2,0),(1,0)), extrapolate=True)
                    mwmp[:,ie2] = (mwmr(fr.omega_precise) + mwmi(fr.omega_precise)*1j)/for_asymptotic_precise
                sc_p[i,  :] = fnc.some_fr_convolutions3c(mwmp, Sc, enk, fr.omega_precise, fr.womeg_precise, omega, True)
                sc_p[i+1,:] = fnc.some_fr_convolutions3c(mwmp.conj(), Sc.conj(), enk, fr.omega_precise, fr.womeg_precise, omega, True)

                #savetxt('gxx.'+str(i1)+'.'+str(i3), vstack((fr.omega,sc_p[i,:].real,sc_p[i,:].imag, sc_p[i+1,:].real,sc_p[i+1,:].imag)).T)
    return sc_p


def Compute_quasiparticles(bands, Ebnd, sigc, sigx, Vxct, omega, io, isp, fout, PRINT):
    PRINT_DEBUG = False
    #(iop_ac, iop_es, iop_gw0, npar_ac, iop_rcf) = (io.iop_ac,io.iop_es,io.iop_gw0,io.npar_ac,io.iop_rcf)
    def f_unc(x, *argv):
        return fpade.pd_funval(x,argv)
    def f_jacob(x, *argv):
        return fpade.pd_jacoby(x,argv)
    def AnalyticPade(omega, sig, n, i1,i3, iop_ac=0, npar_ac=4):
        # Here we can do both pade or pole fitting
        sigt = (sig[i1,i3,:]+sig[i3,i1,:])/2.
        if iop_ac==0:
            z_p, y_p = omega[:n]*1j, sigt[:n]   # first n Matsuara points are used to continue
            apar = fpade.padecof(y_p, z_p)
        elif iop_ac==1:
            ## Here we fit all values of self-energy at Matsubara pointz iw==z to the following rational function
            #          (c[1] + c[2] z)
            #  f(z) = ----------------------
            #          1 + c[3] z + c[4] z^2
            #  which is a rational function with two poles.
            #  It is also equivalent to pade of the level 4, i.e.,
            #  f(z) = a[1]/(1+a[2]z/(1+a[3]z/(1+a[4]z))), where non-linear relation between a[i] and c[i] exists
            #  
            iwpa = arange(npar_ac)*int((len(omega)-1.)/(npar_ac-1.))  # linear mesh of only a few points, equidistant.
            iwpa[-1] = len(omega)-1                             # always take the last point
            cx = zeros(len(iwpa), dtype=complex)  # imaginary frequency at selected points
            cy = zeros(len(iwpa), dtype=complex)  # self-energy at selected points
            for ip,iw in enumerate(iwpa):
                cx[ip] = omega[iw]*1j
                cy[ip] = sigt[iw]
            apar = fpade.init_c(cx, cy)           # just a way to find a good starting point for minimization
            n = len(apar)                         # how many complex coefficints we want to fit
            anl = hstack( (apar.real,apar.imag) ) # we stack these complex coefficients (starting point coefficients) into real array
            
            x = hstack( (-omega.imag, omega.real) )  # take all frequency as x values : x = i*omega
            y = hstack( ( sigt.real, sigt.imag) )    # take all self-energy points as y values: y = [real,imag]
            # now fit all self-energy points to rational function
            #   sigma(iom) = P(iom)/Q(iom), where
            #   P(iom) = sum_{k=0,n  } c_{k}   (iom)^k
            #   Q(iom) = sum_{k=1,n+1} c_{k+n} (iom)^k
            # and determine c_k by minimization of chi2 using Levenberg-Marquardt algorithm.
            popt, pcov = optimize.curve_fit( f_unc, x, y, p0=anl, jac=f_jacob, method='lm')
            apar = popt[:n] + popt[n:]*1j # these are ck coefficients
            z_p = omega[:n]*1j
        else:
            print('ERROR, unkown analytic continuation iop_ac=', iop_ac, 'it should be 0 for Pade or 1 for pole fitting')
            sys.exit(0)
        return (z_p, apar)
    
    def AnalyticEvaluate(enk, z_p, apar, iop_ac=0):
        if iop_ac==0:
            return fpade.padeg( enk, z_p, apar)
        else:
            yc,dyc = fpade.pd_funvalc(enk, apar)
            return yc

    if io.MatrixSelfEnergy:
        #save('sigx', sigx)
        #save('Vxct', Vxct)
        #save('bands', bands)
        #save('sigc', sigc)
        #save('omega', omega)
        #save('twhich_indices', twhich_indices)
        
        (nirkp, nbnd, nbnd2, nom) = shape(sigc)
        twhich_indices=[]
        for irk in range(nirkp):
            which_indices=[(i1,i1) for i1 in range(nbnd)]
            for i in range(nbnd):
                for j in range(i+1, nbnd):
                    ratio = sum(abs(sigc[irk,i,j,:])+abs(sigc[irk,j,i,:]))/sum(abs(sigc[irk,i,i,:])+abs(sigc[irk,j,j,:]))
                    if ratio> 0.1*io.sigma_off_ratio:
                        which_indices.append([i,j])
                        which_indices.append([j,i])
            twhich_indices.append(which_indices)
        if PRINT_DEBUG:
            for irk in range(nirkp):
                Ndiag = nbnd
                for i in itertools.chain(range(Ndiag), range(Ndiag,len(twhich_indices[irk]),2)):
                    i1,i3 = twhich_indices[irk][i]
                    name = 'sigc.'+str(irk)+'.'+str(i1)+'.'+str(i3)
                    if i1==i3:
                        name = 'sigc.'+str(irk)+'.'+str(i1)
                    savetxt(name, vstack((omega*H2eV,sigc[irk,i1,i3,:].real*H2eV,sigc[irk,i1,i3,:].imag*H2eV)).T)
        
        Emax = max(bands.ravel())
        romega = linspace(-1.1*Emax,Emax*1.1,100)
    else:
        (nirkp, nbnd, nom) = shape(sigc)
    
    eqp    = zeros(shape(bands))
    eqp_im = zeros(shape(bands))
    if (PRINT): print('Quasiparticle energies in eV', file=fout)
    lwarn = True
    for irk in range(nirkp):
        if io.MatrixSelfEnergy:
            which_indices = twhich_indices[irk]
            Ndiag = nbnd
            enk = bands[irk,:]
            ###################################
            n = min(32, len(omega))
            qsig = [[] for i in range(len(which_indices))]
            z_nk = [[] for i in range(Ndiag)]
            if PRINT_DEBUG:
                rsig = [[] for i in range(len(which_indices))]
            for i in range(Ndiag):
                z_p,apar = AnalyticPade(omega, sigc[irk], n, *which_indices[i], io.iop_ac, io.npar_ac)
                qsig[i] = AnalyticEvaluate( enk, z_p, apar, io.iop_ac).real
                dqsig = (AnalyticEvaluate( enk+1e-3, z_p, apar, io.iop_ac)-AnalyticEvaluate( enk-1e-3, z_p, apar, io.iop_ac)).real/2e-3
                if PRINT_DEBUG:
                    rsig[i] = AnalyticEvaluate(romega, z_p, apar, io.iop_ac)
                    savetxt('Sqp.'+str(irk)+'.'+str(i), vstack((romega*H2eV,rsig[i].real*H2eV,rsig[i].imag*H2eV)).T)
                    
                # Here we decided that the off-diagonal self-energy is not expanded in frequency to the first order, i.e.,
                #   Sigma(omega) = Sigma(enk) + dSigma/domega (omega-enk),
                # with dSigma/domega diagonal only, which than becomes
                #  Sigma(omega) = Sigma(enk) + (1-1/z_nk) (omega-enk)
                z_nk[i] = 1/(1-dqsig)
                
                for ie in range(nbnd):
                    if (z_nk[i][ie] > 1.0 or z_nk[i][ie] < 0):
                        if (lwarn):
                            print('WARNING : Z_nk at band i=', i, ' and energy e_k=', enk[ie]*H2eV, 'eV is unphysical', z_nk[i][ie], 'irk=', irk, 'ie=', ie, 's_nk=', qsig[i][ie]*H2eV, file=fout)
                            lwarn = False  # we will warn only once
                        z_nk[i][ie] = 1.0
            
            for i in range(Ndiag,len(which_indices),2):
                z_p,apar = AnalyticPade(omega, sigc[irk], n, *which_indices[i], io.iop_ac, io.npar_ac)
                qsig[i] = AnalyticEvaluate( enk, z_p, apar, io.iop_ac).real
                qsig[i+1] = qsig[i]
                if PRINT_DEBUG:
                    rsig[i] = AnalyticEvaluate(romega, z_p, apar, io.iop_ac)
                    rsig[i+1] = rsig[i]
                    i1,i3 = which_indices[i]
                    savetxt('Sqp.'+str(irk)+'.'+str(i1)+'.'+str(i3), vstack((romega*H2eV,rsig[i].real*H2eV,rsig[i].imag*H2eV)).T)

            ## We are solving the following quasiparticle equation:
            ##   omega-Enk0-Sigma_{xc}(omega)+v_xc = 0  around omega ~ enk
            ## here enk is the self-consistent band, and Enk0 is the LDA band
            ## Hence for component enk_i, we can expand
            ##   (omega-enk_i)*I + (enk_i*I-Enk0) + Sigma_{xc}(enk_i)+ dSigma_{xc}(enk_i)/domega (omega-enk_i) + v_xc = 0
            ## Define (I - dSigma_{xc}(enk_i)/domega) = z_i^{-1}
            ##   z_i^{-1/2} * (omega-enk_i) * z_i^{-1/2} + (enk_i*I-Enk0) + Sigma_{xc}(enk_i) + v_xc = 0
            ##   omega-enk_i = z_i^{1/2} ( Enk0 - enk_i*I + Sigma_{xc}(enk_i)-v_{xc}) z_i^{1/2}
            ## The effective Hamiltonian deltaH = z_i^{1/2} ( Enk0 - enk_i*I + Sigma_{xc}(enk_i)-v_{xc}) z_i^{1/2}
            ## is valid only for solution enk_i, which has eigenvector close to (0,0,....,1,...0) with 1 and i.
            ## 
            Id = identity(nbnd)
            enk0 = Ebnd[irk,:]
            Enk0 = diag(enk0)
            for ie in range(nbnd):
                dH = Enk0-Id*enk[ie]+sigx[irk,:,:]-Vxct[irk,:,:]
                for i,(i1,i3) in enumerate(which_indices):
                    dH[i1,i3] += qsig[i][ie]

                if io.iop_es == -1 and io.iop_gw0==1:  # This is the default self-consistent GW0, in which we apparently neglect z. We just follow gap2 code convention here.
                    # Z is set to unity
                    pass
                else:
                    sz = [sqrt(z_nk[i][ie]) for i in range(Ndiag)]
                    #dH[i,j] = sz[i]*dH[i,j]*sz[j]
                    dH *= tensordot(sz,sz,axes=0)
                
                w, v = linalg.eigh(dH)
                v0 = zeros(nbnd); v0[ie]=1
                ii = argmin([linalg.norm(abs(v[:,i])-v0) for i in range(nbnd)])
                #w[ii], v[:,ii]
                eqp[irk,ie] = enk[ie] + w[ii]
                
                if io.iop_es == -1 and io.iop_gw0==1:  # This is the default self-consistent GW0, in which we apparently neglect z. We just follow gap2 code convention here.
                    # Z is set to unity
                    w_expected = (enk0[ie]-enk[ie]+qsig[ie][ie]+sigx[irk,ie,ie]-Vxct[irk,ie,ie]).real
                else:
                    w_expected = z_nk[ie][ie]*(enk0[ie]-enk[ie]+qsig[ie][ie]+sigx[irk,ie,ie]-Vxct[irk,ie,ie]).real
                
                print('eqp[irk=%3d,ie=%3d]=%16.10f and enk0=%16.10f enk=%16.10f znk=%8.6f dw_off=%13.6f w=%12.6f v[ie,ii]=%10.6f' % (irk,ie+1,eqp[irk,ie]*H2eV,Ebnd[irk,ie]*H2eV,enk[ie]*H2eV,z_nk[ie][ie],(w[ii]-w_expected)*H2eV,w[ii]*H2eV,abs(v[ie,ii])), file=fout)
            
            if PRINT_DEBUG:
                Gqp = zeros((len(romega),nbnd,nbnd), dtype=complex)
                for ie in range(nbnd):
                    Gqp[:,ie,ie] = romega-enk[ie]
                for iw in range(len(romega)):
                    Gqp[iw,:,:] -= sigx[irk,:,:]-Vxct[irk,:,:]
                for i,(i1,i3) in enumerate(which_indices):
                    Gqp[:,i1,i3] -= rsig[i]
                for iw in range(len(romega)):
                    Gqp[iw] = linalg.inv(Gqp[iw])
                for i in itertools.chain(range(Ndiag),range(Ndiag,len(which_indices),2)):
                    i1,i3 = which_indices[i]
                    name = 'Gqp.'+str(irk)+'.'+str(i1)+'.'+str(i3)
                    if i1==i3:
                        name = 'Gqp.'+str(irk)+'.'+str(i1)
                    savetxt(name, vstack((romega*H2eV, Gqp[:,i1,i3].real/H2eV, Gqp[:,i1,i3].imag/H2eV)).T)
                del Gqp

                
        else:
            for ie in range(nbnd):
                enk0 = Ebnd[irk,ie]
                vxc_nk = Vxct[irk,ie,ie].real
                enk = bands[irk,ie]
                # enk is the energy at which the self-energy is calculated
                debug = '.0.0' if (irk==0 and ie==0) else ''
                sig,dsig,apar = AnalyticContinuation(io.iop_ac, omega, sigc[irk,ie,:], io.npar_ac, enk, fout, io.iop_rcf, debug)
                # quasiparticle residue, but not at zero freqeucny, but at the energy of the band!
                z_nk = 1/(1-dsig.real)
                # self-energy at the energy of the band. This is what they believe is the best quasiparticle approximation
                s_nk = sig.real + sigx[irk,ie]
                if (z_nk > 1.0 or z_nk < 0):
                    if (lwarn):
                        print('WARNING : Z_nk at energy e_k=', enk*H2eV, 'eV is unphysical', z_nk, 'irk=', irk, 'ie=', ie, 's_nk=', s_nk*H2eV, 'ds/dw=', dsig.real, file=fout)
                        lwarn = False  # we will warn only once
                    z_nk = 1.0
                    dsig = 0.0
                if io.iop_es == -1: # self-consistent GW0
                    if io.iop_gw0==1: # this is default
                        delta = s_nk - vxc_nk + enk0 - enk  # this is used in self-consistent GW0
                    elif io.iop_gw0==2:
                        delta = z_nk*(s_nk-vxc_nk) + enk0 - enk
                    else:
                        delta = z_nk*(s_nk-vxc_nk + enk0 - enk)
                elif io.iop_es == 0: # this is used in G0W0
                    delta = z_nk*(s_nk-vxc_nk)
                else:
                    print('Not implemented here', file=fout)
                    sys.exit(1)
                
                eqp[irk,ie] = enk + delta
                eqp_im[irk,ie] = sig.imag*z_nk

                #print >> fout, 'iop_eps=%2d iop_gw0=%2d delta=%16.10f' % (iop_es, iop_gw0, delta)
                #print 'eqp['+str(irk)+','+str(ie)+']='+str(eqp[irk,ie]), 'and enk=', enk0
                if PRINT:
                    print('eqp[irk=%3d,ie=%3d]=%16.10f and enk0=%16.10f enk=%16.10f znk=%10.8f snk=%16.10f vxc=%16.10f' % (irk,ie,eqp[irk,ie]*H2eV,enk0*H2eV,enk*H2eV,z_nk,s_nk.real*H2eV, vxc_nk.real*H2eV), file=fout)
    return (eqp, eqp_im)
    
def Band_Analys(bande, EF, nbmax, titl, kqm, fout):
    bands = copy(bande[:,:nbmax])
    nirkp = shape(bands)[0]
    print('-'*60, file=fout)
    print('  '+titl+' Band Analysis', file=fout)
    print('-'*60, file=fout)
    print('  Range of bands considered: %5d %5d' % (0,nbmax), file=fout)
    print('  EFermi[eV]= %10.4f' % (EF*H2eV,), file=fout)
    
    if max(bands.ravel()) < EF or min(bands.ravel())>EF:
        print('WARNING from bandanaly:  - Fermi energy outside the energy range of bande!', file=fout)
        print('minimal energy=', min(bands)*H2eV, 'max energy=', max(bands)*H2eV, 'EF=', EF*H2eV, file=fout)

    
    nocc_at_k = [len([x for x in bands[ik,:] if x<EF]) for ik in range(nirkp)] # how many occuiped bands at each k-point
    nomax = max(nocc_at_k)-1                 # index of the last valence band
    numin = min(nocc_at_k)                   # index of the first conduction band

    ikvm = argmax(bands[:,nomax]) # maximum of the valence band
    ikcm = argmin(bands[:,numin]) # minimum of the conduction band

    Qmetal =  (nomax >= numin)
    if Qmetal:
        print(' Valence and Conductance bands overlap: metallic!', file=fout)
    if Qmetal:
        evbm = EF
    else:
        evbm = bands[ikvm,nomax]
    print('  Band index for VBM and CBM=%4d %4d' % (nomax+1, numin+1), file=fout)
    
    bands = (bands - evbm)*H2eV
        
    egap1 =     bands[ikcm,numin] - bands[ikvm,nomax]  # the smallest indirect gap between KS bands
    egap2 = min(bands[ikvm,numin:])-bands[ikvm,nomax]  # the direct gap starting from valence band
    egap3 = bands[ikcm,numin] - max(bands[ikcm,:(nomax+1)])# the direct gap starting from conduction band
    #print 'egap1=', egap1, 'egap=', egap2, 'egap3=', egap3
    
    if ikvm==ikcm: # direct gap
        print(':BandGap_'+titl+' = %12.3feV' % egap1, file=fout)
        kp = kqm.kirlist[ikvm,:]/float(kqm.LCM)
        print(('  Direct gap at k=  %8.3f%8.3f%8.3f') % tuple(kp), 'ik='+str(ikvm+1), file=fout)
    else:
        print((':BandGap_'+titl+' = %12.3f%12.3f%12.3f eV') % (egap1,egap2,egap3), file=fout)
        kv = kqm.kirlist[ikvm,:]/float(kqm.LCM)
        kc = kqm.kirlist[ikcm,:]/float(kqm.LCM)
        print('  Indirect gap, k(VBM)=%8.3f%8.3f%8.3f' % tuple(kv), 'ik='+str(ikvm+1), file=fout)
        print('                k(CBM)=%8.3f%8.3f%8.3f' % tuple(kc), 'ik='+str(ikcm+1), file=fout)
    print('Range of each band with respect to VBM (eV):', file=fout)
    print(('%5s'+'%12s'*3) % ('n ','Bottom','Top','Width'), file=fout)
    for i in range(shape(bands)[1]):
        ebmin = min(bands[:,i])
        ebmax = max(bands[:,i])
        print('%5d%12.3f%12.3f%12.3f' % (i+1, ebmin, ebmax, ebmax-ebmin), file=fout)
    return (nomax,numin)



def padeMatrix(z,f,N,verbose=False):
    """
    Input variables:
    z       - complex. points in the complex plane.
    f       - complex. Corresponding (Green's) function in the complex plane. 
    N       - int. Number of Pade coefficients to use
    verbose - boolean. Determine if to print solution information
    Returns the obtained Pade coefficients.
    """
    # number of input points
    M = len(z)
    r = N/2
    y = f*z**r
    A = ones((M,N),dtype=complex)
    for i in range(M):
        A[i,:r] = z[i]**(arange(r))
        A[i,r:] = -f[i]*z[i]**(arange(r)) 
    # Calculated Pade coefficients
    # rcond=-1 means all singular values will be used in the solution.
    sol = linalg.lstsq(A, y, rcond=-1)
    # Pade coefficents
    x = sol[0]
    if verbose:
        print('error_2= ',linalg.norm(dot(A,x)-y))
        print('residuals = ', sol[1])
        print('rank = ',sol[2])
        print('singular values / highest_singlular_value= ',sol[3]/sol[3][0])
    return x

def epade(z,x):
    """
    Input variables:
    z - complex. Points where continuation is evaluated.
    x - complex. Pade approximant coefficient.
    
    Returns the value of the Pade approximant at the points z.
    """
    r = len(x)/2
    numerator = zeros(len(z),dtype=complex256)
    denomerator = zeros(len(z),dtype=complex256)
    for i in range(r):
        numerator += x[i]*z**i
        denomerator += x[r+i]*z**i
    denomerator += z**r
    return numerator/denomerator

def AnalyticContinuation(iop_ac,omg,sc,npar,enk, fout, iop_rcf=0.5, debug=''):
    # Analytic continuation of self-energy, and its evaluation at energy enk. Both the value and derivat
    # iop_ac==0  -- Thiele's reciprocal difference method as described in
    #       H. J. Vidberg and J. W. Serence, J. Low Temp. Phys. 29, 179 (1977). This is usual Pade, known in DMFT
    # iop_ac==1  -- Rojas, Godby and Needs (PRL 74, 1827 (1996). This is just a fit to imaginary axis data
    #
    #
    # We first take only a few values of self-energy and frequency, which will be used for Pade
    def f_unc(x, *argv):
        return fpade.pd_funval(x,argv)
    
    def f_jacob(x, *argv):
        return fpade.pd_jacoby(x,argv)
    #
    if iop_ac==0 and abs(enk)/Ry2H < iop_rcf:
        # n = npar-1
        n = min(32, len(omg))
        z_p = omg[:n]*1j
        y_p = sc[:n]
        
        apar = fpade.padecof(y_p, z_p)
        
        yc = fpade.padeg([enk], z_p, apar)
        dyc = (fpade.padeg([enk+1e-3], z_p, apar)-fpade.padeg([enk-1e-3], z_p, apar))/2e-3
            
        #print >> fout, 'classic Pade: s['+str(enk*H2eV)+']='+str(yc[0]*H2eV)
    elif iop_ac==1 or (iop_ac==0 and abs(enk)/Ry2H > iop_rcf):
        ## Here we fit all values of self-energy at Matsubara pointz iw==z to the following rational function
        #          (c[1] + c[2] z)
        #  f(z) = ----------------------
        #          1 + c[3] z + c[4] z^2
        #  which is a rational function with two poles.
        #  It is also equivalent to pade of the level 4, i.e.,
        #  f(z) = a[1]/(1+a[2]z/(1+a[3]z/(1+a[4]z))), where non-linear relation between a[i] and c[i] exists
        #  
        iwpa = arange(npar)*int((len(omg)-1.)/(npar-1.))  # linear mesh of only a few points, equidistant.
        iwpa[-1] = len(omg)-1                             # always take the last point
        cx = zeros(len(iwpa), dtype=complex)  # imaginary frequency at selected points
        cy = zeros(len(iwpa), dtype=complex)  # self-energy at selected points
        for ip,iw in enumerate(iwpa):
            cx[ip] = omg[iw]*1j
            cy[ip] = sc[iw]
        apar = fpade.init_c(cx, cy)           # just a way to find a good starting point for minimization
        n = len(apar)                         # how many complex coefficints we want to fit
        anl = hstack( (apar.real,apar.imag) ) # we stack these complex coefficients (starting point coefficients) into real array

        x = hstack( (-omg.imag, omg.real) )  # take all frequency as x values : x = i*omg
        y = hstack( ( sc.real, sc.imag) )    # take all self-energy points as y values: y = [real,imag]
        # now fit all self-energy points to rational function
        #   sigma(iom) = P(iom)/Q(iom), where
        #   P(iom) = sum_{k=0,n  } c_{k}   (iom)^k
        #   Q(iom) = sum_{k=1,n+1} c_{k+n} (iom)^k
        # and determine c_k by minimization of chi2 using Levenberg-Marquardt algorithm.
        popt, pcov = optimize.curve_fit( f_unc, x, y, p0=anl, jac=f_jacob, method='lm')
        apar = popt[:n] + popt[n:]*1j # these are ck coefficients
        
        # Evaluate self-energy at energy of the band ek
        yc,dyc = fpade.pd_funvalc([enk],apar)
            
        #print >> fout, 'modified Pade: s['+str(enk*H2eV)+']='+str(yc[0]*H2eV)
    else:
        s0 = sc[0].real
        dsdw = polyfit([0,omg[0],omg[1]], [0.0,sc[0].imag,sc[1].imag], 1)[0]
        yc = [s0 + dsdw*enk]
        dyc = [dsdw]
        apar = [s0,dsdw]
        #print >> fout, 'Simple quasiparticle approximation: s['+str(enk*H2eV)+']='+str(yc[0]*H2eV)
        
    #print >> fout, 'sig[e=%10.6f]=%10.6f %10.6f' % (enk,yc.real, yc.imag)
    #print ' cx, cy, apar'
    #for ip in range(len(iwpa)):
    #    print '%21.16f%21.16f  %20.16f%20.16f  %20.16f%20.16f' % (cx[ip].real, cx[ip].imag, cy[ip].real, cy[ip].imag, apar[ip].real, apar[ip].imag)
    if debug:
        romega = hstack( (-omg[::-1],omg) )
        if iop_ac==0 and abs(enk)/Ry2H < iop_rcf:
            #print 'iop_ac=', iop_ac, 'abs(enk)/Ry2H=', abs(enk)/Ry2H, 'iop_rcf=', iop_rcf
            if True:
                yre = fpade.padeg(romega, z_p, apar)
                yim = fpade.padeg(romega*1j, z_p, apar)
            else:
                yre = epade(romega, apar)
                yim = epade(romega*1j, apar)
        elif iop_ac==1 or (iop_ac==0 and abs(enk)/Ry2H > iop_rcf):
            yre,dyre = fpade.pd_funvalc(romega, apar)
            x = hstack( (-romega.imag, romega.real) )
            _yim_ = f_unc(x, *popt)
            yim = _yim_[:len(romega)] + _yim_[len(romega):]*1j
        else:
            yre = s0 + dsdw*romega
            yim = s0 + dsdw*romega*1j
        fo = open('sigma_ancont'+debug, 'w')
        print('# enk=', enk*H2eV, 'and result is', yc[0]*H2eV, file=fo)
        for i in range(len(romega)):
            print(romega[i]*H2eV, yre[i].real*H2eV, yre[i].imag*H2eV, yim[i].real*H2eV, yim[i].imag*H2eV, file=fo)
        fo.close()
        fo = open('sigma_data'+debug, 'w')
        for i in range(len(omg)):
            print(omg[i]*H2eV, sc[i].real*H2eV, sc[i].imag*H2eV, file=fo)
        fo.close()
    return yc[0], dyc[0], apar
    
def mpiSplitArray(mrank,msize,leng):
    def SplitArray(irank,msize,leng):
        if leng % msize==0:
            pr_proc = int(leng/msize)
        else:
            pr_proc = int(leng/msize+1)
        if (msize<=leng):
            iqs,iqe = min(irank*pr_proc,leng) , min((irank+1)*pr_proc,leng)
        else:
            rstep=(msize+1)/leng
            if irank%rstep==0 and irank/rstep<leng:
                iqs = irank/rstep
                iqe = iqs+1
            else:
                if irank/rstep<leng:
                    iqs = irank/rstep
                    iqe = irank/rstep
                else:
                    iqs = leng-1
                    iqe = leng-1
        return iqs,iqe
    #print 'mrank=', mrank, 'msize=', msize, 'leng=', leng
    sendcounts=[]
    displacements=[]
    for irank in range(msize):
        iqs,iqe = SplitArray(irank,msize,leng)
        sendcounts.append((iqe-iqs))
        displacements.append(iqs)
    iqs,iqe = SplitArray(mrank,msize,leng)
    return iqs,iqe, array(sendcounts,dtype=int), array(displacements,dtype=int)


if __name__ == '__main__':

    sigx = load('sigx.npy')
    Vxct = load('Vxct.npy')
    bands = load('bands.npy')
    sigc = load('sigc.npy')
    omega = load('omega.npy')
    twhich_indices = load('twhich_indices.npy', allow_pickle = True )

    #print(twhich_indices)
    io = InOut("gw.inp", "debug.out", True)


    io.MatrixSelfEnergy = False
    (nirkp, nbnd, nbnd2, nom) = shape(sigc)
    sigc1 = zeros((nirkp, nbnd, nom), dtype=complex)
    sigx1 = zeros((nirkp,nbnd))
    for ie in range(nbnd):
        sigc1[:,ie,:] = sigc[:,ie,ie,:]
        sigx1[:,ie] = sigx[:,ie,ie]
    sigc = sigc1
    sigx = sigx1
    
    Compute_quasiparticles(bands, bands, sigc, sigx, Vxct, omega, io, 0, io.out, True)
