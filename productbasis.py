from numpy import *
from numpy import linalg
#from scipy import linalg


import radials as rd        # radial wave functions
#from cmn import *

Debug_Print = False

class ProductBasis:
    nspin_mb=1 # option for constructing mixed basis in the spin-polarized case (nspin=2) 
               # 2 or 1 -- 2==use both spin up and down radial wave functions, 1==use only spin up radial wave functions
    mb_ludot = False  # control whether considering \dot{u}_l when setting up the mixed basis 
    mb_lcore = True   # control whether considering core states when building the mixed basis 
    #lmbmax   = 2     # maximum l of radial functions considered when building the mixed basis set
    #lblmax   = 0     # the maximum L for the mixed basis (big l)
    #mb_emin = -1e10   # the energy cut-off for mixed basis funcitons, any radial functions  
    #mb_emax = 20.0    # with energy lower than mb_emin are dropped in constructing mixbasis 
    #
    def __init__(self, io_data, strc, in1, radf, core, nspin, fout):
        (self.kmr, self.pwm, self.lmbmax, self.wftol, self.lblmax, self.mb_emin, self.mb_emax) = io_data # decoding input files from input

        # maximum l at each atom is encoded in lmbmax, and each digit correspond to an atom, i.e., 34 means lmax=3 and lmax=4 at first
        # and second atom, respectively. If the number is negative, we use some default value, depending on Z.
        # If the number is positive, we use the same lmax=lmbmax at each atom.
        self.lmbmax_at = zeros(strc.nat,dtype=intc)
        if self.lmbmax > 10:
            ii, p = self.lmbmax, []
            while (ii>10):
                p.append( ii % 10 )
                ii = ii/10
            p.append(ii)
            p = p[::-1]
            for iat in range(min(strc.nat,len(p))):
                self.lmbmax_at[iat] = p[iat]
        else:  
            if self.lmbmax < 0:
                lvmax_at = zeros(strc.nat, dtype=int)
                for iat in range(strc.nat):
                    znuc= strc.Znuc[iat]
                    if znuc <= 2.0 :
                        lvmax_at[iat] = 0
                    elif znuc > 2.0 and znuc <= 18.0:
                        lvmax_at[iat] = 1
                    elif znuc > 18.0 and znuc <= 54.0:
                        lvmax_at[iat] = 2
                    else:
                        lvmax_at[iat] = 3
                self.lmbmax_at[:] = lvmax_at[:] + abs(self.lmbmax) 
            elif self.lmbmax > 0:
                self.lmbmax_at[:] = self.lmbmax
        # similarly decoding lblmax_at to self.lblmax_at
        self.lblmax_at = zeros(strc.nat,dtype=intc)
        if self.lblmax > 10:
            ii, p = self.lblmax, []
            while (ii>10):
                p.append( ii % 10 )
                ii = ii/10
            p.append(ii)
            p = p[::-1]
            for iat in range(min(strc.nat,len(p))):
                self.lblmax_at[iat] = p[iat]
        else:
            if self.lblmax <= 10:
                self.lblmax_at = self.lmbmax_at*2
            else:
                self.lblmax_at[:] = self.lblmax
      
        self.lmbmax = max(self.lmbmax_at)
        self.lblmax = max(self.lblmax_at)
        print('  Atom-specific lmbmax==(products made from u_l with l<lmbmax) and lblmax==(u_l1*u_l2->w_L with  L<lblmax):', file=fout)
        print('     iat      lmbmax   lblmax', file=fout)
        for iat in range(strc.nat):
            print('    %4d %8d %8d' % (iat,self.lmbmax_at[iat],self.lblmax_at[iat]), file=fout)
        
        #print >> fout, 'lmbmax= ', self.lmbmax
        #print >> fout, 'lblmax= ', self.lblmax 
        #
        #     Initial estimation of the maximum number of mixed basis functions       
        lomaxp1 = shape(in1.nlo)[0]
        nmixmax = (in1.nt + lomaxp1 * in1.nlomax) * (self.lmbmax+1)*(self.lmbmax+1)*self.nspin_mb
        
        self.ncore = zeros(strc.nat, dtype=int)
        self.big_l=[]
        self.ul_product=[]
        self.us_product=[]
        # calculate the radial part of the mixed basis functions  (mb_setumix)
        for iat in range(strc.nat):
            rp, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
            
            _big_l_=[]
            _ul_product_=[]
            _us_product_=[]
            
            print(' '*10+'Product basis functions for atom '+strc.aname[iat]+'\n', file=fout)
            nrf = self.count_nrf(iat,in1,core)
            #print >> fout, "# rad. func. to be considered on the atom %2d is %3d " % (iat+1, nrf)
            nup = nrf*(nrf+1)/2*self.nspin_mb
            #print >> fout, "mb_setuprod: # of uprod for atom %2d is %3d " % (iat,nup)
            ulprod = []
            usprod = []
            eles = []
            for isp in range(min(self.nspin_mb,nspin)):
                ul_all = zeros((nrf,npt))
                us_all = zeros((nrf,npt))
                l_all = zeros(nrf,dtype=int)
                #### collect_u
                irf = 0 
                # core states
                for ic,ecore in enumerate(core.eig_core[isp][iat]):
                    if ecore > self.mb_emin:
                        l_all[irf] = core.l_core[iat][ic]
                        ul_all[irf,:] = core.ul_core[isp][iat][ic][:]
                        us_all[irf,:] = core.us_core[isp][iat][ic][:]
                        irf = irf + 1
                self.ncore[iat] = irf
                for l in range(self.lmbmax_at[iat]+1):
                    l_all[irf] = l 
                    ul_all[irf,:] = radf.ul[isp,iat,l,:npt]
                    us_all[irf,:] = radf.us[isp,iat,l,:npt]
                    irf = irf + 1 
                    if self.mb_ludot:
                        l_all[irf] = l
                        ul_all[irf,:] = radf.udot [isp,iat,l,:npt]
                        us_all[irf,:] = radf.usdot[isp,iat,l,:npt]
                        irf = irf + 1
                    
                    if (l < lomaxp1):
                        for ilo in in1.nLO_at_ind[iat][l]:
                            #print 'in local orbital ilo=', ilo, 'l=', l, 'E=', in1.Elo[0,iat,l,ilo], 'cutoff=', self.mb_emax
                            if in1.Elo[0,iat,l,ilo] >= self.mb_emax : continue
                            l_all[irf] = l
                            ul_all[irf,:] = radf.ulo [isp,iat,l,ilo,:npt]
                            us_all[irf,:] = radf.uslo[isp,iat,l,ilo,:npt]
                            irf = irf + 1
                # construct all possible products of radial functions
                for jrf in range(nrf):
                    l2, u2, us2 = l_all[jrf], ul_all[jrf,:], us_all[jrf,:]
                    for irf in range(jrf+1):
                        l1, u1, us1 = l_all[irf], ul_all[irf,:],  us_all[irf,:]
                        eles.append( (l1,l2) )
                        ulprod.append( u1[:]*u2[:]/rp[:] )
                        usprod.append( us1[:]*us2[:]/rp[:] )
            # calculate the overlap matrix of product functions
            olap = zeros((len(eles),len(eles)))
            for i in range(len(eles)):
                for j in range(i+1):
                    olap[i,j] = rd.rint13g(strc.rel, ulprod[i], usprod[i], ulprod[j], usprod[j], dh, npt, strc.r0[iat])
                    olap[j,i] = olap[i,j]

            #for i in range(len(olap)):
            #    for j in range(i,len(olap)):
            #        print "%2d %2d  %12.7f " % (i,j,olap[i,j])


            nl = zeros(self.lblmax_at[iat]+1, dtype=int)
            for i in range(len(eles)):
                l_max = eles[i][0]+eles[i][1]
                l_min = abs(eles[i][0]-eles[i][1])
                for l in range(self.lblmax_at[iat]+1):
                    if l_min <= l and l <= l_max:
                        nl[l] += 1

            #print 'nl=', nl
            # diagonalize the overlap matrix of the product functions for each  L-block
            for L in range(self.lblmax_at[iat]+1):
                if nl[L] == 0: continue
                acceptable_func = [i for i in range(len(eles)) if ( L >= abs(eles[i][0]-eles[i][1]) and L <= eles[i][0]+eles[i][1] )]
                #print >> fout, ' '*14+'for L=',L,'acceptable_func=', acceptable_func
                # prepare those functions that enter this L
                ul_prod = zeros( (npt,len(acceptable_func)) )
                us_prod = zeros( (npt,len(acceptable_func)) )
                for i,i1 in enumerate(acceptable_func):
                    ul_prod[:,i] = ulprod[i1][:]
                    us_prod[:,i] = usprod[i1][:]
                
                # the L-block overlap matrix (uml)
                uml = zeros( (nl[L],nl[L]) )
                # generate the L-block overlap matrix
                #print >> fout, (' '*14)+' - generate the L-block overlap matrix'
                for i,i1 in enumerate(acceptable_func):
                    uml[i,i] = olap[i1,i1]
                    #print 'uml['+str(i)+','+str(i)+']='+str(uml[i,i])
                    for j,i2 in enumerate(acceptable_func[i+1:]):
                        #print 'i,k=', i, i+j+1
                        uml[i ,i+j+1] = olap[i1,i2]
                        uml[i+j+1,i ] = olap[i1,i2] 
                #print 'uml=', uml
                #print 'ind=', ind
                #print 'acceptable_func=', acceptable_func
                w, Ud = linalg.eigh(uml)

                finite_contribution = [i for i in range(len(acceptable_func)) if w[i]>self.wftol]
                #print >> fout, ' '*14+'finite_contribution=', finite_contribution
                Udn = zeros( (len(acceptable_func),len(finite_contribution)) )
                for i,j in enumerate(finite_contribution):
                    #print 'Udn['+str(i)+']= Ud['+str(j)+']'
                    Udn[:,i] = Ud[:,j]
                
                #print 'w=', w, 'v=', Ud
                ul_prodn = dot(ul_prod, Udn)
                us_prodn = dot(us_prod, Udn)

                #print >> fout, ' '*14+'shape(ul_prodn)=', shape(ul_prodn)
                #  proper normalization of the resulting radial product wave functions
                #print >> fout, " - normalize the rad mixed func"
                norms=[]
                for i,i1 in enumerate(finite_contribution):
                    norm = rd.rint13g(strc.rel, ul_prodn[:,i], us_prodn[:,i], ul_prodn[:,i], us_prodn[:,i], dh, npt, strc.r0[iat])
                    norms.append(norm)
                    ul_prodn[:,i] *= 1./sqrt(norm)
                    us_prodn[:,i] *= 1./sqrt(norm)
                    _big_l_.append(L)
                    _ul_product_.append(ul_prodn[:,i])
                    _us_product_.append(us_prodn[:,i])

                print((' '*5)+'L=%2d Nr. of products:%4d Nr. of basis functions%4d' % (L,  len(acceptable_func), len(finite_contribution) ), file=fout)  
                print(' '*14+'functions have norm', norms, file=fout)

            self.big_l.append(_big_l_)
            self.ul_product.append(_ul_product_)
            self.us_product.append(_us_product_)
            
            nwf = sum([(2*L+1) for L in self.big_l[iat]])
            print((' '*54)+'----', file=fout)
            print((' '*5)+'Total number of radial functions'+' '*17+'%4d    Maximum L %4d'  % ( len(self.big_l[iat]), max(self.big_l[iat]) ), file=fout)
            print((' '*5)+'Total number of basis functions '+' '*17+'%4d' % (nwf,), file=fout)

        print('  maxbigl=', max([max(self.big_l[iat]) for iat in range(len(self.big_l))]), file=fout)
        #     Calculate the total number of mixed wave functions (including M)    
        #     = size of the local part of the matrices
        ww = [sum([(2*L+1)*strc.mult[iat] for L in self.big_l[iat]]) for iat in range(strc.nat)]
        loctmatsize = sum(ww)

        lmixmax    = max(ww)
        print(' Max. nr. of MT-sphere wavefunctions per atom %6d' % (lmixmax,), file=fout)
        print(' Total  nr. of MT-sphere wavefunctions        %6d' % (loctmatsize,), file=fout)
        #
        # set an array that stores the general index of the mixed
        # function for a given mixed function of a given atom
        #
        ndf = sum(strc.mult)
        self.locmixind = zeros((ndf,lmixmax),dtype=int)
        nmixlm = zeros(ndf, dtype=intc)
        self.atm = []
        self.Prod_basis=[]
        self.iProd_basis={}
        idf=0
        imix=0
        for iat in range(strc.nat):
            for ieq in range(strc.mult[iat]):
                self.atm.append( iat )
                im = 0
                for irm in range(len(self.big_l[iat])):
                    L = self.big_l[iat][irm]
                    for M in range(-L,L+1):
                        self.locmixind[idf,im] = imix
                        self.iProd_basis[(idf,irm,L,M)] = imix
                        self.Prod_basis.append( (idf,irm,L,M) )
                        im += 1
                        imix += 1
                nmixlm[idf]=im
                idf += 1

        print(' List of all product basis functions:  total#:  ', imix, file=fout)
        for i in range(len(self.Prod_basis)):
            (idf,irm,L,M) = self.Prod_basis[i]
            print('  %3d %s%2d %s%3d %s%2d %s%3d' % (i, 'idf=', idf, 'irm=', irm, 'L=', L, 'M=', M), file=fout)


        #print 'locmixind=', self.locmixind
        #print 'nmixlm=', nmixlm

    def count_nrf(self,iat,in1,core):
      nrf = 0
      # core states
      for ic,ecore in enumerate(core.eig_core[0][iat]):
          if ecore > self.mb_emin:
              nrf = nrf + 1 
      # normal LAPW radial functions 
      if self.mb_ludot: # whether to consider u_dot
        nrf += (self.lmbmax_at[iat]+1)*2
      else:
        nrf += self.lmbmax_at[iat]+1

      # LO basis
      lomaxp1 = shape(in1.nlo)[0]
      for l in range(min(lomaxp1,self.lmbmax_at[iat]+1)): 
          for ilo in in1.nLO_at_ind[iat][l]:
              if in1.Elo[0,iat,l,ilo] < self.mb_emax :
                  nrf += 1
      return nrf
    
    def cmp_matrices(self, strc, in1, radf, core, nspin, fout):
        """Calculated all matrices related to radial mixbasis functions
         
           Some general matrices (independent of eigen-vectors ) related to
           bare Coulomb interaction:
             - rtl
             - rrint
             - tilg
        """
        print('set_mixbasis: calc mixmat', file=fout)
        import cum_simps as cs

        cfein = 1/137.0359895**2 if strc.rel else 1e-22

        N = max([len(self.big_l[iat]) for iat in range(strc.nat)])
        self.rtl   = zeros((strc.nat,N))          # < r^L | u_{im,at} >
        self.rrint = zeros((strc.nat, int(N*(N+1)/2) ))  # <u_{jm,L,at}| (r_<)^L / (r_>)^{L+1} | u_{im,L,at} >/
        for iat in range(strc.nat):
            #npt = strc.nrpt[iat]
            #dh  = log(strc.rmt[iat]/strc.r0[iat])/(npt - 1)      # logarithmic step for the radial mesh
            #dd = exp(dh)
            #rp = strc.r0[iat]*dd**range(npt)            
            rp, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
            #rm   = rp[::-1]
            for irm in range(len(self.big_l[iat])):
                L = self.big_l[iat][irm]
                r_to_L = rp**(L+1) 
                # Int[ u_i(r) r^{L+1}, {r,0,R}]
                rxov = rd.rint13g(strc.rel, self.ul_product[iat][irm], self.us_product[iat][irm], r_to_L, r_to_L, dh, npt, strc.r0[iat])
                if (rxov < 0):
                  self.rtl[iat,irm] = -rxov
                  self.ul_product[iat][irm] *= -1
                  self.us_product[iat][irm] *= -1
                else:  
                  self.rtl[iat,irm] = rxov
                
        for iat in range(strc.nat):
            #npt = strc.nrpt[iat]
            #dh  = log(strc.rmt[iat]/strc.r0[iat])/(npt - 1)      # logarithmic step for the radial mesh
            #dd = exp(dh)
            #rp = strc.r0[iat]*dd**range(npt)
            rp, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
            rm   = rp[::-1]
            for irm in range(len(self.big_l[iat])):
                L = self.big_l[iat][irm]
                r_to_L = rp**(L+1) 
                # double radial integrals <u_j| (r_<)^L/(r_>)^{L+1} | u_i>
                # the oustide loop for double  radial integrals
                u_i = self.ul_product[iat][irm] + cfein*self.us_product[iat][irm]
                u_i_r_to_L = cs.cum_simps(6, rp, u_i[:]*r_to_L[:])  # Int[ u_i(r)*r * r^L, {r,0,rx}]
                w = u_i[:]*rp[:]/r_to_L[:]
                u_i_m_to_L = cs.cum_simps(6, rm, w[::-1])[::-1]     # -Int[ u_i(r)*r / r^{L+1}, {r,R,rx}]
                
                for jrm in range(irm,len(self.big_l[iat])):
                    # the inside loop for double radial integrals
                    if self.big_l[iat][jrm] != L: continue
                    u_j = self.ul_product[iat][jrm] + cfein*self.us_product[iat][jrm]
                    #   Int[ u_j(rx)*( rx /rx^{L+1} * Int[ u_i(r)*r * r^L, {r,0,rx}] - rx^{L+1} * Int[ u_i(r)*r / r^{L+1}, {r,R,rx}] ), {rx,0,R}]
                    # = Int[ u_j(rx)*rx * Int[ u_i(r)*r * (r_<)^L/(r_>)^{L+1}, {r,0,R}], {rx,0,R}]
                    uij = u_j[:]*(rp[:]/r_to_L[:]*u_i_r_to_L[:] - r_to_L[:]*u_i_m_to_L[:])
                    ijrm = irm + int(jrm*(jrm+1)/2) # store into this index
                    self.rrint[iat,ijrm] = cs.cum_simps(6,rp,uij)[-1]
                
        if Debug_Print:
            for iat in range(strc.nat):
                print((' '*5)+'Radial integrals', file=fout)
                print((' '*13)+'N  L     <r^(L+1)|v_L>'+' '*9+'<r^(L+2)|v_L> ', file=fout)
                for irm in range(len(self.big_l[iat])):
                    L = self.big_l[iat][irm]
                    print(((' '*10)+'%4d%3d%18.10f ') % (irm,L,self.rtl[iat,irm]), file=fout)
                    
            for iat in range(strc.nat):
                print((' '*5)+'Double radial integrals', file=fout)
                print((' '*13)+'N1  N2  L     <v_i| (r_<)^L/(r_<)^{L+1} |v_j>', file=fout)
                for irm in range(len(self.big_l[iat])):
                    L = self.big_l[iat][irm]
                    for jrm in range(irm,len(self.big_l[iat])):
                        if self.big_l[iat][jrm] != L: continue
                        ijrm = irm + int(jrm*(jrm+1)/2)
                        print(((' '*10)+'%4d%4d %3d%18.10f%5d') % (irm,jrm,L,self.rrint[iat,ijrm],ijrm), file=fout)

        lomaxp1 = shape(in1.nlo)[0]
        how_many_fnc = zeros((nspin,strc.nat),dtype=int)
        for isp in range(nspin):
            for iat in range(strc.nat):
                # first collect all core radial wavefunctions
                how_many_functions  = len(core.l_core[iat])
                # ul functions
                for l in range(in1.nt):
                    how_many_functions += 1
                # energy derivative ul_dot functions
                for l in range(in1.nt):
                    how_many_functions += 1
                # LO local orbitals
                for l in range(lomaxp1):
                    for ilo in in1.nLO_at_ind[iat][l]:
                        how_many_functions += 1
                how_many_fnc[isp,iat] = how_many_functions
        
        n_mix = max([len(self.big_l[iat]) for iat in range(strc.nat)])
        #self.s3r = zeros( (nspin,strc.nat,amax(how_many_fnc),amax(how_many_fnc),n_mix) )
        self.s3r = zeros( (n_mix,amax(how_many_fnc),amax(how_many_fnc),strc.nat,nspin), order='F' )
        orb_info = [' core','     ',' dot ',' lo  ']
        for isp in range(nspin):
            for iat in range(strc.nat):
                #npt = strc.nrpt[iat]
                #dh  = log(strc.rmt[iat]/strc.r0[iat])/(npt - 1)      # logarithmic step for the radial mesh
                #dd = exp(dh)
                #rp = strc.r0[iat]*dd**range(npt)
                rp, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
                
                rwf_all = []
                lrw_all = []
                # first collect all core radial wavefunctions
                for ic,lc in enumerate(core.l_core[iat]):
                    lrw_all.append((lc,0))
                    rwf_all.append( (core.ul_core[isp][iat][ic][:], core.us_core[isp][iat][ic][:]) )   # note that these wave functions are normalized to sqrt(occupancy/degeneracy)
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
                        a3 = a1[:] * a2[:] / rp[:]
                        b3 = b1[:] * b2[:] / rp[:]
                        for irm in range(len(self.big_l[iat])):  # over all product functions
                            L = self.big_l[iat][irm]             # L of the product function
                            if L < abs(l1-l2) or L > (l1+l2): continue # triangular inequality violated
                            ## s3r == < u-product-basis_{irm}| u_{a1,l1} u_{a2,l2} > 
                            rint = rd.rint13g(strc.rel, self.ul_product[iat][irm], self.us_product[iat][irm], a3, b3, dh, npt, strc.r0[iat])
                            #self.s3r[isp,iat,ir2,ir1,irm] = rint
                            #self.s3r[isp,iat,ir1,ir2,irm] = rint
                            self.s3r[irm,ir1,ir2,iat,isp] = rint
                            self.s3r[irm,ir2,ir1,iat,isp] = rint
                if Debug_Print:
                    print((' '*5)+'Integrals <v_(NL) | u_(l1)*u_(l2)> for atom  %10s' % (strc.aname[iat],), file=fout)
                    print((' '*13)+'N   L   l1  u_   l2 u_        <v | u*u>', file=fout)
                    for irm in range(len(self.big_l[iat])):  # over all product functions
                        L = self.big_l[iat][irm]             # L of the product function
                        for ir2 in range(len(lrw_all)):
                            l2, it2 = lrw_all[ir2]
                            for ir1 in range(ir2+1):
                                l1, it1 = lrw_all[ir1]
                                #if self.s3r[isp,iat,ir2,ir1,irm]!=0:
                                if self.s3r[irm,ir1,ir2,iat,isp]!=0:
                                    #print >> fout, ((' '*10)+('%4d'*3)+'%s%4d%s%19.11e') % (irm,L,l1,orb_info[it1],l2,orb_info[it2],self.s3r[isp,iat,ir2,ir1,irm])
                                    print(('%4d%4d'+(' '*2)+('%4d'*3)+'%s%4d%s%19.11e') % (ir1,ir2,irm,L,l1,orb_info[it1],l2,orb_info[it2],self.s3r[irm,ir1,ir2,iat,isp]), file=fout)
                            
    def get_djmm(self,idf,l,m1,m2):
      if (abs(m2) > l) or (abs(m1) > l):
        return 0
      #  calculate the index of the djmm vector
      sgn_m2 = 1 if m2>=0 else -1  # note that this is like sign(m2), except at m2=0
      idx = int( l*(l+1)*(4*l-1)/6 ) + (l+1)*(sgn_m2*m1+l)+abs(m2)
      if m2 >= 0:
          # For m2>=0 D^j_m1m2=djmm(i)
          #print 'extracting l=', l, 'm1=', m1, 'm2=', m2, 'idx=', idx, 'val=', self.djmm[idf,idx]
          return self.djmm[idx,idf]
      else:
          # For m2<0 D^j_m1m2=(-1)^(m1-m2)djmm^*(i)
          m_1_m1m2 = 1-2*(abs(m1-m2) % 2)  # (-1)^(m1-m2)
          val = m_1_m1m2 * conj(self.djmm[idx,idf])
          #print 'extracting l=', l, 'm1=', m1, 'm2=', m2, 'idx=', idx, 'val=', val
          return val
    
    def generate_djmm(self,strc,latgen_trotij,in1_nt,fout):
        """
         Generates the rotation matrices for spherical harmonics $D^j_{mm'}$ up
        to \texttt{maxbigl} once
         $D^1_{mm'}$ is known using the inverse Clebsch-Gordan series:
        
        \begin{equation}
        D^j_{\mu m}=\sum\limits_{\mu_1=-1}^1{\sum\limits_{m_1=1}^1{C(1,j-1,j;%
        \mu_1,\mu-\mu_1)C(1,j-1,j;m_1,m-m_1)D^1_{\mu_1m_1}D^{j-1}_{\mu-\mu_1,m-m_1}}}
        \end{equation}.
        
         Since the program already calculates the Gaunt coefficients
        $G^{LM}_{l_1m_1,l_2m_2}$ we replace the product of Clebsch-Gordan
        coefficients using:
        
         \begin{equation}
        C(1,j-1,j;\mu_1,\mu-\mu_1)C(1,j-1,j;m_1,m-m_1)=\sqrt{\frac{4\pi(2j+1)}%
        {3(2j-1)}}\frac{G^{j\mu}_{1\mu_1,j-1\mu-\mu_1}G^{jm}_{1m_1,j-1m-m_1}}%
        {G^{j0}_{10,j-10}}
        \end{equation}
        """
        #
        #     calculate Gaunt coefficients
        #
        maxnt = max(in1_nt, 2*(self.lmbmax+1))
        import gaunt as gn
        self.cgcoef = gn.cmp_all_gaunt(maxnt)
        # print l1,l2,l3,m1,m2,m3, gn.getcgcoef(l1,l2,l3,m1,m2,cgcoef)
        
        iateq_ind=[]
        for iat in range(len(strc.mult)):
            iateq_ind += [iat]*strc.mult[iat]
        
        # Calculate the dimension of djmm
        #maxbigl = amax(self.big_l)
        maxbigl = max([max(self.big_l[iat]) for iat in range(len(self.big_l))])   # bug jul.7 2020
        dimdj = int( (maxbigl+1)*(maxbigl+2)*(4*maxbigl+3)/6 )
        ndf = len(iateq_ind)
        #print 'ndf=', ndf, 'dimdj=', dimdj, 'iateq_ind=', iateq_ind
        # Allocate djmm and initialize it to zero
        self.djmm = zeros((dimdj,ndf), dtype=complex, order='F')

        # Transform a rotation matrix from cartesian $(x,y,z)$ to spherical basis $(Y_{1,-11},Y_{1,0},Y_{1,1})$.
        s2 = 1/sqrt(2.)
        C2Sph = array([[ s2, -s2*1j,   0],
                       [ 0,    0,      1],
                       [-s2, -s2*1j,   0]], dtype=complex)

        if Debug_Print:
            for iat in range(len(strc.mult)):
                print('iat=', iat, 'rotloc=', file=fout)
                for i in range(3):
                    print(('%10.6f'*3) % tuple(strc.rotloc[iat][i,:]), file=fout)
            for idf in range(len(iateq_ind)):
                print('idf=', idf, 'rotij=', file=fout)
                for i in range(3):
                    print(('%10.6f'*3) % tuple(latgen_trotij[idf,:,i]), file=fout)
                
        #  Loop over all atoms
        for idf in range(len(iateq_ind)): # atom index counting all atoms
            iat = iateq_ind[idf]          # atom index counting only equivalent
            self.djmm[0,idf] = 1.0
            # Calculate the rotation matrix in the cartesian basis
            # rotcart=transpose(rotloc x rotij)
            rotcart = dot(strc.rotloc[iat], latgen_trotij[idf,:,:].T)
            
            # Transform the rotation matrix to the spherical basis
            rotsph = dot( dot( C2Sph, rotcart), C2Sph.conj().T )
            # Obtain the rotation matrix for j=1 D^1_{m,m')=transpose(rotsph)
            
            if Debug_Print:
                print('rotij=', file=fout)
                for i in range(3):
                    print(('%10.6f'*3) % tuple(latgen_trotij[idf,:,i]), file=fout)
                print('rotcart=', file=fout)
                for i in range(3):
                    print(('%10.6f'*3) % tuple(rotcart[i,:]), file=fout)
                print('rotsph=', file=fout)
                for i in range(3):
                    for j in range(3):
                        print(('%10.6f%10.6f') % (rotsph[i,j].real, rotsph[i,j].imag), ' ', end=' ', file=fout)
                    print(file=fout)
                
            for mu in [-1,0,1]:
                for m in [0,1]:
                    idx = 2*(mu+1)+m+1  # index for l=1
                    self.djmm[idx,idf] = rotsph[m+1,mu+1]
                    if Debug_Print:
                        print(' djmm[idf=%2d,l=%2d,mu=%3d,m=%3d,idx=%5d]=%16.10f%16.10f' % (idf+1,1,mu,m,idx+1,self.djmm[idx,idf].real,self.djmm[idx,idf].imag), file=fout)
            #  Obtain the rotation matrix for j> 1 by recursive relation
            for l in range(2,maxbigl+1):
                sql = sqrt(4*pi*(2*l+1)/(2*l-1.0)/3.)
                prefac = sql/gn.getcgcoef(1,l-1,l,0,0,self.cgcoef)
                #print '%2d %12.7f' % (l, prefac)
                _djmm_ = zeros( (2*l+1,l+1), dtype=complex )
                for mu in range(-l,l+1):
                    for mu1 in [-1,0,1]:
                        mu2 = mu-mu1
                        #print 'mu2=', mu2
                        if abs(mu2) > l-1: continue
                        cg1 = gn.getcgcoef(1,l-1,l,mu1,mu2,self.cgcoef)
                        #print '%3d %3d %3d %3d %12.7f' % (l, mu, mu1, mu2, cg1)
                        for m in range(l+1):
                            for m1 in [-1,0,1]:
                                m2 = m-m1
                                if abs(m2) > l-1: continue
                                dj1 = self.get_djmm(idf,1,mu1,m1)
                                dj2 = self.get_djmm(idf,l-1,mu2,m2)
                                cg2 = gn.getcgcoef(1,l-1,l,m1,m2,self.cgcoef)
                                #print '%3d %3d %3d %3d %3d  %12.7f %12.7f  %12.7f %12.7f  %12.7f' % (l, mu, mu1, m, m1, dj1.real, dj1.imag, dj2.real, dj2.imag, cg2)
                                _djmm_[l+mu,m] += cg1*cg2*dj1*dj2
                # pack the current results into array self.djmm
                for mu in range(-l,l+1):
                    for m in range(l+1):
                        idx = int( l*(l+1)*(4*l-1)/6) + (l+1)*(mu+l)+m
                        self.djmm[idx,idf] = _djmm_[l+mu,m]*prefac
                        if Debug_Print:
                            print(' djmm[idf=%2d,l=%2d,mu=%3d,m=%3d,idx=%5d]=%16.10f%16.10f' % (idf+1,l,mu,m,idx+1,self.djmm[idx,idf].real,self.djmm[idx,idf].imag), file=fout)

