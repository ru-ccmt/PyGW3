from numpy import array,sqrt,pi,dot,ones,zeros
from timeit import default_timer as timer

import for_kpts as fkp      # 

from cmn import FORT
import gwienfile as w2k

Debug_Print = False

class PlaneWaves:
    """Generates fixed basis of plane waves G, which will be used in subsequent calculations.
    """
    def __init__(self, hsrws, kmr, pwm, case, strc, in1, latgen, kqm, debug, fout):
        tm1 = timer()
        maxngk = max(hsrws)
        kxcmax = w2k.Read_xc_file_dimension(case, strc, fout)
        
        kmax = in1.rkmax/min(strc.rmt)
        ng = zeros(3,dtype=int)
        for i in range(3):
            bleng = sqrt(sum(latgen.br2[i,:]**2))
            ng[i] = int(kmr * kmax * pwm/bleng) + 1
        
        print('kxcmax=', kxcmax, 'ng1,ng2,ng3=', ng, file=fout)
        igm = array([max(4*ng[i],2*kxcmax+ng[i]) for i in range(3)],dtype=int)
        
        ng = list(map(int, igm/2))
        npw = (2*ng[0]+1)*(2*ng[1]+1)*(2*ng[2]+1)
        
        maxgxc = 2.*kmax+4
        maxgcoul = kmr*kmax*(pwm+2.)
        gmax = max(maxgxc, maxgcoul)
        
        tm2 = timer()
        print('## PlaneWave t(read_xc)            =%14.9f' % (tm2-tm1,), file=fout)
        print('gmax=', gmax, 'br2=', latgen.br2, file=fout)
        
        ortho = (latgen.ortho or strc.lattice[1:3]=='CXZ')
        if FORT:
            #glen=[]    # |G|
            #gindex=[]  #  G in terms of (i0,i1,i2)
            #G_c=[]     #  G in cartesian
            self.npw2apw, ngindx = fkp.count_number_pw(ng,kmax,gmax,latgen.br2)
            is_CXZ = (strc.lattice[1:3]=='CXZ')
            self.glen, self.G_c, self.gindex = fkp.generate_pw(ng,ngindx,gmax,latgen.pia,latgen.br2,ortho,is_CXZ,True)
        else:
            glen=[]    # |G|
            gindex=[]  #  G in terms of (i0,i1,i2)
            G_c=[]     #  G in cartesian
            self.npw2apw = 0
            for ii,ki in enumerate(itertools.product(list(range(-ng[0],ng[0]+1)),list(range(-ng[1],ng[1]+1)),list(range(-ng[2],ng[2]+1)))):
                kc = dot(latgen.br2,ki)
                kk = sqrt(sum(kc**2))
                if kk < 2.*kmax+4: self.npw2apw += 1
                if kk < gmax :
                    glen.append( kk )
                    if ortho:
                        gindex.append( list(map(int, round_(kc/latgen.pia))) )
                    else:
                        if strc.lattice[1:3]=='CXZ':
                            gindex.append( [ki[0]+ki[2],ki[1],ki[2]-ki[0]] )
                        else:
                            gindex.append( ki )
                    G_c.append( kc )
            G_c = array(G_c)
            gindex = array(gindex,dtype=int)
            
            # Now we will sort plane waves according to their length
            indx = argsort(glen, kind='stable')  # just obtaining index to the sorted sequence
            
            # here we rearange all arrays so that they are in sorted order
            self.glen = zeros(shape(glen))           # |G|
            self.G_c  = zeros(shape(G_c))            # \vG in cartesian
            self.gindex = zeros(shape(gindex),dtype=int) # \vG in integer
            #self.ig0={}
            for i0,i in enumerate(indx):
                self.gindex[i0,:] = gindex[i]
                self.glen[i0]     = glen[i]
                self.G_c[i0]      = G_c[i]
                #iG = gindex[i]
                #self.ig0[tuple(iG)] = i0
        
        
        tm3 = timer()
        print('## PlaneWave t(gen_ipw_fixed_basis)=%14.9f' % (tm3-tm2,), file=fout)
        #####
        if False:
            ft = open('python_sorted_index.dat', 'w')
            for i in range(len(self.glen)):
                print('%4d  %4d%4d%4d  %15.10f  %15.10f%15.10f%15.10f' % (i+1, self.gindex[i,0], self.gindex[i,1], self.gindex[i,2], self.glen[i], self.G_c[i,0], self.G_c[i,1], self.G_c[i,2]), file=ft)
            ft.close()
            
        # This is just temporary to debug the code, we want to be completely compatible with the fortran cdoe.
        if False:  ##### ATTENTION ????
            fix = open('sorted_index.dat')
            self.glen=[]
            self.gindex=[]
            self.G_c=[]
            for ii,line in enumerate(fix):
                iG = list(map(int,line.split()[1:4]))
                Gcc = iG * latgen.pia
                kk = sqrt(sum(Gcc**2))
                self.glen.append( kk )
                self.gindex.append( iG )
                self.G_c.append( Gcc )
            self.glen = array(self.glen)
            self.G_c = array(self.G_c)
            self.gindex = array(self.gindex, dtype=int)
            
        ##### 
        self.ig0={}
        for i in range(ngindx):
            self.ig0[ tuple(self.gindex[i]) ] = i
        
        # so that we can keep using short glen, G_c and gindex
        glen = self.glen      # |G|
        G_c  = self.G_c       # \vG in cartesian
        gindex = self.gindex  # \vG in integer
        
        if Debug_Print:
            print('Sorted Gs', file=fout)
            for i0 in range(len(gindex)):
                print('%4d ' % (i0+1,), '(%3d,%3d,%3d)' % tuple(self.gindex[i0]), '%9.5f' % (self.glen[i0],), '[%9.5f,%9.5f,%9.5f]' % tuple(self.G_c[i0]), file=fout)
        
        tm4 = timer()
        print('## PlaneWave t(sort_ipw)           =%14.9f' % (tm4-tm3,), file=fout)
        
        # This part calculates the integral of a plane wave with wave vector k
        #     1/V * Integral[ e^{i*k*r} , {interstitial}]
        # belonging to the reciprocal Bravais lattice in the
        # interstitial region by the difference between the integral over the whole
        # unit cell and the Muffin Tin spheres
        # First, get vmt == the volume ration for each atom type
        self.vmt = array([4*pi*strc.rmt[iat]**3/3. * 1/latgen.Vol for iat in range(strc.nat)])
        mult = array(strc.mult)
        Rmt = array(strc.rmt)
        if FORT:
            self.ipwint = fkp.pw_integ(gindex,glen,self.vmt,Rmt,strc.vpos,mult)
        else:
            vmt_3 = self.vmt*3.0
            #
            self.ipwint = zeros(len(gindex),dtype=complex) # result= Integrate[e^{i*k*r},{r in interstitials}]_{cell,interstitial}/V_{cell}
            for i in range(len(gindex)):
                ki, ak = gindex[i], glen[i]  # ki=(i0,i1,i2); ak=|kc|
                if ak < 1e-10:
                    self.ipwint[i] = 1 - sum(self.vmt*mult)
                else:
                    kr  = ak * Rmt                       # kr[iat] = |kc|*Rmt[iat]
                    j1 = special.spherical_jn(1,kr)      # spherical_bessel == (sin(x)/x-cos(x))/x
                    intmod = vmt_3 * j1 / kr             # intmod[iat] = j1*3*vmt/kr
                    integc = 0j
                    for iat in range(strc.nat):
                        ekr = exp(2*pi*1j*dot(array(strc.pos[iat]),ki))  # ekr[ieq]
                        integc += intmod[iat] * sum(ekr)                 # sum(ekr)==sum(phase over all equivalent atoms)
                    self.ipwint[i] = -integc
        if debug:
            print('ipwint: ', file=fout)
            for i in range(len(gindex)):
                ki, ak = gindex[i], glen[i]  # ki=(i0,i1,i2); ak=|kc|
                print('%4d ' % (i+1,), '%3d'*3 % tuple(ki), ' %16.11f%16.11f' %(self.ipwint[i].real, self.ipwint[i].imag), file=fout)
        
        tm5 = timer()
        print('## PlaneWave t(integral_ipw)       =%14.9f' % (tm5-tm4,), file=fout)
        # First determine the maximum number of plane waves in variuos parts of the calculation
        maxlen_mb   = kmr * kmax      # |G+q| should be smaller than maxlen_mb for Mixbasis (maxngq)
        maxlen_coul = maxlen_mb * pwm # |G+q| should be smaller than maxlen_coul for bare Coulomb
        q_c = dot(kqm.qlist, latgen.br2.T)/float(kqm.LCMq)  # q in cartesian coordinates
        if FORT: # faster code written in Fortran
            self.ngq,self.ngq_barc,self.ngqlen = fkp.ngq_size(q_c,G_c,maxlen_mb,maxlen_coul)
        else:    # slower but equivalent Python code
            self.ngq = zeros(len(kqm.qlist),dtype=int) # plane waves for Mixbasis
            self.ngq_barc = zeros(len(kqm.qlist),dtype=int) # plane waves for bare Coulomb
            for iq in range(len(kqm.qlist)):
                #print '%3d ' % (iq+1,), ('%15.10f'*3) % tuple(q_c[iq]), ' %3d%3d%3d' % tuple(kqm.qlist[iq,:])
                kpq = [linalg.norm(G_c[i,:] + q_c[iq,:]) for i in range(len(G_c))]
                self.ngq[iq] = sum(1 for akq in kpq if akq<maxlen_mb)
                self.ngq_barc[iq] = sum(1 for akq in kpq if akq<maxlen_coul)
        maxngq = max(self.ngq)               # max number of plane waves for Mixbasis across all q-points
        maxngq_barc = max(self.ngq_barc)     # max number of plane waves for bare Coulomb across all q-points
        if maxngq == len(gindex):
            print('WARNING !! maxngq = npw !!!', file=fout)
        if maxngq_barc == len(gindex):
            print('WARNING!! maxngq_barc = npw !!!', file=fout)
        
        if FORT:  # faster code written in Fortran
            maxngqlen   = max(self.ngqlen)
            self.indgq, self.gqlen, self.G_unique = fkp.ngq_sort(q_c,G_c,maxlen_coul,self.ngqlen,maxngq_barc,maxngqlen)
        else:     # slower but equivalent Python code
            self.indgq    = zeros((len(kqm.qlist),maxngq_barc),dtype=int)  # index to the plane wave G for which |G+q| is beyond certain cutoff for bare Coulomb
            self.gqlen = zeros((len(kqm.qlist),maxngq_barc))               # the value of |G+q| for plane waves beyond certain cutoff for bare Coulomb
            G_unique_ = [[] for iq in range(len(kqm.qlist))]               # many of |G+q| are equal, and we want to store only unique terms. This is just index to the first occurence with unique length
            
            Gpq_len = zeros(maxngq_barc)
            which_ik= zeros(maxngq_barc, dtype=int)
            for iq in range(len(kqm.qlist)):
                kpq = [linalg.norm(G_c[i,:] + q_c[iq,:]) for i in range(len(G_c))]
                n = 0
                for i in range(len(G_c)):
                    if (kpq[i] < maxlen_coul):
                        Gpq_len[n] = kpq[i]
                        which_ik[n] = i
                        n += 1
                indxc = argsort(Gpq_len[:n], kind='stable')
                aG_prev = -1
                for i0,i in enumerate(indxc):
                    aG = Gpq_len[i]
                    self.indgq[iq,i0] = which_ik[i]
                    self.gqlen[iq,i0] = aG
                    if abs(aG-aG_prev)>1e-6:
                        G_unique_[iq].append(i0)
                        aG_prev = aG
            self.ngqlen = [len(G_unique_[iq]) for iq in range(len(kqm.qlist))]
            maxngqlen   = max(self.ngqlen)
            # Now creating array from list of lists for G_unique, to be compatible with above Fortran code
            self.G_unique = zeros((len(G_unique_),maxngqlen),dtype=int)
            for iq in range(len(G_unique_)):
                self.G_unique[iq,:self.ngqlen[iq]] = G_unique_[iq][:]



        tm7 = timer()
        print('## PlaneWave t(max |k+q|)          =%14.9f' % (tm7-tm5,), file=fout)

        print('Nr. of IPW(npw):',npw, file=fout)
        print('Nr. of pws for lapw overlaps (npw2apw):',self.npw2apw, file=fout)
        print('Max nr. of IPW for APW (maxngk):',maxngk, file=fout)
        print('Max nr. of IPW for Mixbasis (maxngq) :',maxngq, file=fout)
        print('Max nr. of IPW for bare Coulomb (maxngq_barc) :',maxngq_barc, file=fout)

        
        print('maxlen_mb=', maxlen_mb, ' maxlen_coul=', maxlen_coul, file=fout)
        print('gqlen and G_unique', file=fout)
        for iq in range(len(kqm.qlist)):
            print('%5d' %(iq+1,), 'q=', q_c[iq,:], 'ngq=%5d ngq_barc=%5d' % (self.ngq[iq], self.ngq_barc[iq]), file=fout)
            if debug:
                for i in range(self.ngq_barc[iq]):
                    ii = self.indgq[iq,i]
                    print('   %5d %15.10f' % (i+1,self.gqlen[iq,i]), '%4d' % (ii+1,), '%15.10f'*3 % tuple(G_c[ii,:]), gindex[ii,:], file=fout)
            if Debug_Print:
                for i in range(self.ngqlen[iq]):
                    i0 = self.G_unique[iq,i]
                    ii = self.indgq[iq,i0]
                    print('    --- %3d %15.10f' % (i0, self.gqlen[iq,i0]), ii, gindex[ii,:], G_c[ii,:], file=fout)
                
    def convert_ig0_2_array(self):
        ivects = list(self.ig0.keys())
        avects = array( ivects, dtype=int)
        mx, my, mz = max(abs(avects[:,0])), max(abs(avects[:,1])), max(abs(avects[:,2]))
        iag0 = -ones((2*mx+1,2*my+1,2*mz+1), dtype=int)
        for ik in ivects:
            iag0[ik[0]+mx, ik[1]+my, ik[2]+mz] = self.ig0[ik]
        return iag0

    def inverse_indgq(self, iq):
        # this is indggq
        max_len = max(self.indgq[iq,:self.ngq_barc[iq]])+1
        indgq_inverse = -ones(max_len, dtype=int)
        for ibasis in range(self.ngq_barc[iq]):
            ig = self.indgq[iq,ibasis]   # ig == index in gvec
            indgq_inverse[ig] = ibasis # if we have index in gvec, we can index in basis in mesh up to ngq_barc
        return indgq_inverse
