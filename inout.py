#from scipy import *
from numpy import sqrt
import re
import os,sys
from cmn import Ry2H, H2eV, FORT

class InOut:
    def  __init__(self, gwinp, gwout, PRINT=True):
        if PRINT:
            self.out = open(gwout,'w')
        else:
            self.out = open(os.devnull, 'w')
        self.float = "[-+]?\d*\.?\d*[e|E]?[-+]?\d*"
        self.inpdata = open(gwinp,'r').readlines()
        self.ParsAll(gwinp)
        #del self.inpdata
        
    def Pars(self, pattern, default=''):
        for line in self.inpdata:
            m = re.search(pattern,line)
            if m is not None:
                return m.group(1)
        return default
    
    def ParsBlock(self, pattern):
        for ist,line in enumerate(self.inpdata):
            m = re.search(pattern,line)
            if m is not  None:
                break
        block_data=[]
        if ist < len(self.inpdata)-1:
            iend = ist+1
            while (iend < len(self.inpdata) ):
                m = re.search('\%', self.inpdata[iend])
                if m is not None:
                    break
                no_comments = ((self.inpdata[iend]).split('#'))[0]
                block_data.append( no_comments.split('|') )
                iend += 1
        return block_data
        
    def ParsAll(self, gwinp):
        from datetime import datetime
        print('*'*80, file=self.out)
        print('*',' '*28, 'GW Program : PyGW',' '*28, '*', file=self.out)
        print('*',' '*28, datetime.now(), file=self.out)
        print('*'*80, file=self.out)
        print(file=self.out)
        print('-'*32, ' inout.py:InOut::ParsAll() ', '-'*32, file=self.out)
        print('parsing old input file gw.inp, which will eventually be replaced', file=self.out)
        self.case = self.Pars("CaseName\s*= \"(\w*)\"")
        
        print('CaseName=', self.case, file=self.out)
        
        #self.iop_scratch = self.Pars("UseScratch\s*=\s*(\d)")
        #if(self.iop_scratch == '0'):
        #    print >> self.out, " -- Do not use scratch space"
        #elif (self.iop_scratch == '1'):
        #    print >> self.out, " -- Use different scratch files for different proc."
        #else:
        #    print >> self.out, " -- Use same scratch files for different proc."
        
        if (not os.path.exists(self.case+".struct")):
          print('ERROR: structure file '+self.case+'.struct does not exist', file=self.out)
          sys.exit(0)
        
        #self.savdir = self.Pars("SavDir\s*=\s*\"(.*)\"")
        #print >> self.out, "SavDir: ", self.savdir
        
        #self.taskname = self.Pars("Task\s*=\s*([\w|']*)")
        #print >> self.out, "Task: ", self.taskname
        
        self.nspin = int( self.Pars("nspin\s*=\s*(\d)") )
        print('nspin=', self.nspin, file=self.out)
        fspin = 2./self.nspin
        if(self.nspin == 2):
            self.spflag = ['up','dn']
        else:
            self.spflag = ['']
        
        self.iop_core =  0
        print('iop_core=', self.iop_core, ' -- using all core states in this calculation', file=self.out)
        self.lomax =  int( self.Pars("LOmax\s*=\s*(\d+)") )      # Number of local orbitals
        print('LOmax=', self.lomax, '-- maximum for local orbitals, needed to be specified for wien2k vector/energy file', file=self.out)
        
        #self.lsymvector = self.Pars("SymVector\s*=\s*([T|F])")   #     whether using vector files taking symmetry into account
        #if self.lsymvector=='T':
        #    print >> self.out, "Use symmetrized eigenvector file"
        #else:
        #    print >> self.out, "Use non-symmetrized eigenvector file"
        
        # Set the window (in Ry) for the number of unoccupied bands for which GW band correction is to be calculated
        self.emaxgw = float( self.Pars("emaxgw\s*=\s*("+self.float+")",1e4) )
        self.emingw = float( self.Pars("emingw\s*=\s*("+self.float+")",-1e4) ) 
        print('(emingw,emaxgw)=('+str(self.emingw)+','+str(self.emaxgw)+') energy range in Ry (determines active bands) for self-energy calculation (only for external legs)', file=self.out)
        # Convert to Hartree unit
        self.emaxgw *= Ry2H
        self.emingw *= Ry2H

        self.emin_tetra = float( self.Pars("emin_tetra\s*=\s*("+self.float+")", -1) )
        self.emax_tetra = float( self.Pars("emax_tetra\s*=\s*("+self.float+")",  1) )
        self.emin_tetra *= Ry2H
        self.emax_tetra *= Ry2H
        
        self.mb_emin = float( self.Pars("MB_emin\s*=\s*("+self.float+")",-1e10) )
        self.mb_emax = float( self.Pars("MB_emax\s*=\s*("+self.float+")",20) )
        #self.mb_emin *= Ry2H we actually use Ry for linearization energies
        #self.mb_emax *= Ry2H
        
        # Block size used for Minm matrix operations
        #print >> self.out, "Options related to Minm:"
        #self.mblksiz = int( self.Pars("Minm_mblksiz\s*=\s*(\d+)",10) )
        #print >> self.out, "block size for m-index(mblksiz):", self.mblksiz
        
        # Read the energy cut-off for polarization matrix and correlation selfenergies 
        #  when emaxpol/emaxsc is negative, all unoccupied bands are used  
        #self.eminpol = float( self.Pars("eminpol\s*=\s*("+self.float+")",-1e10) )
        self.emax_pol = float( self.Pars("emaxpol\s*=\s*("+self.float+")",-1e10) )
        self.emax_sc  = float( self.Pars("emaxsc\s*=\s*("+self.float+")",-1e10) )
        #self.emin_sc  = float( self.Pars("eminsc\s*=\s*("+self.float+")",-1e10) )
        print('emaxpol=', self.emax_pol, ' -- upper energy cutoff (Ry) for computing polarization', end=' ', file=self.out)
        if self.emax_pol < 0 :
            print('-- We take all available bands in computing polarization', file=self.out)
        else:
            print(file=self.out)
        print('emaxsc =', self.emax_sc, '-- upper cutoff energy (Ry) in computing dynamic self-energy in internal loop', end=' ', file=self.out)
        if self.emax_sc < 0:
            print('-- We take all available bands in computing dynamic self-energy', file=self.out)
        else:
            print(file=self.out)
        
        #self.eminpol *= Ry2H # convert to unit of Ha. 
        self.emax_pol*= Ry2H # convert to unit of Ha. 
        self.emax_sc *= Ry2H # convert to unit of Ha. 
        #self.eminsc  *= Ry2H # convert to unit of Ha.
        
        #self.nvel = float( self.Pars("nvel\s*=\s*("+self.float+")") )
        #print >> self.out, 'nvel=',  self.nvel
        
        #self.core_ortho = self.Pars("Core_ortho",'F')
        #print >> self.out, 'core_ortho=', self.core_ortho
        
        save_mwm = self.Pars("save_mwm\s*=\s*([T|F])",'T')
        if save_mwm=='T':
            self.save_mwm = True
        else:
            self.save_mwm = False
        print('save_mwm=', self.save_mwm, file=self.out)
        
        self.lcmplx = self.Pars("ComplexVector\s*=\s*([T|F])",'F')
        print('Complex or real KS vectors?', end=' ', file=self.out)
        if self.lcmplx=='T':
            print(" -- Complex Vector", file=self.out)
        else:
            print(" -- Real Vector", file=self.out)
            
        # Parameters related to self-consistency
        self.eps_sc = float( self.Pars("eps_sc\s*=\s*("+self.float+")",1e-4) )
        self.nmax_sc = int(  self.Pars("nmax_sc\s*=\s*(\d+)",20) )
        self.mix_sc = float( self.Pars("mix_sc\s*=\s*("+self.float+")",1.0) )
        # iop_vxc : control how to treat vxc 
        #  0 -- calculate from vxc data read from the wien2k case.r2v file 
        #  1 -- directly read from an external file
        #self.iop_vxc = int( self.Pars("iop_vxc\s*=\s*(\d+)",0) )
        
        self.iop_bzint  = 0 
        self.iop_bzintq = 0
        
        self.eta_head = float( self.Pars("eta_head\s*=\s*("+self.float+")",0.01) )
        print('eta_head=', self.eta_head, '-- broadening of plasmon in the head', file=self.out)
        #self.ztol_sorteq = 0.01
        #self.tol_taylor = 10.
        self.esmear = 0.01
        #self.eta_freq = 0.01
        #self.n_gauq = 8
        #self.ztol_vol =  1e-10
        self.iop_bcor = 0
        
        self.efermi = float( self.Pars("EFermi\s*=\s*("+self.float+")",1e4) )
        if self.efermi < 1e2:
            self.efermi *= Ry2H
            print("Read Fermi energy from the gw input:", self.efermi, file=self.out)
        
        #self.iop_metallic = 0 # it seems we always expect insulator
        #self.spinmom = 0
        
        mbd = self.ParsBlock('^\%BZConv')
        self.fdep  = mbd[0][1].strip().strip('"')
        
        #print >> self.out, 'bzcon=', self.bzcon
        print('Note: Using tetrahedron method on imaginary axis', file=self.out)
        print('fdep=', self.fdep, file=self.out)
        if self.fdep == 'nofreq':
            self.fflg=1
        elif self.fdep == 'refreq':
            self.fflg=2
        elif self.fdep=='imfreq' or fdep=='IMFREQ':
            self.fflg=3
        else:
            print("WARNING: unsupported option for fdep!", file=self.out)
            print("--Taking default value: imfreq", file=self.out)
            self.fdep = 'imfreq'
            self.fflg=3
        #print >> self.out, "fdep=", self.fdep
        self.rmax=40.0

        mbd = self.ParsBlock('^\%FourIntp')
        if mbd:
            rmax = float(mbd[0][0])
        else:
            rmax = 40.0
            
        #mbd = self.ParsBlock('^\%kMeshIntp')
        #if mbd:
        #    self.iop_kip    = int(mbd[0][0])
        #    self.eqptag_kip = mbd[0][1].strip().strip('"')
        #else:
        #    self.iop_kip=0
        #    self.eqptag_kip=''
        #print >> self.out, 'iop_kip=', self.iop_kip, 'eqptag_kip=', self.eqptag_kip
        
        self.iop_drude = 1
        self.omega_plasma = -1.0
        self.omega_plasma /= H2eV
        
        self.iop_epsw = 0
        self.iop_mask_eps = 0
        self.q0_eps = 1/sqrt(3.0)
        
        mbd = self.ParsBlock('^\%FreqGrid')
        self.iopfreq, self.nomeg, self.omegmax, self.omegmin, self.iopMultiple = 4, 32, 20., 0.02, 1
        if mbd:
            self.iopfreq = int(mbd[0][0])
            self.nomeg   = int(mbd[0][1])
            self.omegmax = float(mbd[0][2])
            self.omegmin = 0
            if self.iopfreq == 1 or self.iopfreq>3:
                self.omegmin = float(mbd[0][3])
            self.svd_cutoff = 1e-10
            if self.iopfreq == 5:
                if len(mbd[0])>4:
                    self.svd_cutoff = float(mbd[0][4])
                if self.svd_cutoff > 1e-5:
                    print('WARNING: svd_cutoff is large ==', self.svd_cutoff, 'This would be very imprecise calculation.', file=self.out)
                    print('WARNING: Setting cutoff to 1e-10', file=self.out)
                    self.svd_cutoff = 1e-10
            if self.iopfreq == 4:
                if len(mbd[0])>4:
                    self.iopMultiple  = int(mbd[0][4])
                if self.iopMultiple < 1 or self.iopMultiple > 1000:
                    print('WARNING: mesh for convolution can have iopMultiple more points, where specified %FreqGrid[4]==iopMultiple='+str(self.iopMultiple), file=self.out)
                    self.iopMultiple = 1
                    print('WARNING: this value of iopMultiple does not make sense, hence setting it to '+str(self.iopMultiple), file=self.out)
                
        print('FreqGrid:', file=self.out)
        fginfo = ['Equally spaced mesh', 'Grid for Gauss-Laguerre quadrature', 'Grid for double Gauss-Legendre quadrature,', 'Grid of Tan-mesh for convolution', 'Using SVD basis and Tan-mesh for convolution']
        print('  iopfreq= %4d' % (self.iopfreq,), '-- '+ fginfo[self.iopfreq-1], file=self.out)
        print('  nomeg  = %4d' % (self.nomeg,)  , '-- number of Matsubara frequency points', file=self.out)
        print('  omegmax= '+str(self.omegmax), '-- upper frequency cutoff in Hartree', file=self.out)
        print('  omegmin= '+str(self.omegmin), '-- the low energy cutoff in Hartree', file=self.out)
        if self.iopfreq == 4:
            print('  iopMultiple='+str(self.iopMultiple), '-- how many more points should be used in frequency convolution', file=self.out)
        
        self.nproc_col, self.nproc_row = 0, 1
        
        mbd = self.ParsBlock('^\%kmesh')
        self.nkdivs=[1,1,1]
        self.k0shift=[0,0,0]
        if mbd:
            self.nkdivs = [int(mbd[0][0]), int(mbd[0][1]), int(mbd[0][2])]
            self.k0shift = [int(mbd[1][0]), int(mbd[1][1]), int(mbd[1][2])]
        print('kmesh:', file=self.out)
        print('  nkdivs =', self.nkdivs, ' -- how many k-points along each reciprocal vector', file=self.out)
        print('  deltak =', self.k0shift, ' -- shift of k-mesh', file=self.out)

        mbd = self.ParsBlock('^\%SelfEnergy')
        self.fnpol_ac, self.iop_es, self.iop_ac = 2, 0, 1
        if mbd:
            self.npol_ac = int(mbd[0][0])
            self.iop_es  = int(mbd[0][1])
            self.iop_ac  = int(mbd[0][2])
        
        self.iop_esgw0 = 1 # whether shift the Fermi energy during self-consistent GW0
        #self.iop_gw0   = 1 # how the do GW0 self-consistent iteration
        self.iop_gw0 = int( self.Pars("iop_gw0\s*=\s*(\d)", 1) )
        print('\niop_gw0=', self.iop_gw0, end=' ', file=self.out)
        if self.iop_gw0==1:
            print(' Note, in GW0 we use e_{qp}=e_{ks} + (Sig-Vxc) and G0W0 we use e_{qp}=e_{ks} + Zk*(Sig-Vxc)', file=self.out)
        elif self.iop_gw0==2:
            print(' Note, in GW0 and G0W0 we use  e_{qp}=e_{ks} + Zk*(Sig-Vxc)', file=self.out)
        else:
            print('  I think this iop_gw0 is not specified', file=self.out)
        print(file=self.out)

        self.shift_semicore = int( self.Pars("shift_semicore\s*=\s*(\d)", 0) )
        print('\nshift_semicore=', self.shift_semicore, file=self.out)

        self.iop_rcf   = 0.8 # real-frequency cutoff for using conventional pade
        self.npar_ac=2*self.npol_ac
        if self.nomeg < self.npar_ac:
            print('WARNING: not enough freq for analytic continuation', file=self.out)
            print('  - npar_ac .gt. nomeg', file=self.out)
            print('  - npar_ac,nomeg=', self.npar_ac, self.nomeg, file=self.out)
            print('  - reset npar_ac =nomeg', file=self.out)
            self.npar_ac = self.nomeg
            self.npol_ac = self.npar_ac/2
        
        self.anc_type=['old-fashioned Pade with n='+str(self.npar_ac-1),'modified Pade (Rojas, Godby and Needs) with '+str(self.npar_ac)+' coefficients','Simple quasiparticle approximation']
        print('SelfEnergy: (analytic continuation of self-energy)', file=self.out)
        print('  npol_ac='+str(self.npol_ac), ', iop_es='+str(self.iop_es), ', iop_ac='+str(self.iop_ac), file=self.out)
        print('  iop_ac =', self.iop_ac, 'i.e., analytic continuation type is '+self.anc_type[self.iop_ac], file=self.out)
        print('  number of AC poles (npol_ac) =', self.npar_ac/2, file=self.out)
        if self.iop_ac==0:
            print('  iop_rcf ='+str(self.iop_rcf), ' -- above this energy (in Ry) we use modified Pade (Rojas, Godby and Needs) and below this energy the original pade', file=self.out)
        #print >> self.out, '- Nr. of poles used in analytic continuation:', self.npol_ac
        #print >> self.out, '- Option for calculating selfenergy(iop_es): ', self.iop_es
        #if self.iop_es == 0:
        #    print >> self.out, "  -- perturbative calculation"
        #else:
        #    print >> self.out, "  -- iterative calculation"
        
        #print >> self.out, '- Option of analytic continuation (iop_ac):', self.iop_ac
        #if self.iop_ac == 1:
        #    print >> self.out, "  -- RGN method(Rojas, Godby and Needs)"
        #else:
        #    print >> self.out, "  -- Pade's approximation "
        
        mbd = self.ParsBlock('^\%MixBasis')
        if mbd:
            self.kmr = float(mbd[0][0])
            self.lmbmax, self.wftol, self.lblmax = int(mbd[1][0]), float(mbd[1][1]), int(mbd[1][2])
        else:
            self.kmr = 1.0
            self.lmbmax = 3
            self.lblmax = self.lmbmax*2
            self.wftol = 1e-4
        print('Product (mixed) basis parameters', file=self.out)
        print('  kmr=', self.kmr, '-- Interstitial: Maximum |G| for plane waves. Note RKmax_{PB}=RKmax_{in1}*kmr', file=self.out)
        print('  lmbmax=', self.lmbmax, '-- MT-Spheres: Maximum l for products', file=self.out)
        print('  wftol=', self.wftol, '-- Linear dependence tolerance, i.e., disregard basis functions with smaller singular values', file=self.out)
        print('  MB_emin=', self.mb_emin, '-- low-energy cutoff (in linearization energy) for radial functions included in the product basis', file=self.out)
        print('  MB_emax=', self.mb_emax, '-- high-energy cutoff (in linearization energy) for radial functions included in the product basis', file=self.out)
        self.nspin_mb = 1
        #self.ibgw = -1   # default stating index for GW calculation. -1 means it will be set later.
        #self.nbgw = -1   # default last band index for GW calculation, -1 means it will be set later
        self.ibgw = int( self.Pars("ibgw\s*=\s*(\d+)",-1) )
        self.nbgw = int( self.Pars("nbgw\s*=\s*(\d+)",-1) )
        if self.ibgw>=0: print('ibgw=', self.ibgw, '-- first band consider in gw', file=self.out) 
        if self.nbgw>=0: print('nbgw=', self.nbgw, '-- last band consider in gw', file=self.out) 
        self.barcevtol = float( self.Pars("barcevtol\s*=\s*("+self.float+")",-1e-10) )
        #print >> self.out, 'barcevtol=', self.barcevtol
        
        self.lvorb = False
        
        #     Read the parameters for the Bare coulomb potential
        self.pwm, self.stctol =  2.0, 1e-15
        mbd = self.ParsBlock('^\%BareCoul')
        self.pwm    = float(mbd[0][0])
        self.stctol = float(mbd[0][1])
        
        # set the trancation radius for the bare Coulomb interaction, needed for finite systems
        self.iop_coulvm, self.iop_coul_x, self.iop_coul_c = 0, 0, 0
        self.rcut_coul = -1.0
        print('Parameters for Coulomb matrix:', file=self.out)
        print("  pwm="+str(self.pwm), "-- Maximum |G| in kmr units ", file=self.out)
        print("  stctol="+str(self.stctol)+ "-- Error tolerance for struc. const in Ewald summation", file=self.out)
        #print >> self.out, "  Coulomb interaction for exchange",    self.iop_coul_x 
        #print >> self.out, "  Coulomb interaction for correlation", self.iop_coul_c 
        #print >> self.out, '-'*55
        
        #mbd = self.ParsBlock('^\%gw')
        #self.iop_sxc, self.iop_vxc = 0, 0
        #if mbd:
        #    self.iop_sxc = int(mbd[0][0])
        #    self.iop_vxc = int(mbd[0][1])
        #print >> self.out, 'gw: sxc=%d vxc=%d' % (self.iop_sxc, self.iop_vxc)

        MatrixSelfEnergy = self.Pars("MatrixSelfEnergy\s*=\s*([T|F])",'F')
        if MatrixSelfEnergy=='T':
            self.MatrixSelfEnergy = True
        else:
            self.MatrixSelfEnergy = False
        self.sigma_off_ratio = float( self.Pars("sigma_off_ratio\s*=\s*("+self.float+")",1e-2) )

        print('Parameters for off-diagonal Self-energy:', file=self.out)
        print('  MatrixSelfEnergy=', self.MatrixSelfEnergy, file=self.out)
        print('  sigma_off_ratio=', self.sigma_off_ratio, file=self.out)
        
        print('-'*32, ' Finished  inout.py:InOut::ParsAll() ', '-'*32, file=self.out)
        self.out.flush()
