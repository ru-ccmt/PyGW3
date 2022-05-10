from numpy import array,shape,zeros,copy
import for_tetrahedra as ft #


class Kweights:
    def __init__(self, io, ks, kqm, fout):
        # setkiw(io, ks, kqm, fout):
        """Computes tetrahedron weights for Green's function like objects
        """
        wkir = array(kqm.weight)
        (nspin,nkir,nbnd) = shape(ks.Ebnd)
        ankp = kqm.ndiv[0]*kqm.ndiv[1]*kqm.ndiv[2]
        _ankp_ = float(ankp)
        
        if io.iop_bzint == 0:   # perform BZ integration over tetrahedra in the IBZ
            isp=0
            self.kiw = ft.intw(ks.Ebnd[isp], ks.EF, kqm.atet, kqm.wtet, io.iop_bcor==1)
            self.kwfer = zeros(shape(self.kiw))
            if (ks.Eg <= 0): # metallic
                self.kwfer = ft.intwsurf(ks.Ebnd[isp], ks.EF, kqm.atet, kqm.wtet)
            
        elif io.iop_bzint == 1:  # Set the weights by the smearing
            (ns,nkir,nbnd) = shape(ks.Ebnd)
            isp=0
            self.kwfer = zeros((nkir,nbnd))
            self.kiw   = zeros((nkir,nbnd))
            wkir_nkp = wkir/_ankp_
    
            self.kiw = cfermi(ks.Ebnd[isp].flatten()/io.esmear).reshape((nkir,nbnd))
            if (ks.Eg <= 0): # metallic
                self.kwfer = cgauss(ks.Ebnd[isp].flatten(),io.esmear).reshape((nkir,nbnd))
    
            # Finding bands which are either not fully empty (nomx) or not fully occupied (numn)
            nomx, numn = ks.nomax_numin
            nomx = max([len([x for x in self.kiw[ik,:] if x>1e-4]) for ik in range(len(self.kiw))]) # band at which the weight is very small
            numn = min([nbnd-len([x for x in self.kiw[ik,:] if x<1-1e-4]) for ik in range(len(self.kiw))]) # band which is not fully occupied, and weight deviates from 1 for tiny amount
                
            for ik in range(nkir):
                self.kiw  [ik,:] *= wkir_nkp[ik]
                self.kwfer[ik,:] *= wkir_nkp[ik]
                
            print('nomx=', nomx, 'numn=', numn, file=fout)
            if nomx > ks.nomax_numin[0]:
                print(" nomax in terms of the occupation:", nomx, file=fout)
                ks.nomax_numin[0] = nomx
            
            if numn < ks.nomax_numin[1]:
                print(" numin in terms of the occupation:", numn, file=fout) 
                ks.nomax_numin[1] = numin
        else:
            print("ERROR: unsupported option iop_bzint=", io.iop_bzint, file=fout)
            sys.exit(1)
            
        
        print('tetrahedron weight: kiw kwfer and trivial_weight:', file=fout)
        for ik in range(nkir):
            for ib in range(nbnd):
                if self.kiw[ik,ib]!=0 or self.kwfer[ik,ib]!=0:
                    print('%3d %3d %16.10f   %16.10f   %16.10f' % (ik+1, ib+1, self.kiw[ik,ib], self.kwfer[ik,ib], wkir[ik]/_ankp_), file=fout)
    
        # set up the band-independent k-weight in the IBZ                
        self.kwt_ibz = copy(self.kiw[:,0]) # should be equal to wkir/ankp or kqm.weight/ankp
        
        # kiw and kwfer correspond to the weight for an irreducible point.
        # If we work with all (even reducible) k-points, the weight has to be divided by wkir
        for ik in range(nkir):            
            self.kiw  [ik,:] *= 1.0/wkir[ik]
            self.kwfer[ik,:] *= 1.0/wkir[ik]
            
        # set up the band-independent k-weight in the RBZ
        kwt_bz = zeros(ankp)
        for ik in range(ankp):
            irk = kqm.kii_ind[ik]
            kwt_bz[ik] = self.kwt_ibz[irk]  # should be equal to 1/ankp
        
        print('tetrahedron weight: kiw kwfer and trivial_weight:', file=fout)
        for ik in range(nkir):
            for ib in range(nbnd):
                if self.kiw[ik,ib]!=0 or self.kwfer[ik,ib]!=0:
                    print('%3d %3d %16.10f   %16.10f  %16.10f' % (ik+1, ib+1, self.kiw[ik,ib], self.kwfer[ik,ib], 1-ankp*self.kiw[ik,ib]), file=fout)
        
        #print >> fout, 'k-points weights for all points'
        #for ik in range(ankp):
        #    print >> fout, '%3d %10.6f' % (ik, kwt_bz[ik])
