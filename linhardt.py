from numpy import *
import for_tetrahedra as ft #

def Polarization_weights(iq, ks, kqm, core, fr, iop_bzintq, sgnfrq, dUl, fout, PartialTetra=True, isp=0):
    nomx, numin = ks.nomax_numin
    ankp = len(kqm.kii_ind)

    # kcw(No-Nactual,Ne-Neactual,nom,nkp)= kcw(ks.nomax_numin[0]-ks.ibmin_tetra, ks.ibmax_tetra-ks.nomax_numin[1],nom,nkp )
    if PartialTetra:
        nbnd = ks.ibmax_tetra-ks.ibmin_tetra-1
        enk = zeros((nbnd, ankp), order='F')
        for ik in range(ankp):
            irk = kqm.kii_ind[ik]
            enk[:,ik] = ks.Ebnd[isp,irk,ks.ibmin_tetra+1:ks.ibmax_tetra]
        # kcw(nomx-ks.ibmin_tetra, ks.ibmax_tetra-numin, nom,nkp)
        kcw = ft.bz_calcqdepw_par2(enk, ks.EF, kqm.kqid[:,iq], 0, nomx-ks.ibmin_tetra-1, numin-ks.ibmin_tetra-1, fr.omega, kqm.tetc)
    else:
        nbnd = ks.ncg_p + ks.nbmaxpol
        enk = zeros((nbnd, ankp), order='F')
        
        for icg in range(ks.ncg_p):
          iat, idf, ic = core.corind[icg][0:3]
          enk[icg,:] = core.eig_core[isp][iat][ic]
        for ik in range(ankp):
            irk = kqm.kii_ind[ik]
            enk[ks.ncg_p:(ks.ncg_p+ks.nbmaxpol),ik] = ks.Ebnd[isp,irk,:ks.nbmaxpol]
        
        if False:
            nmax=ks.ncg_p+ks.nbmaxpol
            print('Energies', file=fout)
            for ik in range(ankp):
                print('%3d ' % ik, ('%10.6f'*nmax) % tuple(enk[:,ik]), file=fout)

        if dUl is not None:
            kcw = ft.bz_calcqdepw_par3(enk, ks.EF, kqm.kqid[:,iq], ks.ncg_p, nomx, numin, dUl, fr.omega, kqm.tetc)
        else:
            kcw = ft.bz_calcqdepw_par2(enk, ks.EF, kqm.kqid[:,iq], ks.ncg_p, nomx, numin, fr.omega, kqm.tetc)
            
        if False:
            save('enk',enk)
            save('kqid', kqm.kqid)
            save('tetc', kqm.tetc)
            rest = {'EF': ks.EF, 'ncg': ks.ncg_p, 'nomx': nomx, 'numin':numin}
            print(rest)
            import pickle
            f = open("rest.pkl","wb")
            pickle.dump(rest,f)
            f.close()
            save('kcw.'+str(iq), kcw)
            
        if False:
            #fl = sorted(glob.glob('kcw.*'))
            #if len(fl)>0:
            #    n = int(fl[-1].split('.')[-1])
            #else:
            #    n=0
            #fnm = 'kcw.'+str(n+1)
            fnm = 'kcw.0'
            fo = open(fnm, 'w')
            print('nk=', shape(kcw)[2], 'shape(kcw)=', shape(kcw), 'kcw=', file=fo)
            (nb1,nb2,nom,nkp) = shape(kcw)
            for ik in range(nkp):
                for ib in range(ks.ncg_p+nomx+1):
                    for jb in range(nb2):
                        print('%4d %4d %4d' % (ik+1, ib+1, jb+numin+1), ('%17.13f'*nom) % tuple(kcw[ib,jb,:,ik].real), file=fo)
            fo.close()
            sys.exit(0)
    
    return kcw
