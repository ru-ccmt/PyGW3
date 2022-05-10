from numpy import *
from numpy import linalg

def svd_functions(iomega, wiomeg, om_min, om_max, om_nom, svd_cutoff, fout):
    print('******* SVD of Matsubara Kernel for frequency dependence *******', file=fout)
    print('svd_cutoff=', svd_cutoff, 'real axis mesh has min='+str(om_min)+'H max='+str(om_max)+'H and nom='+str(om_nom), file=fout)
    # The real frequency mesh
    om, dom = mcmn.Give2TanMesh(om_min,om_max,om_nom)  
    # The imaginary frequency mesh for positive and negative frequency
    iom  = hstack( (-iomega[::-1], iomega[:]) )
    diom = hstack( ( wiomeg[::-1], wiomeg[:]) )
    # bosonic kernel for real part
    Ker = zeros((len(iom),len(om)))
    num = om*dom/pi
    om2 = om**2
    for i,iw in enumerate(iom):
        Ker[i,:] = num*sqrt(diom[i])/(om2+iom[i]**2)
    u, s, vh = linalg.svd(Ker, full_matrices=True)
    n = where(s < svd_cutoff)[0][0]
    print('singular eigenvalues kept=', file=fout)
    for i in range(n):
        print("%2d %16.10f" % (i+1, s[i]), file=fout)
    Ul = transpose(u[:,:n])
    Ul *= 1/sqrt(diom)
    N2 = len(iom)/2
    dUl = (Ul[:,N2:]*diom[N2:]*2.0).T  # This is used to compute coefficients cl. Needs factor of two, because we integrate only over positive Matsubara frequencies
    return (Ul[:,N2:], dUl)
