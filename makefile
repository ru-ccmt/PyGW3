# MAC
F2P = f2py
OPT  = --fcompiler=gnu95
OPTM = 
mangling = cpython-38-darwin.

# linux
#F2P = f2py3
#OPT = --link-lapack_opt
#OPTM = -lgomp -lpthread
#mangling = cpython-36m-x86_64-linux-gnu.


CMP  = $(F2P) --opt='-O3 -ffree-line-length-none'  $(OPT)
CMPM = $(F2P) --opt='-O3 -ffree-line-length-none -fopenmp'  $(OPT) $(OPTM)

FORT = gfortran
FFLAGS = -O2 -ffree-line-length-none -fopenmp


modules = fnc.$(mangling)so cum_simps.$(mangling)so gaunt.$(mangling)so rdVec.$(mangling)so radials.$(mangling)so radd.$(mangling)so for_kpts.$(mangling)so for_tetrahedra.$(mangling)so for_vxcnn.$(mangling)so lapwcoeff.$(mangling)so for_Coulomb.$(mangling)so for_q_0.$(mangling)so for_pade.$(mangling)so sphbes.$(mangling)so
objects = sphbes.o for_tetrahedra2.o for_tetrahedra3.o for_pade2.o

all : $(objects) $(modules) 

lapwcoeff.$(mangling)so : lapwcoeff.f90 sphbes.o
	$(CMP) -c $< sphbes.o -m lapwcoeff

for_Coulomb.$(mangling)so : for_Coulomb.f90 sphbes.o
	$(CMP) -c $< sphbes.o -m for_Coulomb

for_tetrahedra.$(mangling)so : for_tetrahedra.f90 for_tetrahedra2.o for_tetrahedra3.o
	$(CMPM) -c $< for_tetrahedra2.o for_tetrahedra3.o -m for_tetrahedra

for_q_0.$(mangling)so : for_q_0.f90
	$(CMPM) -c for_q_0.f90 -m for_q_0 

for_vxcnn.$(mangling)so : for_vxcnn.f90
	$(CMPM) -c for_vxcnn.f90 -m for_vxcnn

for_pade.$(mangling)so : for_pade.f90 for_pade2.o
	$(CMP) -c $< for_pade2.o -m for_pade

clean :
	rm -f $(modules) $(objects) *.pyc *.mod *.o

%.$(mangling)so : %.f90
	$(CMP) -c $< -m $(*F)

%.o : %.f90
	$(FORT) -fPIC -c $(FFLAGS) $<
