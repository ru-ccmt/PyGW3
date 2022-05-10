from numpy import *
import re
import os

Ry2H = 0.5         # Rydberg2Hartree
H2Ry = 2.0
H2eV = 27.2113961  # Hartree2eV
Ry2eV= 13.60569805 # 
FORT = True        # use fortran modules
sr2pi = sqrt(2*pi)
b2MB = 1./1024**2
Br2Ang = 0.5291772106712
import numpy
if "lcm" not in dir(numpy):
    def lcm(a, b):
        if a > b:
            greater = a
        else:
            greater = b
        while True:
            if greater % a == 0 and greater % b == 0:
                lcm = greater
                break
            greater += 1
        return lcm

def fermi(x):
    return 1/(exp(x)+1.)
def gauss(x, s):
    return 1/(s*sr2pi)*exp(-x**2/(2*s**2))
def _cfermi_(x):
    if x > 10.:
        return 0.0
    elif x< -10:
        return 1.0
    else:
        return fermi(x)
cfermi = vectorize(_cfermi_)
def _cgauss_(x,s):
    if abs(x) > 10*s:
        return 0.0
    else:
        return gauss(x,s)
cgauss = vectorize(_cgauss_, excluded=['s'])

