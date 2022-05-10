from numpy import array

def Check_Equal_k_lists(klist, kqm, fout):
    klst = array(klist)
    Equal_k_lists = True
    if len(klist) == len(kqm.kirlist):
        for ik in range(len(klist)):
            if sum(abs(klst[ik,:] - kqm.kirlist[ik,:]/float(kqm.LCM))) > 1e-5:
                Equal_k_lists = False
    else:
        Equal_k_lists = False
        
    if not Equal_k_lists:
        print('Num. irr. k-points=', len(kqm.kirlist), 'while in energy file irr k-points=', len(klist), file=fout)
        print('ERROR: the irreducible k-mesh generated here is inconsistent with that used in WIEN2k!!!', file=fout)
        print(' - there may be some bugs in libbzint ', file=fout)
        print(' - use KS-vectors in the full BZ (check the flag \'-sv\' in gap2_init) to avoid such problems', file=fout)
        print('k-points from vector file:', file=fout)
        for ik in range(len(klist)):
            print(klist[ik], file=fout)
        print('k-points generated here:', file=fout)
        for ik in range(len(kqm.kirlist)):
            print(kqm.kirlist[ik,:]/float(kqm.LCM), file=fout)
        sys.exit(1)

