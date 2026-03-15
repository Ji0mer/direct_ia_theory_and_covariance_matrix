from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import numpy as np
import scipy.interpolate as spi
import scipy.integrate as sint
import mcfit
from mcfit import P2xi, xi2P
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.special import eval_legendre as legendre
from scipy import integrate
from astropy.cosmology import FlatLambdaCDM
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import json
import os

import time

def wgg_calc_parts_g(xii,alpha,bg1,bg2):
    return (xii.T*alpha*bg1*bg2).T

def compute_xii_g(selfj,selfrG,selfL,selfdpi):
    ri,xi_i = selfj
    xi_intp=interp1d(ri,xi_i,bounds_error=False,fill_value=0)
    xii = np.dot((xi_intp(selfrG)*selfL),selfdpi)
    xii *= 2
    return xii

def pk_terms_sum(bv1,bv2,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8):
    return ( bv1[0]*bv2[0]*pk1+0.5*(bv1[0]*bv2[1]+bv2[0]*bv1[1])*pk2+
                0.25*bv1[1]*bv2[1]*pk3+0.5*(bv1[0]*bv2[2]+bv2[0]*bv1[2])*pk4+
                0.25*(bv2[1]*bv1[2]+bv1[1]*bv2[2])*pk5+0.25*bv1[2]*bv2[2]*pk6+
                0.5*(bv1[0]*bv2[3]+bv2[0]*bv1[3])*pk7+(bv1[0]*bv2[4]+bv2[0]*bv1[4])*pk8 )
def return_pk_terms(bv1,bv2):
    c1 = bv1[0]*bv2[0]
    c2 = 0.5*(bv1[0]*bv2[1]+bv2[0]*bv1[1])
    c3 = 0.25*bv1[1]*bv2[1]
    c4 = 0.5*(bv1[0]*bv2[2]+bv2[0]*bv1[2])
    c5 = 0.25*(bv2[1]*bv1[2]+bv1[1]*bv2[2])
    c6 = 0.25*bv1[2]*bv2[2]
    c7 = 0.5*(bv1[0]*bv2[3]+bv2[0]*bv1[3])
    c8 = bv1[0]*bv2[4]+bv2[0]*bv1[4]
    return c1,c2,c3,c4,c5,c6,c7,c8

# Check whether the folder contains any files
def folder_has_files(folder_path):
    # List the contents of the folder
    return any(os.path.isfile(os.path.join(folder_path, f)) for f in os.listdir(folder_path))

def interp_power(input_k,input_z,input_power,knew,znew):
    """
    if (input_power>0).all():
        inter = interp2d(np.log10(input_k), input_z, np.log10(input_power))
        #import pdb ; pdb.set_trace()
        Pnew = 10**inter(np.log10(knew), znew)
    else:
        #import pdb ; pdb.set_trace()
        inter = interp2d(np.log10(input_k), input_z, np.log10(-input_power))
        Pnew = -10**inter(np.log10(knew), znew)
    """
    mini_power = np.min(input_power)
    modified_power = input_power - mini_power + 10
    inter = interp2d( np.log10(input_k), input_z, np.log10(modified_power) )
    Pnew = inter( np.log10(knew),znew )
    Pnew = 10**(Pnew) - 10 + mini_power
    return Pnew

y3fid_cosmology = FlatLambdaCDM(H0=69., Om0=0.30, Ob0=0.048)

def setup(options):
    sample_a = options.get_string(option_section, "sample_a", default="lens lens").split()
    sample_b = options.get_string(option_section, "sample_b", default="lens source").split()
    rmin = options.get_double(option_section, "rpmin", default=0.01)
    rmax = options.get_double(option_section, "rpmax", default=500.)
    nr = options.get_int(option_section, "nr", default=1024)
    nk = options.get_int(option_section, "nk", default=200)

    rp = np.logspace(np.log10(rmin), np.log10(rmax), nr)

    pimax = options.get_double(option_section, "pimax", default=100.) # in h^-1 Mpc

    corrs = options.get_string(option_section, "correlations", default="wgp").split()

    do_rsd = options.get_bool(option_section, "include_rsd", default=False)
    do_lensing = options.get_bool(option_section, "include_lensing", default=False)
    do_magnification = options.get_bool(option_section, "include_magnification", default=False)
    wgg_folder = options.get_string(option_section,"wgg_folder")
    pks_folder = options.get_string(option_section,"pks_folder")
    os.makedirs(wgg_folder, exist_ok=True)

    cl_dir = options.get_string(option_section, "cl_loc", default="")

    if do_rsd:
        print('will include RSDs (Pi_max = %3.1f)'%pimax)
    else:
        print('will not include RSDs :( ')
        print("redshift space will not be distorted, and it's your fault...")

 
    return sample_a, sample_b, rp, pimax, nk, corrs, do_rsd, do_lensing, do_magnification, cl_dir,pks_folder,wgg_folder



def execute(block, config):

    sample_a,sample_b,rp,pimax,nk,corrs,do_rsd, do_lensing, do_magnification, cl_dir, pks_folder, wgg_folder = config

    k = block['galaxy_power', 'k_h']
    print(k.min(),k.max())
    #with open("/home/jiomer/research/direct_ia_theory/test_output/k.txt", 'a') as f_k:
    #    np.savetxt(f_k,[k.min(),k.max(),nk],fmt='%e')
    knew = np.logspace(np.log10(0.001), np.log10(k.max()), nk)
    X = Projected_Corr_RSD(rp=rp, pi_max=pimax, k=knew, lowring=True)

    # bookkeeping
    pknames = {'wgg':'galaxy_power', 'wgp':'galaxy_intrinsic_power'}


    if do_rsd:
        fz = block['growth_parameters', 'f_z']
        z1 = block['growth_parameters', 'z']
        #interp = interp1d(z,f0)
        #fz = interp(0.27)
        beta2 = -1

        Dz=block['growth_parameters', 'd_z']/block['growth_parameters', 'd_z'][0]
        lnD=np.log(Dz)
        lna=np.log(block['growth_parameters', 'a'])
        fz = np.gradient(lnD,lna)
    else:
        fz = 0.
        beta2 = 0.
        z1 = block['growth_parameters', 'z']
        
    

    print (corrs)

    for c,s1,s2 in zip(corrs,sample_a,sample_b):
        print(c,s1,s2)

        if ('bias_parameters','b_%s'%s1) in block.keys():
            ba = block['bias_parameters', 'b_%s'%s1]
        else:
            ba = 1.

        if ('bias_parameters','b_%s'%s2) in block.keys():
            bb = block['bias_parameters', 'b_%s'%s2]
        else:
            bb = 1.

        P = block[pknames[c],'p_k']
        z = block[pknames[c],'z']
        #import pdb ; pdb.set_trace()
        
        """
        if (P>0).all():
            inter = interp2d(np.log10(k), z, np.log10(P))
            #import pdb ; pdb.set_trace()
            Pnew = 10**inter(np.log10(knew), z1)
        else:
            #import pdb ; pdb.set_trace()
            inter = interp2d(np.log10(k), z, np.log10(-P))
            Pnew = -10**inter(np.log10(knew), z1)
        """
        
        
        if (c=='wgg'):
            pks_path = pks_folder
            wggs_path = wgg_folder
            
            za, W = get_redshift_kernel(block, 0, 0, z1, block['distances','d_m'], s1, s2)
            #time_array.append( time.time() )
            #import pdb ; pdb.set_trace()
            #Wofz = interp1d(W,za)
            K = np.array([W]*len(X.rp))

            z0 = np.trapz(za*W,za)
            #time_array.append( time.time() )

            if do_magnification:
                Pnew = add_gg_mag_terms(block, Pnew, za, knew, z0, s1, s2, cl_dir=cl_dir)
            bb = ba
            #time_array.append( time.time() )
            # print('BIAS : %f %f'%(ba,bb))

            bv1 = block[pknames[c],"bias_values_a[bin1]"]
            bv2 = block[pknames[c],"bias_values_b[bin2]"]

            if folder_has_files(wggs_path):
                beta1 = fz/ba
                beta2 = fz/bb

                # load xi from npz files
                # calculate correlation function w 
                # components and sum top get wgg

                xi1 = {}
                xi1[0] = np.load(wggs_path+"xi10.npz")['arr_0']
                xi1[2] = np.load(wggs_path+"xi12.npz")['arr_0']
                xi1[4] = np.load(wggs_path+"xi14.npz")['arr_0']

                xi2 = {}
                xi2[0] = np.load(wggs_path+"xi20.npz")['arr_0']
                xi2[2] = np.load(wggs_path+"xi22.npz")['arr_0']
                xi2[4] = np.load(wggs_path+"xi24.npz")['arr_0']

                xi3 = {}
                xi3[0] = np.load(wggs_path+"xi30.npz")['arr_0']
                xi3[2] = np.load(wggs_path+"xi32.npz")['arr_0']
                xi3[4] = np.load(wggs_path+"xi34.npz")['arr_0']

                xi4 = {}
                xi4[0] = np.load(wggs_path+"xi40.npz")['arr_0']
                xi4[2] = np.load(wggs_path+"xi42.npz")['arr_0']
                xi4[4] = np.load(wggs_path+"xi44.npz")['arr_0']

                xi5 = {}
                xi5[0] = np.load(wggs_path+"xi50.npz")['arr_0']
                xi5[2] = np.load(wggs_path+"xi52.npz")['arr_0']
                xi5[4] = np.load(wggs_path+"xi54.npz")['arr_0']

                xi6 = {}
                xi6[0] = np.load(wggs_path+"xi60.npz")['arr_0']
                xi6[2] = np.load(wggs_path+"xi62.npz")['arr_0']
                xi6[4] = np.load(wggs_path+"xi64.npz")['arr_0']

                xi7 = {}
                xi7[0] = np.load(wggs_path+"xi70.npz")['arr_0']
                xi7[2] = np.load(wggs_path+"xi72.npz")['arr_0']
                xi7[4] = np.load(wggs_path+"xi74.npz")['arr_0']

                xi8 = {}
                xi8[0] = np.load(wggs_path+"xi80.npz")['arr_0']
                xi8[2] = np.load(wggs_path+"xi82.npz")['arr_0']
                xi8[4] = np.load(wggs_path+"xi84.npz")['arr_0']

                xisum = {}
                for i in [0,2,4]:
                    xisum[i] = pk_terms_sum(bv1,bv2,xi1[i],xi2[i],xi3[i],xi4[i],xi5[i],xi6[i],xi7[i],xi8[i])/block["galaxy_power","blin_1"] / block["galaxy_power","blin_2"]
                
                W,xisum = X.xi_wgg(f=fz, bg=ba, bg2=bb, pk=[], xi=xisum, l=[0,2,4])

                #P = block[pknames[c],'p_k']
                #Pnew = interp_power(k,z,P,knew,z1)
                #Wnew,xinew = X.wgg_calc(f=fz, bg=ba, bg2=bb, pk=Pnew, xi=None, l=[0,2,4])

            else:
                # load pk terms from npz files
                # using get_xi get pk terms' xi
                # save xi for fast computation 

                P = block[pknames[c],'p_k']
                Pterm1 = np.load( pks_path+"Pk1_Pd1d1.npz")['arr_0']
                Pterm2 = np.load( pks_path+"Pk2_Pd1d2.npz")['arr_0']
                Pterm3 = np.load( pks_path+"Pk3_Pd2d2.npz")['arr_0']
                Pterm4 = np.load( pks_path+"Pk4_Pd1s2.npz")['arr_0']
                Pterm5 = np.load( pks_path+"Pk5_Pd2s2.npz")['arr_0']
                Pterm6 = np.load( pks_path+"Pk6_Ps2s2.npz")['arr_0']
                Pterm7 = np.load( pks_path+"Pk7_sig3nl.npz")['arr_0']
                Pterm8 = np.load( pks_path+"Pk8_k2P.npz")['arr_0']

                Pnew = interp_power(k,z,P,knew,z1)
                P1new = interp_power(k,z,Pterm1,knew,z1)
                P2new = interp_power(k,z,Pterm2,knew,z1)
                P3new = interp_power(k,z,Pterm3,knew,z1)
                P4new = interp_power(k,z,Pterm4,knew,z1)
                P5new = interp_power(k,z,Pterm5,knew,z1)
                P6new = interp_power(k,z,Pterm6,knew,z1)
                P7new = interp_power(k,z,Pterm7,knew,z1)
                P8new = interp_power(k,z,Pterm8,knew,z1)

                Wnew,xinew = X.wgg_calc(f=fz, bg=ba, bg2=bb, pk=Pnew, xi=None, l=[0,2,4])
                
                xi1 = X.get_xi(P1new,l=[0,2,4])
                xi2 = X.get_xi(P2new,l=[0,2,4])
                xi3 = X.get_xi(P3new,l=[0,2,4])
                xi4 = X.get_xi(P4new,l=[0,2,4])
                xi5 = X.get_xi(P5new,l=[0,2,4])
                xi6 = X.get_xi(P6new,l=[0,2,4])
                xi7 = X.get_xi(P7new,l=[0,2,4])
                xi8 = X.get_xi(P8new,l=[0,2,4])
                
                w1,xi1 = X.xi_wgg(f=fz, bg=ba, bg2=bb, pk=P1new, xi=xi1, l=[0,2,4])
                w2,xi2 = X.xi_wgg(f=fz, bg=ba, bg2=bb, pk=P2new, xi=xi2, l=[0,2,4])
                w3,xi3 = X.xi_wgg(f=fz, bg=ba, bg2=bb, pk=P3new, xi=xi3, l=[0,2,4])
                w4,xi4 = X.xi_wgg(f=fz, bg=ba, bg2=bb, pk=P4new, xi=xi4, l=[0,2,4])
                w5,xi5 = X.xi_wgg(f=fz, bg=ba, bg2=bb, pk=P5new, xi=xi5, l=[0,2,4])
                w6,xi6 = X.xi_wgg(f=fz, bg=ba, bg2=bb, pk=P6new, xi=xi6, l=[0,2,4])
                w7,xi7 = X.xi_wgg(f=fz, bg=ba, bg2=bb, pk=P7new, xi=xi7, l=[0,2,4])
                w8,xi8 = X.xi_wgg(f=fz, bg=ba, bg2=bb, pk=P8new, xi=xi8, l=[0,2,4])
                w1 = w1 / block["galaxy_power","blin_1"] / block["galaxy_power","blin_2"]
                w2 = w2 / block["galaxy_power","blin_1"] / block["galaxy_power","blin_2"]
                w3 = w3 / block["galaxy_power","blin_1"] / block["galaxy_power","blin_2"]
                w4 = w4 / block["galaxy_power","blin_1"] / block["galaxy_power","blin_2"]
                w5 = w5 / block["galaxy_power","blin_1"] / block["galaxy_power","blin_2"]
                w6 = w6 / block["galaxy_power","blin_1"] / block["galaxy_power","blin_2"]
                w7 = w7 / block["galaxy_power","blin_1"] / block["galaxy_power","blin_2"]
                w8 = w8 / block["galaxy_power","blin_1"] / block["galaxy_power","blin_2"]
                
                xisum = {}
                for i in [0,2,4]:
                    xisum[i] = pk_terms_sum(bv1,bv2,xi1[i],xi2[i],xi3[i],xi4[i],xi5[i],xi6[i],xi7[i],xi8[i])
                W = pk_terms_sum(bv1,bv2,w1,w2,w3,w4,w5,w6,w7,w8)
                
                for key,value in xi1.items():
                    np.savez(wggs_path+"xi1"+str(key)+".npz",value)
                for key,value in xi2.items():
                    np.savez(wggs_path+"xi2"+str(key)+".npz",value)
                for key,value in xi3.items():
                    np.savez(wggs_path+"xi3"+str(key)+".npz",value)
                for key,value in xi4.items():
                    np.savez(wggs_path+"xi4"+str(key)+".npz",value)
                for key,value in xi5.items():
                    np.savez(wggs_path+"xi5"+str(key)+".npz",value)
                for key,value in xi6.items():
                    np.savez(wggs_path+"xi6"+str(key)+".npz",value)
                for key,value in xi7.items():
                    np.savez(wggs_path+"xi7"+str(key)+".npz",value)
                for key,value in xi8.items():
                    np.savez(wggs_path+"xi8"+str(key)+".npz",value)
                for key,value in xinew.items():
                    np.savez(wggs_path+"xi"+str(key)+".npz",value)
                #with open(pks_path+"diffw.txt","a") as fdiffw:
                #    np.savetxt(fdiffw,Wnew-W,fmt="%e")
                #    fdiffw.write("\n")
                #    fdiffw.write("\n")

                np.savez(pks_path+"p1new.npz",P1new)
                np.savez(pks_path+"p2new.npz",P2new)
                np.savez(pks_path+"p3new.npz",P3new)
                np.savez(pks_path+"p4new.npz",P4new)
                np.savez(pks_path+"p5new.npz",P5new)
                np.savez(pks_path+"p6new.npz",P6new)
                np.savez(pks_path+"p7new.npz",P7new)
                np.savez(pks_path+"p8new.npz",P8new)
                np.savez(pks_path+"pnew.npz",Pnew)
                np.savez(pks_path+"bv1.npz",bv1)
                np.savez(pks_path+"bv2.npz",bv2)
                np.savez(pks_path+"rp.npz",rp)
                np.savez(pks_path+"knew.npz",knew)
                np.savez(pks_path+"fz.npz",fz)
                np.savez(pks_path+"babb.npz",[ba,bb])
                np.savez(pks_path+"w.npz",Wnew)

            #with open(pks_path+"diffxi.txt","a") as fdiffw:
            #    np.savetxt(fdiffw, [np.max(abs(xinew[0]-xisum[0])/xinew[0]),np.max(abs(xinew[2]-xisum[2])/xinew[2]),np.max(abs(xinew[4]-xisum[4])/xinew[4])] ,fmt="%e")
            #    fdiffw.write("\n")


        elif (c=='wgp'):
            za, W = get_redshift_kernel(block, 0, 0, z1, block['distances','d_m'], s1, s2)
           # Wofz = interp1d(W,za)
            K = np.array([W]*len(X.rp))

            z0 = np.trapz(za*W,za)
            #import pdb ; pdb.set_trace()

            W = X.wgm_calc(f=fz, bg=ba, beta2=beta2, pk=-Pnew, xi=None, l=[0,2,4]) 


            #W*=np.sqrt(2.) 
                      

    #    import pdb ; pdb.set_trace()
        
        integrand = K.T * W
        #integrand_new = K.T * Wnew
        W_flat = sint.trapz(integrand,z1,axis=0) / sint.trapz(K.T,z1,axis=0)
        #W_flat_new = sint.trapz(integrand_new,z1,axis=0) / sint.trapz(K.T,z1,axis=0)
        #with open(pks_path+"diffwflat.txt","a") as fdiffw:
        #    np.savetxt(fdiffw,[(W_flat-W_flat_new)/W_flat_new],fmt="%e")
        #    fdiffw.write("\n")
        #    fdiffw.write("\n")
        

        section = pknames[c].replace('_power','_w')
        block.put_double_array_1d(section, 'w_rp_1_1_%s_%s'%(s1,s2), W_flat)
        try:
        	block.put_double_array_1d(section, 'r_p', X.rp)
        except:
        	block.replace_double_array_1d(section, 'r_p', X.rp) 

        #if (c=='wgp'):
        #    import pdb ; pdb.set_trace()

    #time_array = np.array(time_array) - time_array[0]
    #np.savetxt("/home/jiomer/research/direct_ia_theory/test_output/maintimepoints.txt",time_array,fmt="%e")
        
    return 0




def add_gg_mag_terms(block, Pnew, z, k, z0, s1,s2, cl_dir=""):
    
    h0 = y3fid_cosmology.h


    if (len(cl_dir)==0):
        c_mm = block['magnification_cl', 'bin_1_1']
        c_gm = block['magnification_galaxy_cl', 'bin_1_1']
        c_mI = block['magnification_intrinsic_cl', 'bin_1_1']
        ell = block['magnification_cl', 'ell']
    else:
        dsample1 = s1.replace('_density', '')
        dsample2 = s2.replace('_density', '')


        c_mm = np.loadtxt("%s/magnification_cl_%s_%s.txt"%(cl_dir, dsample1, dsample2))
        c_mg = np.loadtxt("%s/magnification_galaxy_cl_%s_%s.txt"%(cl_dir, dsample1, dsample2))

        ell = np.loadtxt("%s/ell.txt"%cl_dir)
    
    
    p_mm = c_mm * y3fid_cosmology.comoving_transverse_distance(z0).value**3 / 2 * h0 * h0 * h0 #this is an approximation
    k_mm = ell / y3fid_cosmology.comoving_transverse_distance(z0).value/ h0

    p_mg = c_mg * y3fid_cosmology.comoving_transverse_distance(z0).value**3 / 2* h0 * h0 * h0 #this is an approximation
    k_mg = ell / y3fid_cosmology.comoving_transverse_distance(z0).value/ h0



    p_mm_int = interp1d(np.log10(k_mm), p_mm, bounds_error=False, fill_value=0)
    p_mm = p_mm_int(np.log10(k))

    p_mg_int = interp1d(np.log10(k_mg), p_mg, bounds_error=False, fill_value=0)
    p_mg = p_mg_int(np.log10(k))


    Pmm = np.array([p_mm]*len(z))
    Pmg = np.array([p_mg]*len(z))

    print('Adding magnification....')
    
    return Pnew + Pmm + 2*Pmg



def add_gp_lensmag_terms(block, Pnew, z, k, z0, s1, s2, cl_dir=None, do_lensing=True, do_magnification=True):

    # lensing contribution to g+
    # this is a back-of-the-envelope thing Sukhdeep came up with
    # assuming the contribution is small 

    h0 = y3fid_cosmology.h


    if (len(cl_dir)==0):
        c_mm = block['magnification_cl', 'bin_1_1']
        c_gm = block['magnification_galaxy_cl', 'bin_1_1']
        c_mI = block['magnification_intrinsic_cl', 'bin_1_1']
        ell = block['magnification_cl', 'ell']
    else:
        dsample = s1.replace('_density', '')
        ssample = s2.replace('_shape', '')


        c_mI = np.loadtxt("%s/magnification_intrinsic_cl_%s_%s.txt"%(cl_dir, dsample, ssample))
        c_mG = np.loadtxt("%s/magnification_shear_cl_%s_%s.txt"%(cl_dir, dsample, ssample))
        c_gG = np.loadtxt("%s/galaxy_shear_cl_%s_%s.txt"%(cl_dir, dsample, ssample))

        ell = np.loadtxt("%s/ell.txt"%cl_dir)

    p_mI = c_mI * y3fid_cosmology.comoving_transverse_distance(z0).value**3 / 2 * h0 * h0 * h0 #this is an approximation
    k_mI = ell / y3fid_cosmology.comoving_transverse_distance(z0).value/ h0

    p_mG = c_mG * y3fid_cosmology.comoving_transverse_distance(z0).value**3 / 2* h0 * h0 * h0 #this is an approximation
    k_mG = ell / y3fid_cosmology.comoving_transverse_distance(z0).value/ h0

    p_gG = c_gG * y3fid_cosmology.comoving_transverse_distance(z0).value**3 / 2* h0 * h0 * h0 #this is an approximation
    k_gG = ell / y3fid_cosmology.comoving_transverse_distance(z0).value/ h0



    p_mI_int = interp1d(np.log10(k_mI), p_mI, bounds_error=False, fill_value=0)
    p_mI = p_mI_int(np.log10(k))

    p_mG_int = interp1d(np.log10(k_mG), p_mG, bounds_error=False, fill_value=0)
    p_mG = p_mG_int(np.log10(k))

    p_gG_int = interp1d(np.log10(k_gG), p_gG, bounds_error=False, fill_value=0)
    p_gG = p_gG_int(np.log10(k))

    PmI = np.array([p_mI]*len(z))
    PmG = np.array([p_mG]*len(z))
    PgG = np.array([p_gG]*len(z))

    #import pdb ; pdb.set_trace()

    if do_lensing and do_magnification:
        return Pnew + PmI + PmG + PgG
    elif do_lensing and (not do_magnification):
        return Pnew + PgG
    else:
        return Pnew + PmI + PmG

def get_redshift_kernel(block, i, j, z0, x, sample_a, sample_b):

    interp_x = spi.interp1d(block['distances','z'],x)
    x = interp_x(z0)

    dz = z0[1]-z0[0]
    dxdz = np.gradient(x,dz)
    #interp_dchi = spi.interp1d(z,Dchi)

    na = block['nz_%s'%sample_a, 'nbin']
    nb = block['nz_%s'%sample_b, 'nbin']
    zmin = 0.01

    nz_b = block['nz_%s'%sample_b, 'bin_%d'%(j+1)]
    zb = block['nz_%s'%sample_b, 'z']
    nz_a = block['nz_%s'%sample_a, 'bin_%d'%(i+1)]
    za = block['nz_%s'%sample_a, 'z']

    #import pdb ; pdb.set_trace()

    interp_nz = spi.interp1d(zb, nz_b, fill_value='extrapolate')
    nz_b = interp_nz(z0)
    interp_nz = spi.interp1d(za, nz_a, fill_value='extrapolate')
    nz_a = interp_nz(z0)


    X = nz_a * nz_b/x/x/dxdz
    X[0]=0
    interp_X = spi.interp1d(z0, X, fill_value='extrapolate')

    # Inner integral over redshift
    V,Verr = sint.quad(interp_X, zmin, z0.max())
    W = nz_a*nz_b/x/x/dxdz/V
    W[0]=0
    
    #np.savetxt("/home/jiomer/research/direct_ia_theory/nb.txt",nb,fmt="%e")
    #np.savetxt("/home/jiomer/research/direct_ia_theory/na.txt",na,fmt="%e")

    return z0,W


class Projected_Corr_RSD():
    def __init__(self,rp=None,pi=None,pi_max=100,l=[0,2,4],k=None, lowring=True):
        self.rp=rp
        self.pi=pi
        if rp is None:
            self.rp=np.logspace(-1,np.log10(200),60)
        if pi is None:
            self.pi=np.logspace(-3,np.log10(pi_max),125)
#            self.pi=np.append(0,self.pi)
        self.dpi=np.gradient(self.pi)
        self.piG,self.rpG=np.meshgrid(self.pi,self.rp)
        self.rG=np.sqrt(self.rpG**2+self.piG**2)
        self.muG=self.piG/self.rG
        self.L={}
        self.j={}
        for i in l:
            self.L[i]=legendre(i,self.muG)
            self.j[i]=P2xi(k,l=i, lowring=lowring)

        
    def alpha(self, l, beta1, beta2):
        if l==0:
            return 1 + 1./3.*(beta1+beta2) + 1./5*(beta1*beta2)
        elif l==2:
            return (2./3.*(beta1+beta2) + 4./7.*(beta1*beta2))
        elif l==4:
            return 8./35.*(beta1*beta2)

    def w_to_DS(self,rp=[],w=[]):
        DS0=2*w[0]*rp[0]**2
        return 2.*integrate.cumtrapz(w*rp,x=rp,initial=0)/rp**2-w+DS0/rp**2
    
    #def compute_rixii(self,pk,li):
    #    ri,xi_i = self.j[li](pk, extrap=False)
    #    xi_intp=interp1d(ri,xi_i,bounds_error=False,fill_value=0)
    #    xii = np.dot((xi_intp(self.rG)*self.L[li]),self.dpi)
    #    xii *= 2
    #    return ri,xii

    def get_xi_noext(self, pk=[], l=[0,2,4]):
        xi={}
        for i in l:
            #
            ri, xi_i = self.j[i](pk, extrap=False)
            xi_intp=interp1d(ri,xi_i,bounds_error=False,fill_value=0)
            xi[i] = np.dot((xi_intp(self.rG)*self.L[i]),self.dpi)
            xi[i]*=2#one sided pi
        return xi

    def get_xi(self, pk=[], l=[0,2,4]):
        xi={}
        for i in l:
            #
            ri, xi_i = self.j[i](pk, extrap=True)
            xi_intp=interp1d(ri,xi_i,bounds_error=False,fill_value=0)
            xi[i] = np.dot((xi_intp(self.rG)*self.L[i]),self.dpi)
            #if (self.nu==2):
            #    xi[i]*=(-1)**(i/2)
            xi[i]*=2#one sided pi
            #import pdb ; pdb.set_trace()
                
        return xi

    #def get_rixi(self, pk=[], l=[0,2,4]):
    #    xi={}
    #    ri = {}
    #    for i in range(3):
    #        ri[int(i*2)],xi[int(i*2)] = self.compute_rixii(pk,i*2)
    #    return ri,xi

    def xi_wgg(self,f=0,bg=0,bg2=None,pk=[],xi=[],l=[0,2,4],threshold=1e10):
        bg1=bg
        if bg2 is None:
            bg2=bg
        beta1=f/bg1
        beta2=f/bg2
        max_xi1 = np.max(abs(xi[0]))
        max_xi2 = np.max(abs(xi[2]))
        max_xi3 = np.max(abs(xi[4]))
        max_xi = max( [max_xi1,max_xi2,max_xi3] )
        if max_xi > threshold:
            xi = self.get_xi_noext(pk=pk,l=l)
        W = np.zeros_like(xi[[k for k in xi.keys()][0]])
        for i in l:
            W+=(xi[i].T*self.alpha(i,beta1,beta2)*bg1*bg2).T
        return W,xi

    def wgg_calc(self,f=0,bg=0,bg2=None,pk=[],xi=None,l=[0,2,4]):
        bg1=bg
        if bg2 is None:
            bg2=bg
        beta1=f/bg1
        beta2=f/bg2
        if xi is None:
            xi=self.get_xi(pk=pk,l=l)
#        import pdb ; pdb.set_trace()
        W = np.zeros_like(xi[[k for k in xi.keys()][0]])
       # import pdb ; pdb.set_trace()
        for i in l:
            W+=(xi[i].T*self.alpha(i,beta1,beta2)*bg1*bg2).T

        return W,xi

    def wgm_calc(self,f=0,bg=0,beta2=0,pk=[],xi=None,l=[0,2,4],do_DS=True):
        beta1=f/bg
        if xi is None:
            xi=self.get_xi(pk=pk,l=l)
        W=np.zeros_like(xi[[k for k in xi.keys()][0]])
        y=[]
        for i in l:
            W+=(xi[i].T*self.alpha(i,beta1,beta2)*bg).T

        #import pdb ; pdb.set_trace()
        if do_DS:
            W=self.w_to_DS(rp=self.rp,w=W)
        return W
