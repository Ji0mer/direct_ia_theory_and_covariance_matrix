import os
import numpy as np
from scipy.integrate import simpson,quad
from scipy.interpolate import interp1d
from scipy.special import jn
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations_with_replacement

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(MODULE_DIR, "..", ".."))
DEFAULT_AVG_JN_PATH = os.path.join(os.environ.get("IA_LIB", REPO_ROOT), "output", "avg_jn")
MAX_BESSEL_CHUNK_ELEMENTS = 2_000_000
MAX_BESSEL_WORKERS = 1

#############################################################################################################################################
#############################################################################################################################################

def ht(nv,k,fk,rout,kres = 5e-4):
    interp_fk = interp1d(k,fk)
    knew = np.arange( k[0],k[-1],kres )
    fknew = interp_fk( knew )
    kr = np.outer( rout,knew )
    j = jn(nv,kr)
    Fr = simpson(j*fknew*knew/(2*np.pi),x=knew)
    return Fr

def iht(nv,r,Fr,kout,rres=5e-4):
    interp_Fr = interp1d(r,Fr)
    rnew = np.arange( r[0],r[-1],rres )
    Frnew = interp_Fr( rnew )
    kr = np.outer( kout,rnew )
    j = jn(nv,kr)
    fk = simpson(j*Frnew*rnew*2*np.pi,x=rnew)
    return fk

def interp_func(x,y,xnew,axis=0,kind='linear'):
    interp_func = interp1d(x,y,axis=axis,kind=kind,fill_value="extrapolate")
    y_new = interp_func(xnew)
    return y_new

# Copied from CosmoSIS.
def compute_c1_baseline():
    C1_M_sun = 5e-14  # h^-2 M_S^-1 Mpc^3
    M_sun = 1.9891e30  # kg
    Mpc_in_m = 3.0857e22  # meters
    C1_SI = C1_M_sun / M_sun * (Mpc_in_m) ** 3  # h^-2 kg^-1 m^3
    # rho_crit_0 = 3 H^2 / 8 pi G
    G = 6.67384e-11  # m^3 kg^-1 s^-2
    H = 100  #  h km s^-1 Mpc^-1
    H_SI = H * 1000.0 / Mpc_in_m  # h s^-1
    rho_crit_0 = 3 * H_SI ** 2 / (8 * np.pi * G)  #  h^2 kg m^-3
    f = C1_SI * rho_crit_0
    return f

def compute_c1(A1,Dz,z_out,z_piv=0,alpha1=0,Omega_m=0.3):
    C1_RHOCRIT = compute_c1_baseline()
    return -1.0*A1*C1_RHOCRIT*Omega_m/Dz*( (1.0+z_out)/(1.0+z_piv) )**alpha1

#############################################################################################################################################
#############################################################################################################################################


#############################################################################################################################################
#############################################################################################################################################

class Compute_covmat():
    def __init__(self,rbins,rres,kuse,nv=[0,2,[0,4]],logspace=True,avg_jn=True,load_data=False,path=None,quad_limits=5000):
        self.rbins = rbins
        self.res = rres
        self.ktemp = kuse
        self.k = kuse
        self.nv = nv
        self.rp = {}
        self.j = {}
        self.avg_jn = avg_jn
        self.quad_limits = quad_limits
        if load_data == False:
            self.set_jn_data()
        elif load_data == True:
            if path == None:
                self.load_jn_data()
            else:
                self.load_jn_data(file_path = path)
        else:
            pass
    
    def save_jn_data(self,file_path="./data/avg_jn/simpson/"):
        os.makedirs(file_path, exist_ok=True)
        np.save(file_path+"k.npy",self.k)
        np.save(file_path+"rbins.npy",self.rbins)

        np.save(file_path+"rp_nv0.npy",self.rp[0])
        np.save(file_path+"rp_nv2.npy",self.rp[2])
        np.save(file_path+"rp_nv04.npy",self.rp["[0, 4]"])


        for i in range( len(self.j[0]) ):
            file_name = file_path+"j0_"+str(i)+".npy"
            np.save(file_name,self.j[0][i])

        for i in range( len(self.j[2]) ):
            file_name = file_path+"j2_"+str(i)+".npy"
            np.save(file_name,self.j[2][i])

        for i in range( len(self.j["[0, 4]"]) ):
            file_name = file_path+"j04_"+str(i)+".npy"
            np.save(file_name,self.j["[0, 4]"][i])
        print("Finished...")

    def _bin_r_samples(self, bin_index):
        if self.rbins[bin_index+1] < 1:
            return np.arange(self.rbins[bin_index], self.rbins[bin_index+1], self.res/5)
        return np.arange(self.rbins[bin_index], self.rbins[bin_index+1], self.res)

    def _compute_avg_jn_bin(self, bin_index, unique_orders, requests):
        ruse = self._bin_r_samples(bin_index)
        weight = 2*np.pi*ruse

        r_count = max(int(ruse.size), 1)
        chunk_size = max(
            1,
            min(
                self.k.size,
                int(MAX_BESSEL_CHUNK_ELEMENTS // r_count),
            ),
        )

        order_integrals = {
            order: np.zeros_like(self.k, dtype=float) for order in unique_orders
        }
        for start in range(0, self.k.size, chunk_size):
            stop = min(start + chunk_size, self.k.size)
            kr = np.outer(self.k[start:stop], ruse)
            for order in unique_orders:
                order_integrals[order][start:stop] = simpson(
                    weight * jn(order, kr),
                    x=ruse,
                )

        exact_area = np.pi*(np.max(ruse)**2 - np.min(ruse)**2)
        weighted_area = simpson(weight, x=ruse)

        results = {}
        for key, orders, mode in requests:
            numerator = np.zeros_like(self.k, dtype=float)
            for order in orders:
                numerator += order_integrals[order]
            denominator = exact_area if mode == "single" else weighted_area
            results[key] = numerator / denominator

        return bin_index, results
    
    def set_jn_data(self):
        
        print("Compute Bessel function parallel...")
    
        if self.avg_jn == True:
            requests = []
            unique_orders = []
            seen_orders = set()
            
            for i in self.nv:
                if isinstance(i, list):
                    print(i)
                    orders = tuple(i)
                    requests.append((str(i), orders, "sum"))
                    for order in orders:
                        if order not in seen_orders:
                            seen_orders.add(order)
                            unique_orders.append(order)
                elif isinstance(i, int):
                    print(i)
                    requests.append((i, (i,), "single"))
                    if i not in seen_orders:
                        seen_orders.add(i)
                        unique_orders.append(i)
                else:
                    print(i)
                    self.rp[i], self.j[i] = 0, 0
            
            if requests:
                rnew = (self.rbins[:-1]+self.rbins[1:])/2
                for key, _, _ in requests:
                    self.rp[key] = rnew
                    self.j[key] = {}

                request_tuple = tuple(requests)
                order_tuple = tuple(unique_orders)
                bin_indices = range(len(self.rbins)-1)
                max_workers = min(
                    len(self.rbins)-1,
                    os.cpu_count() or 1,
                    MAX_BESSEL_WORKERS,
                )
                try:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_bin = {
                            executor.submit(
                                self._compute_avg_jn_bin,
                                bin_index,
                                order_tuple,
                                request_tuple,
                            ): bin_index
                            for bin_index in bin_indices
                        }

                        for future in as_completed(future_to_bin):
                            try:
                                bin_index, result_map = future.result()
                                for key, values in result_map.items():
                                    self.j[key][bin_index] = values
                            except Exception as exc:
                                bin_index = future_to_bin[future]
                                raise RuntimeError(
                                    f"Failed to compute averaged Bessel cache for bin {bin_index}"
                                ) from exc
                except Exception as exc:
                    print(
                        "Threaded Bessel-cache build failed; falling back to serial. "
                        f"Reason: {exc}"
                    )
                    for bin_index in bin_indices:
                        _, result_map = self._compute_avg_jn_bin(
                            bin_index,
                            order_tuple,
                            request_tuple,
                        )
                        for key, values in result_map.items():
                            self.j[key][bin_index] = values
        
        else:
            # Keep the non-averaged Bessel-function computation unchanged (usually faster)
            for i in self.nv:
                if isinstance(i, int):
                    self.rp[i], self.j[i] = self.compute_jn(i)
                elif isinstance(i, list):
                    self.rp[str(i)], self.j[i] = self.compute_jns(i)
                else:
                    self.rp[i], self.j[i] = 0, 0
        
        return 0


    def load_jn_data(self, file_path=None, numbins=20):
        if file_path is None:
            file_path = DEFAULT_AVG_JN_PATH
        file_path = os.path.abspath(os.path.expanduser(os.path.expandvars(file_path)))
        if not os.path.isdir(file_path):
            raise FileNotFoundError(
                "Missing averaged Bessel cache directory: %s" % file_path
            )
        
        print("Only using saved k, rp, averaged jn....")
        
        self.k = np.load(os.path.join(file_path, "k.npy"))
        #self.k *= ( 1-np.heaviside(self.k-20,0) )
        
        self.rp[0] = np.load(os.path.join(file_path, "rp_nv0.npy"))
        self.rp[2] = np.load(os.path.join(file_path, "rp_nv2.npy"))
        self.rp["[0, 4]"] = np.load(os.path.join(file_path, "rp_nv04.npy"))

        j0 = {}
        j2 = {}
        j04 = {}
        for i in range(numbins):
            j0[i] = np.load(os.path.join(file_path, "j0_"+str(i)+".npy"))
            j2[i] = np.load(os.path.join(file_path, "j2_"+str(i)+".npy"))
            j04[i] = np.load(os.path.join(file_path, "j04_"+str(i)+".npy"))
        self.j[0] = j0
        self.j[2] = j2
        self.j["[0, 4]"] = j04
    
        return True
    
    def compute_jn(self,nvi):
        j = {}
        #rnew = np.sqrt( self.rbins[:-1]*self.rbins[1:] )
        rnew = ( self.rbins[:-1]+self.rbins[1:] )/2
        for ind1 in range( len(rnew) ):
            kr = self.k*rnew[ind1]
            j[ind1] = jn(nvi,kr)
        
        return rnew,j
    
    def compute_jns(self,nvi):
        j = {}
        #rnew = np.sqrt( self.rbins[:-1]*self.rbins[1:] )
        rnew = ( self.rbins[:-1]+self.rbins[1:] )/2
        for ind1 in range( len(rnew) ):
            kr = self.k*rnew[ind1]
            sum_jns = np.zeros_like(kr)
            for ind2 in nvi:
                sum_jns += jn(ind2,kr)
            j[ind1] = sum_jns
            
        return rnew,j
    
    def compute_avg_jn(self,nvi):
        avg_j = {}
        for i in range( len(self.rbins)-1 ):
            if self.rbins[i+1] < 1:
                ruse = np.arange( self.rbins[i],self.rbins[i+1],self.res/5 )
            else:
                ruse = np.arange( self.rbins[i],self.rbins[i+1],self.res )
            kr = np.outer( self.k,ruse )
            avg_jn = simpson( 2*np.pi*ruse*jn(nvi,kr),x=ruse )
            avg_jn /= np.pi*(np.max(ruse)**2 - np.min(ruse)**2)
            avg_j[i] = avg_jn

        rnew = ( self.rbins[:-1]+self.rbins[1:] )/2
        
        return rnew,avg_j
    
    def compute_avg_jns(self,nvi):
        avg_j = {}
        for i in range( len(self.rbins)-1 ):
            if self.rbins[i+1] < 1:
                ruse = np.arange( self.rbins[i],self.rbins[i+1],self.res/5 )
            else:
                ruse = np.arange( self.rbins[i],self.rbins[i+1],self.res )
            kr = np.outer( self.k,ruse )
            sum_jn = np.zeros_like( kr )
            for j in nvi:
                sum_jn += jn(j,kr)
            avg_jn = simpson( 2*np.pi*ruse*sum_jn,x=ruse )
            avg_jn /= simpson( 2*np.pi*ruse,x=ruse )
            avg_j[i] = avg_jn

        rnew = ( self.rbins[:-1]+self.rbins[1:] )/2
        return rnew,avg_j
    
    def covariance_wgpwgp(self,pgg,pii,pgi,Ng=0,Np=0):
        pgg = interp_func(self.ktemp,pgg,self.k)
        pii = interp_func(self.ktemp,pii,self.k)
        pgi = interp_func(self.ktemp,pgi,self.k)
        
        cov = np.zeros( (len(self.rbins)-1,len(self.rbins)-1) )
        for i in range( len(self.rbins)-1 ):
            for j in range( len(self.rbins)-1 ):
                if i == j:
                    cov[i,j] = simpson( self.k/(2*np.pi)*self.j[2][i]*self.j[2][j]*( (pgg+Ng)*(pii+Np)+pgi**2 ),x=self.k )
                else:
                    cov[i,j] = simpson( self.k/(2*np.pi)*self.j[2][i]*self.j[2][j]*( pgg*Np+pii*Ng+2*pgi**2 ),x=self.k )
        return cov
    
    def covariance_wgpwgp_component(self,pgg,pii,pgi,Ng=0,Np=0):
        pgg = interp_func(self.ktemp,pgg,self.k)
        pii = interp_func(self.ktemp,pii,self.k)
        pgi = interp_func(self.ktemp,pgi,self.k)
        
        covpgi2 = np.zeros( (len(self.rbins)-1,len(self.rbins)-1) )
        covc = np.zeros( (len(self.rbins)-1,len(self.rbins)-1) )
        covpggxc = np.zeros( (len(self.rbins)-1,len(self.rbins)-1) )
        covpiixc = np.zeros( (len(self.rbins)-1,len(self.rbins)-1) )
        for i in range( len(self.rbins)-1 ):
            for j in range( len(self.rbins)-1 ):
                if i == j:
                    covc[i,j] = simpson( self.k/(2*np.pi)*self.j[2][i]*self.j[2][j]*( Ng*Np ),x=self.k )
                covpgi2[i,j] = simpson( self.k/(2*np.pi)*self.j[2][i]*self.j[2][j]*( pgi**2 ),x=self.k )
                covpggxc[i,j] = simpson( self.k/(2*np.pi)*self.j[2][i]*self.j[2][j]*( pgg*Np ),x=self.k )
                covpiixc[i,j] = simpson( self.k/(2*np.pi)*self.j[2][i]*self.j[2][j]*( pii*Ng ),x=self.k )
        return covc,covpgi2,covpggxc,covpiixc
    
    def covariance_wgpwpp(self,pgg,pii,pgi,Ng=0,Np=0):
        pgg = interp_func(self.ktemp,pgg,self.k)
        pii = interp_func(self.ktemp,pii,self.k)
        pgi = interp_func(self.ktemp,pgi,self.k)
        
        cov = np.zeros( (len(self.rbins)-1,len(self.rbins)-1) )
        for i in range( len(self.rbins)-1 ):
            for j in range( len(self.rbins)-1 ):
                cov[i,j] = simpson( self.k/(2*np.pi)*self.j[2][i]*self.j["[0, 4]"][j]*( 2*pgi*(pii+Np) ),x=self.k )
        return cov
    
    def covariance_wgpwgg(self,pgg,pii,pgi,Ng=0,Np=0):
        pgg = interp_func(self.ktemp,pgg,self.k)
        pii = interp_func(self.ktemp,pii,self.k)
        pgi = interp_func(self.ktemp,pgi,self.k)
        
        cov = np.zeros( (len(self.rbins)-1,len(self.rbins)-1) )
        for i in range( len(self.rbins)-1 ):
            for j in range( len(self.rbins)-1 ):
                cov[i,j] = simpson( self.k/(2*np.pi)*self.j[2][i]*self.j[0][j]*( 2*pgi*(pgg+Ng) ),x=self.k )
        return cov
    
    def covariance_wggwgg(self,pgg,Ng=0):
        pgg = interp_func(self.ktemp,pgg,self.k)
        
        cov = np.zeros( (len(self.rbins)-1,len(self.rbins)-1) )
        for i in range( len(self.rbins)-1 ):
            for j in range( len(self.rbins)-1 ):
                if i == j:
                    cov[i,j] = simpson( self.k/(2*np.pi)*self.j[0][i]*self.j[0][j]*( 2*(pgg+Ng)*(pgg+Ng) ),x=self.k )
                else:
                    cov[i,j] = simpson( self.k/(2*np.pi)*self.j[0][i]*self.j[0][j]*( 2*(pgg**2+2*pgg*Ng) ),x=self.k )
        return cov
    
    def covariance_wggwgp(self,pgg,pii,pgi,Ng=0,Np=0):
        pgg = interp_func(self.ktemp,pgg,self.k)
        pii = interp_func(self.ktemp,pii,self.k)
        pgi = interp_func(self.ktemp,pgi,self.k)
        
        cov = np.zeros( (len(self.rbins)-1,len(self.rbins)-1) )
        for i in range( len(self.rbins)-1 ):
            for j in range( len(self.rbins)-1 ):
                cov[i,j] = simpson( self.k/(2*np.pi)*self.j[0][i]*self.j[2][j]*( 2*(pgg+Ng)*pgi ),x=self.k )
        return cov
    
    def covariance_wggwpp(self,pgg,pii,pgi,Ng=0,Np=0):
        pgg = interp_func(self.ktemp,pgg,self.k)
        pii = interp_func(self.ktemp,pii,self.k)
        pgi = interp_func(self.ktemp,pgi,self.k)
        
        cov = np.zeros( (len(self.rbins)-1,len(self.rbins)-1) )
        for i in range( len(self.rbins)-1 ):
            for j in range( len(self.rbins)-1 ):
                cov[i,j] = simpson( self.k/(2*np.pi)*self.j[0][i]*self.j["[0, 4]"][j]*( 2*pgi**2 ),x=self.k )
        return cov
    
    def covariance_wppwpp(self,pii,Np=0):
        pii = interp_func(self.ktemp,pii,self.k)
        
        cov = np.zeros( (len(self.rbins)-1,len(self.rbins)-1) )
        for i in range( len(self.rbins)-1 ):
            for j in range( len(self.rbins)-1 ):
                if i == j:
                    cov[i,j] = simpson( self.k/(2*np.pi)*self.j["[0, 4]"][i]*self.j["[0, 4]"][j]*( 2*(pii+Np)*(pii+Np) ),x=self.k )
                else:
                    cov[i,j] = simpson( self.k/(2*np.pi)*self.j["[0, 4]"][i]*self.j["[0, 4]"][j]*( 2*(pii**2+2*pii*Np) ),x=self.k )
        return cov
    
    def covariance_wppwgp(self,pgg,pii,pgi,Ng=0,Np=0):
        pgg = interp_func(self.ktemp,pgg,self.k)
        pii = interp_func(self.ktemp,pii,self.k)
        pgi = interp_func(self.ktemp,pgi,self.k)
        
        cov = np.zeros( (len(self.rbins)-1,len(self.rbins)-1) )
        for i in range( len(self.rbins)-1 ):
            for j in range( len(self.rbins)-1 ):
                cov[i,j] = simpson( self.k/(2*np.pi)*self.j["[0, 4]"][i]*self.j[2][j]*( 2*(pii+Np)*pgi ),x=self.k )
        return cov
    
    def covariance_wppwgg(self,pgg,pii,pgi,Ng=0,Np=0):
        pgg = interp_func(self.ktemp,pgg,self.k)
        pii = interp_func(self.ktemp,pii,self.k)
        pgi = interp_func(self.ktemp,pgi,self.k)
        
        cov = np.zeros( (len(self.rbins)-1,len(self.rbins)-1) )
        for i in range( len(self.rbins)-1 ):
            for j in range( len(self.rbins)-1 ):
                cov[i,j] = simpson( self.k/(2*np.pi)*self.j["[0, 4]"][i]*self.j[0][j]*( 2*pgi**2 ),x=self.k )
        return cov

    ############
    # Romain method
    ############
    
    def cov_romain_wgpwgp(self, pgg, pii, pgi, Ng=0, Np=0):
        pgg = interp_func(self.ktemp, pgg, self.k)
        pii = interp_func(self.ktemp, pii, self.k)
        pgi = interp_func(self.ktemp, pgi, self.k)
    
        cov = np.zeros((len(self.rbins) - 1, len(self.rbins) - 1))
        
        kmin = float(self.k[0])
        kmax = float(self.k[-1])
    
        for i in range(len(self.rbins) - 1):
            for j in range(len(self.rbins) - 1):
                if i == j:
                    y = (self.k / (2*np.pi)) * self.j[2][i] * self.j[2][j] * ( (pgg + Ng) * (pii + Np) + pgi**2 )
                    f = interp1d(self.k, y, kind='cubic', bounds_error=False, fill_value=0.0)
                else:
                    y = (self.k / (2*np.pi)) * self.j[2][i] * self.j[2][j] * ( (pgg*Np + Ng*pii ) + 2*pgi**2 )
                    f = interp1d(self.k, y, kind='cubic', bounds_error=False, fill_value=0.0)
                cov[i, j] = quad(f, kmin, kmax, limit=self.quad_limits)[0]
    
        return cov

    def cov_romain_wgpwgp_component(self, p_gg_linz, ppp_temp, pgp_temp, Ng, Np, n_workers=None):
        p_gg_linz = interp_func(self.ktemp,p_gg_linz,self.k)
        ppp_temp = interp_func(self.ktemp,ppp_temp,self.k)
        pgp_temp = interp_func(self.ktemp,pgp_temp,self.k)
        
        # ---- Basic sizes and preallocation ----
        nbins = len(self.rbins) - 1
        covc_romain     = np.zeros((nbins, nbins), dtype=float)
        covpgi2_romain  = np.zeros((nbins, nbins), dtype=float)
        covpggxc_romain = np.zeros((nbins, nbins), dtype=float)
        covpppxc_romain = np.zeros((nbins, nbins), dtype=float)
    
        # ---- One-time precomputation (independent of l) ----
        k    = self.k                   # (Nk,)
        j2   = self.j[2]                # dict: i -> j2_i(k), each with shape (Nk,)
        kmin = float(np.min(k))
        kmax = float(np.max(k))
    
        # To match the original implementation, each integrand for (i, j) has the form:
        #   y(k) = (k / 2π) * j2_i(k) * j2_j(k) * X(k)
        # Then perform the numerical integral over [kmin, kmax]
        # Keep the original quad + cubic interpolation approach here (minimal change)
        two_pi = 2.0 * np.pi
    
        # Convert the input power spectra to np.ndarray uniformly (in case a list is passed in)
        pgi = np.asarray(pgp_temp)   # P_gi(k)
        pgg = np.asarray(p_gg_linz)  # P_gg(k)
        pii = np.asarray(ppp_temp)   # P_ii(k)
    
        if n_workers is None or n_workers <= 0:
            n_workers = os.cpu_count() or 1
    
        # Upper-triangular index list (including the diagonal): compute once, then fill symmetrically
        pairs = list(combinations_with_replacement(range(nbins), 2))
    
        def _one_pair(i, j):
            # j2_i, j2_j: (Nk,)
            j2_i = np.asarray(j2[i])
            j2_j = np.asarray(j2[j])
    
            base = (k / two_pi) * j2_i * j2_j  # (Nk,)
    
            # The four integrands y1..y4
            y1 = base * (pgi ** 2)    # Corresponds to covpgi2
            y2 = base * (Ng * Np)     # Corresponds to covc
            y3 = base * (pgg * Np)    # Corresponds to covpggxc
            y4 = base * (Ng * pii)    # Corresponds to covpppxc
    
            # Interpolate as scalar functions for quad; set out-of-range values to 0
            f1 = interp1d(k, y1, kind='cubic', bounds_error=False, fill_value=0.0)
            f2 = interp1d(k, y2, kind='cubic', bounds_error=False, fill_value=0.0)
            f3 = interp1d(k, y3, kind='cubic', bounds_error=False, fill_value=0.0)
            f4 = interp1d(k, y4, kind='cubic', bounds_error=False, fill_value=0.0)
    
            # Perform the integration
            v_pgi2 = quad(f1, kmin, kmax, limit=self.quad_limits)[0]
            v_c    = quad(f2, kmin, kmax, limit=self.quad_limits)[0]
            v_pggx = quad(f3, kmin, kmax, limit=self.quad_limits)[0]
            v_pppx = quad(f4, kmin, kmax, limit=self.quad_limits)[0]
    
            return i, j, v_c, v_pgi2, v_pggx, v_pppx
    
        # Parallel scheduling
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_one_pair, i, j) for (i, j) in pairs]
            for fut in as_completed(futures):
                i, j, v_c, v_pgi2, v_pggx, v_pppx = fut.result()
    
                # Upper-triangular position
                covc_romain[i, j]     = v_c
                covpgi2_romain[i, j]  = v_pgi2
                covpggxc_romain[i, j] = v_pggx
                covpppxc_romain[i, j] = v_pppx
    
                # Fill the symmetric entry as well (if this is not a diagonal element)
                if i != j:
                    covc_romain[j, i]     = v_c
                    covpgi2_romain[j, i]  = v_pgi2
                    covpggxc_romain[j, i] = v_pggx
                    covpppxc_romain[j, i] = v_pppx
    
        return covc_romain, covpgi2_romain, covpggxc_romain, covpppxc_romain
    
    def cov_romain_wgpwpp(self, pgg, pii, pgi, Ng=0, Np=0):
        pgg = interp_func(self.ktemp, pgg, self.k)
        pii = interp_func(self.ktemp, pii, self.k)
        pgi = interp_func(self.ktemp, pgi, self.k)
    
        cov = np.zeros((len(self.rbins) - 1, len(self.rbins) - 1))
        
        kmin = float(self.k[0])
        kmax = float(self.k[-1])
        
        for i in range(len(self.rbins) - 1):
            for j in range(len(self.rbins) - 1):
            
                y = (self.k / (2*np.pi)) * self.j[2][i] * self.j["[0, 4]"][j] * ( 2*pgi*(pii+Np) )
                f = interp1d(self.k, y, kind='cubic', bounds_error=False, fill_value=0.0)
                cov[i, j] = quad(f, kmin, kmax, limit=self.quad_limits)[0]
    
        return cov

    def cov_romain_wgpwgg(self, pgg, pii, pgi, Ng=0, Np=0):
        pgg = interp_func(self.ktemp, pgg, self.k)
        pii = interp_func(self.ktemp, pii, self.k)
        pgi = interp_func(self.ktemp, pgi, self.k)
    
        cov = np.zeros((len(self.rbins) - 1, len(self.rbins) - 1))
        
        kmin = float(self.k[0])
        kmax = float(self.k[-1])
    
        for i in range(len(self.rbins) - 1):
            for j in range(len(self.rbins) - 1):
                
                y = (self.k / (2*np.pi)) * self.j[2][i] * self.j[0][j] * ( 2*pgi*(pgg+Ng) )
                f = interp1d(self.k, y, kind='cubic', bounds_error=False, fill_value=0.0)
                cov[i, j] = quad(f, kmin, kmax, limit=self.quad_limits)[0]
    
        return cov

    def cov_romain_wppwgp(self, pgg, pii, pgi, Ng=0, Np=0):
        pgg = interp_func(self.ktemp, pgg, self.k)
        pii = interp_func(self.ktemp, pii, self.k)
        pgi = interp_func(self.ktemp, pgi, self.k)
    
        cov = np.zeros((len(self.rbins) - 1, len(self.rbins) - 1))
        
        kmin = float(self.k[0])
        kmax = float(self.k[-1])
    
        for i in range(len(self.rbins) - 1):
            for j in range(len(self.rbins) - 1):
                
                y = (self.k / (2*np.pi)) * self.j["[0, 4]"][i] * self.j[2][j] * ( 2*(pii+Np)*pgi )
                f = interp1d(self.k, y, kind='cubic', bounds_error=False, fill_value=0.0)
                cov[i, j] = quad(f, kmin, kmax, limit=self.quad_limits)[0]
    
        return cov
        
    def cov_romain_wppwpp(self, pii, Np=0):
        pii = interp_func(self.ktemp, pii, self.k)
    
        cov = np.zeros((len(self.rbins) - 1, len(self.rbins) - 1))
        
        kmin = float(self.k[0])
        kmax = float(self.k[-1])
    
        for i in range(len(self.rbins) - 1):
            for j in range(len(self.rbins) - 1):
                
                if i == j:
                    y = (self.k / (2*np.pi)) * self.j["[0, 4]"][i] * self.j["[0, 4]"][j] * ( 2*(pii+Np)*(pii+Np) )
                    f = interp1d(self.k, y, kind='cubic', bounds_error=False, fill_value=0.0)
                else:
                    y = (self.k / (2*np.pi)) * self.j["[0, 4]"][i] * self.j["[0, 4]"][j] * ( 2*(pii**2+2*pii*Np) )
                    f = interp1d(self.k, y, kind='cubic', bounds_error=False, fill_value=0.0)
                cov[i, j] = quad(f, kmin, kmax, limit=self.quad_limits)[0]
    
        return cov

    def cov_romain_wppwgg(self, pgg, pii, pgi, Ng=0, Np=0):
        pgg = interp_func(self.ktemp, pgg, self.k)
        pii = interp_func(self.ktemp, pii, self.k)
        pgi = interp_func(self.ktemp, pgi, self.k)
    
        cov = np.zeros((len(self.rbins) - 1, len(self.rbins) - 1))
        
        kmin = float(self.k[0])
        kmax = float(self.k[-1])
    
        for i in range(len(self.rbins) - 1):
            for j in range(len(self.rbins) - 1):
                
                y = (self.k / (2*np.pi)) * self.j["[0, 4]"][i] * self.j[0][j] * ( 2*pgi**2 )
                f = interp1d(self.k, y, kind='cubic', bounds_error=False, fill_value=0.0)
                cov[i, j] = quad(f, kmin, kmax, limit=self.quad_limits)[0]
    
        return cov

    def cov_romain_wggwgp(self, pgg, pii, pgi, Ng=0, Np=0):
        pgg = interp_func(self.ktemp, pgg, self.k)
        pii = interp_func(self.ktemp, pii, self.k)
        pgi = interp_func(self.ktemp, pgi, self.k)
    
        cov = np.zeros((len(self.rbins) - 1, len(self.rbins) - 1))
        
        kmin = float(self.k[0])
        kmax = float(self.k[-1])
    
        for i in range(len(self.rbins) - 1):
            for j in range(len(self.rbins) - 1):
                
                y = (self.k / (2*np.pi)) * self.j[0][i] * self.j[2][j] * ( 2*(pgg+Ng)*pgi )
                f = interp1d(self.k, y, kind='cubic', bounds_error=False, fill_value=0.0)
                cov[i, j] = quad(f, kmin, kmax, limit=self.quad_limits)[0]
    
        return cov

    def cov_romain_wggwpp(self, pgg, pii, pgi, Ng=0, Np=0):
        pgg = interp_func(self.ktemp, pgg, self.k)
        pii = interp_func(self.ktemp, pii, self.k)
        pgi = interp_func(self.ktemp, pgi, self.k)
    
        cov = np.zeros((len(self.rbins) - 1, len(self.rbins) - 1))
        
        kmin = float(self.k[0])
        kmax = float(self.k[-1])
    
        for i in range(len(self.rbins) - 1):
            for j in range(len(self.rbins) - 1):
                
                y = (self.k / (2*np.pi)) * self.j[0][i] * self.j["[0, 4]"][j] * ( 2*pgi**2 )
                f = interp1d(self.k, y, kind='cubic', bounds_error=False, fill_value=0.0)
                cov[i, j] = quad(f, kmin, kmax, limit=self.quad_limits)[0]
    
        return cov

    def cov_romain_wggwgg(self, pgg, Ng=0):
        pgg = interp_func(self.ktemp, pgg, self.k)
    
        cov = np.zeros((len(self.rbins) - 1, len(self.rbins) - 1))
        
        kmin = float(self.k[0])
        kmax = float(self.k[-1])
    
        for i in range(len(self.rbins) - 1):
            for j in range(len(self.rbins) - 1):

                if i == j:
                    y = (self.k / (2*np.pi)) * self.j[0][i] * self.j[0][j] * ( 2*(pgg+Ng)*(pgg+Ng) )
                    f = interp1d(self.k, y, kind='cubic', bounds_error=False, fill_value=0.0)
                else:
                    y = (self.k / (2*np.pi)) * self.j[0][i] * self.j[0][j] * ( 2*(2*pgg*Ng+pgg**2) )
                    f = interp1d(self.k, y, kind='cubic', bounds_error=False, fill_value=0.0)
                cov[i, j] = quad(f, kmin, kmax, limit=self.quad_limits)[0]
    
        return cov
