import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy import constants as c, units as u
from astropy import cosmology
from astropy.wcs import WCS
from spectral_cube import SpectralCube
from matplotlib.patches import Rectangle
import seaborn as sns
from astropy.io import ascii
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
from collections import OrderedDict

from sympy import ode_order

sns.set_style("ticks", {"xtick.direction": u"in", "ytick.direction": u"in"})
fontsize = 18
plt.rc("font", **{"family": "sans-serif", "serif": ["Arial"]})
# plt.rc('font', size=fontsize)          # controls default text sizes
plt.rc('axes', titlesize=fontsize)     # fontsize of the axes title
plt.rc('axes', labelsize=fontsize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
plt.rc('legend', fontsize=fontsize)    # legend fontsize
plt.rc('figure', titlesize=fontsize)  # fontsize of the figure title

files = ("J1059_tworegions_norm_old.dat","J1059_tworegions_norm.dat")
# title = ('Blue region','Red region')
ap0color = "red"
ap1color = "blue"
col = (ap0color,ap1color)

rlines = OrderedDict(eval((open('rlines.txt','r')).read()))
glines = OrderedDict(eval((open('glines.txt','r')).read()))
tlines = OrderedDict(eval((open('tlines.txt','r')).read()))
newlines = OrderedDict()
for i in ascii.read('J1059_newlinelist.txt'):
    newlines["%s"%i[1]] = i[0]
norm_data = ascii.read(files[1])
nwavelength = norm_data["restwave"] * u.angstrom
bnorm = norm_data["bluenorm"] * u.erg / u.s / u.cm ** 2 / u.angstrom
bnorm_err = norm_data["blueerr"] * u.erg / u.s / u.cm ** 2 / u.angstrom
rnorm = norm_data["rednorm"] * u.erg / u.s / u.cm ** 2 / u.angstrom
rnorm_err = norm_data["rederr"] * u.erg / u.s / u.cm ** 2 / u.angstrom
errp = np.sqrt(rnorm_err**2+bnorm_err**2)

# --------------------------------------------------------------------

abslines = OrderedDict(
    [
        
        ("SiIIa", 1260.4221),
        ("OI", 1302.1685),
        ("SiIIb", 1304.3702),
        ("CII", 1334.5323),
        ('SiIV_1393', 1393.7550),
        ('SiIV_1402', 1402.77)])

low_ion = OrderedDict(
    [
        ("SiIIa", 1260.4221),
        ("OI", 1302.1685),
        ("SiIIb", 1304.3702),
        ("CII", 1334.5323)])

high_ion = OrderedDict(
    [
        ('SiIV_1393', 1393.7550),
        ('SiIV_1402', 1402.77)])

bluelines = OrderedDict(
    [
        ("CIII", 977),
        ("OI", 989),
        ("CII", 1036),
        ("SIVFeII", 1063),
        ("FeII", 1122),
        ("FeIIb", 1143.226),
        ("CIIIb", 1176),
        ("SiIIa", 1190),
        ("SiIIb", 1193),
        ("SiIII", 1207),])

bluelines1 = OrderedDict([
    ('NI_1200.2', 1200.2233),
    ('NI_1134_9', 1134.9803),
    ('FeII_1142', 1142.3656),
    ('FeII_1143', 1143.226),
    ('FeII_1144', 1144.9379),])

lymanlinelist = [1215.7, 1025.7, 972.54, 949.74, 937.80]
foreground_lymanlines = np.array(lymanlinelist) * 3.3865 / 3.79

# --------------------------------------------------------------------
#                       Velocity profiles
# --------------------------------------------------------------------

def velo(absline):
    vel = (norm_data["restwave"] * u.angstrom - absline[1] * u.AA)/(absline[1] * u.AA)* c.c.to(u.km/u.s)
    return vel

def vel_prof(spec1,spec2,absline):
        N = velo(absline)
        #         velocity profile
        # --------------------------------------------------------------------        
        fig = plt.figure(figsize=(20,10),dpi=90)
        ax1 = plt.subplot2grid((5,2), (0,0), rowspan=2)
        ax1.step(N,spec1,color='red',label = 'red region')  #spec1.quantity or spec2.quantity
        ax1.step(N,spec2,color='blue',label = 'blue region')
        ax1.axhline(1,ls = '--',color = 'grey')
        ax1.set_ylim(-0.1,1.4)
        ax1.set_xlim(-1500,1000)
        ax1.set_ylabel('normalized flux')
        ax1.text(-1400,1.1e-17,'%s_%s'%(absline[0],absline[1]))
        ax1.legend(loc="lower right")
        #         difference b/w rnorm and bnorm w/ error
        #--------------------------------------------------------------------
        ax2 = plt.subplot2grid((5,2), (2,0))
        ax2.step(N, rnorm-bnorm,alpha=1)
        ax2.axhline(y=0,ls='--')
        ax2.fill_between(N, rnorm-bnorm-errp, rnorm-bnorm+errp, step='mid', color=ap1color, alpha =0.2)
        ax2.set_ylim(-0.65,0.65)
        ax2.set_xlim(-1500,1000)
        ax1.get_shared_x_axes().join(ax1, ax2)
        ax1.set_xticklabels([])
        ax2.set_xlabel('relative velocity (km/s)')
        ax2.set_ylabel('difference')
    
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.savefig(f'velprof_{absline[1]:.4f}.png')
        plt.show()

# --------------------------------------------------------------------
#                   Average velocity profiles
# --------------------------------------------------------------------

xvals = np.linspace(-1500,1500,106) * u.km/u.s
def avg_velpro(dict,norm_data): # dict = low_ion / high_ion / absline etc.
                                # norm_data = (rnorm,)/(bnorm,)/(rnorm,bnorm)

    fig, ax = plt.subplots(len(norm_data),1,figsize=(10,len(norm_data)*5),squeeze=False,dpi=90)
    for i in range(len(norm_data)):
        tot = np.zeros(106) * norm_data[0].unit
        for lines in dict.items():
            norm_inter = np.empty(106) * norm_data[i].unit
            velocity = (nwavelength.value - lines[1])/(lines[1]) * c.c.to(u.km/u.s)
            norm_inter = np.interp(xvals, velocity, norm_data[i])
            tot += norm_inter
            avg = tot/len(dict)
            ax[i][0].step(velo(lines),norm_data[i],lw = 0.5,color='gray',label = 'region')
        
        ax[i][0].axhline(1,ls = '--',color = 'k')
        ax[i][0].set_ylim(-0.1,1.39)
        ax[i][0].set_xlim(-1500,1000)
        ax[i][0].set_xlabel('relative velocity (km/s)')
        ax[i][0].step(xvals,avg,
                    color=col[i] # remove this while plotting more than 2 profiles
                    )
        ax[i][0].set_ylabel('normalized flux (%s region)'%col[i])
        

        # Clean up the labels
        ax[i][0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=i == len(norm_data) - 1)
    

    def namestr(obj, namespace):
        return [name for name in namespace if namespace[name] is obj]
    
    plt.subplots_adjust(hspace=0)
    plt.suptitle('%sization'%(namestr(dict,globals())[0]),y=0.9)
    plt.savefig('avg_profs_%s.png'%(namestr(dict,globals())[0]))
    plt.show()
    
# --------------------------------------------------------------------
#                       Measure_absline
# --------------------------------------------------------------------

def measure_absline(wave, flux, err, line, vmin, vmax):
        """
        ew=measure_ew(wave, flux, line)
        returns the equivalent width (in the same units as "wave"),
        equivalent width-weighted velocity, maximum blueshifted velocity
        for the absorption line closest in wavelength to "line"
        """
        # no units
        wave = wave.value
        flux = flux.value
        disp = np.mean(np.diff(wave))
        err = err.value

        vmin = vmin * u.km / u.s
        vmax = vmax * u.km / u.s
        
        down = (vmin/c.c.to(u.km/u.s) + 1) * line * u.AA
        up = (vmax/c.c.to(u.km/u.s) + 1) * line * u.AA
        downer = np.where(wave > down.value)[0][0]
        upper = np.where(wave < up.value)[0][-1]
        
        conterr = np.median(err[(wave > 1250) & (wave < 1430)])
        ew = disp * np.sum((1 + np.random.normal(0, conterr, 1)[0]) - flux[downer:upper])
        cent = np.sum(disp * (1 - flux[downer:upper]) * wave[downer:upper]) / ew
        vel = (cent - line) / line * c.c.to(u.km / u.s)
        maxvel = (wave[downer] - line) / line * c.c.to(u.km / u.s)

        ew = ew * u.angstrom
        return ew, vel, maxvel, wave[downer], wave[upper]

# --------------------------------------------------------------------
#                       Equivalent width
# --------------------------------------------------------------------
class EquivalentWidth:

    def __init__(self, wave, flux, absline, vmin, vmax):
        self.wave = wave
        self.flux = flux
        self.err = self.get_error_data()
        self.absline = absline
        self.vmin = vmin
        self.vmax = vmax
        self.ew = None
        self.vel = None
        self.maxvel = None
        self.minwave = None
        self.maxwave = None

    def get_error_data(self):
        err_name = self.namestr(self.flux, globals())[0] + '_err'
        err = eval(err_name)
        return err
    
    @staticmethod
    def namestr(obj, namespace):
        return [name for name in namespace if namespace[name] is obj]
        
    def measure_lines(self):
        self.ew = np.zeros(len(self.absline)) * u.angstrom
        self.vel = np.zeros(len(self.absline)) * u.km / u.s
        self.maxvel = np.zeros(len(self.absline)) * u.km / u.s
        self.minwave = np.zeros(len(self.absline))
        self.maxwave = np.zeros(len(self.absline))
        lines_to_measure = list(self.absline.values())

        for j, linewave in enumerate(lines_to_measure):
            self.ew[j], self.vel[j], self.maxvel[j], self.minwave[j], self.maxwave[j] = measure_absline(
                self.wave, self.flux, self.err, linewave, self.vmin, self.vmax
            )

    def plot(self, color=None, **kwargs):
        self.measure_lines()
        if self.ew is None or self.minwave is None:
            raise ValueError("cll measure_lines before plotting.")
        
        fig, ax = plt.subplots(figsize=(30, 7.5), dpi=90)
        
        ax.plot(self.wave, self.flux, drawstyle="steps-mid", color=color, label=f'{color} Region')
        
        # filling in the regions
        for j in range(len(self.minwave)):
            ax.fill_between(
                [self.minwave[j], self.maxwave[j]],
                [-0.1, -0.1],
                y2=[1, 1],
                color=sns.xkcd_rgb["pink"],
                alpha=0.35,
                zorder=1,
            )
        
        # plotting the lines
        ax.axhline(y=1, color='gray', ls='--')
        if 'xlim' in kwargs:
            ax.set_xlim(kwargs['xlim'])
        else:
            ax.set_xlim((1250, 1425))
        ax.set_ylim(-0.1, 1.5)
        ax.legend(loc='lower right')
        
        lines_to_measure = list(self.absline.values())
        for line in lines_to_measure:
            ax.axvline(line, color="xkcd:dark grey", linestyle="--", zorder=2)
        for blueline in list(bluelines.values()):
            ax.axvline(blueline, color="xkcd:deep sky blue", linestyle="--", zorder=2)
        for lymanline in lymanlinelist:
            ax.axvline(lymanline, color="xkcd:faded red", linestyle="--", zorder=2)
        for intervening_line in foreground_lymanlines:
            ax.axvline(intervening_line, color="xkcd:medium green", linestyle="--", zorder=2)
        
        
        # plotting the lines and text within xlim
        xlim = ax.get_xlim()
        for k in list(self.absline.values()):
            if xlim[0] <= k <= xlim[1]:
                ax.text(k-1, 1.3, r'%0.2f$\AA$' % -self.ew[list(self.absline.values()).index(k)].value, 
                        {'ha': 'center'}, rotation=90)
                
        ax.vlines(x=[1247.38, 1323.93, 1324.31, 1343.35, 1417.24, 1501.76, 1718.55] * u.AA, ymin=1.3, ymax=1.5, colors='hotpink', ls='-', lw=5)

        # plt.title('Red region vs Blue region')
        plt.xlabel("Rest wavelength (Å)", fontsize=fontsize)
        plt.ylabel("F$_{\\lambda}$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)", fontsize=fontsize)
        plt.show()

    def compute_uncertainties(self, n_iter=500, save=False, **kwargs):
        ew1 = np.zeros((n_iter, len(self.absline))) * u.angstrom
        
        for n in range(n_iter):
            flux_new = self.flux + np.random.normal(0, self.err.value, len(self.err)) * self.err.unit
            self.measure_lines()
            ew1[n] = self.ew

        del_ew = np.std(ew1, axis=0) * u.angstrom
        m_ew = np.mean(ew1, axis=0) * u.angstrom

        print("-----------------------------------------------------")
        for ii in range(len(self.absline)):
            print('EW for', (list(self.absline.items())[ii]), 'is', f"{m_ew.value[ii]:0.3f}", '+/-', f"{del_ew[ii]:0.3f}")
        print("-----------------------------------------------------")

        # Saving to a text file
        if save:
            data_to_write = []
            keys_list = list(self.absline.keys())
            for key, value in self.absline.items():
                row = [key, value, "{:.3f}".format(m_ew[keys_list.index(key)].value), "{:.3f}".format(del_ew[keys_list.index(key)].value)]
                data_to_write.append(row)

            if 'file_path' in kwargs:
                file_path = kwargs['file_path']
            else:
                file_path = "ew_data.txt"
            with open(file_path, 'w') as file:
                file.write("Line\t\tWavelength\t\tm_ew\t\tDel_ew\n")
                for row in data_to_write:
                    file.write("{}\t\t{}\t\t{}\t\t{}\n".format(*row))
            return file_path
        
    def calculate(self, save=False, **kwargs):
        self.measure_lines()
        self.compute_uncertainties(save=save, **kwargs)
        
    def calculate_and_plot(self, save=False, color=None, **kwargs):
        self.plot(color=color, **kwargs) # This includes self.measure_lines()
        self.compute_uncertainties(save=save, **kwargs)
# --------------------------------------------------------------------
#                       Column Density
# --------------------------------------------------------------------
class ColumnDensity:
    def __init__(self, absline, f, norm_data, vmax):#specify absorption line| f-value| data| region(blue/red)| color
    #f-values from Donald C. Morton 10.1086/377639
        self.absline = absline
        self.f = f
        self.norm_data = norm_data
        self.vmax = vmax
        self.err, self.conterr = self.get_error_data()
        self.vel = None
        self.N_a = None
        self.Ntot = None
        self.log_Ntot = None

    def get_error_data(self):
        
        err_name = self.namestr(self.norm_data, globals())[0] + '_err'
        err = eval(err_name)
        conterr = np.median(err[(nwavelength.value > 1250) & (nwavelength.value < 1430)])
        return err, conterr
    
    @staticmethod
    def namestr(obj, namespace):
        return [name for name in namespace if namespace[name] is obj]

    def calculate_column_density(self):
        
        od = np.log(1 / self.norm_data.value)
        self.N_a = 3.768e14 * od / (self.absline[1] * self.f) / u.cm / u.cm / u.km * u.s
        self.vel = (nwavelength.value - self.absline[1]) / self.absline[1] * c.c.to(u.km / u.s)

        maxvel = self.vmax * self.vel.unit
        minvel = -1000 * self.vel.unit

        down = np.abs(self.vel - minvel)
        up = np.abs(self.vel - maxvel)
        downer = int(np.where(down == down.min())[0])
        upper = int(np.where(up == up.min())[0])
        self.Ntot = np.trapz(self.N_a[downer:upper], self.vel[downer:upper])
        self.log_Ntot = np.log10(self.Ntot.value)
        return self.N_a, self.vel, self.Ntot, self.log_Ntot
    
    def error(self, n_iter=500):
        Ntot_arr = np.zeros(n_iter)
        log_Ntot_arr = np.zeros(n_iter)
        og_norm_data = self.norm_data
        for n in range(n_iter):
            self.norm_data = og_norm_data + np.random.normal(0, self.conterr.value, 1)[0] * self.err.unit
            _, _, Ntot, log_Ntot = self.calculate_column_density()  # Assuming calculate_column_density is modified to take norm_data as an argument
            Ntot_arr[n] = Ntot.value
            log_Ntot_arr[n] = log_Ntot
        self.norm_data = og_norm_data
        del_log_Ntot = np.nanstd(log_Ntot_arr)
        return np.nanmean(log_Ntot_arr), del_log_Ntot
    
    def plot(self, ax, region, color=None):
        # plt.figure(figsize=(10, 6))

        if color == 1:
            ax.step(self.vel, np.log10(self.N_a.value) * self.N_a.unit, lw=2,
                     color=f"{region}",
                     label=f"{self.absline[0]} {region} region")
        else:
            ax.step(self.vel, np.log10(self.N_a.value) * self.N_a.unit, lw=2, label=f"{self.absline[0]} {region} region")
            # plt.title(f"{region} region")

        ax.set_xlabel('relative velocity (km/s)')
        ax.set_ylabel(r'$log N_a(v)[cm^{-2}/(km s^{-1}]$')
        ax.set_xlim(-1000, self.vmax)
        ax.set_ylim(9,)
        yticks = ax.get_yticks()
        ax.set_yticks(yticks[1:])
        ax.legend(loc='lower center', framealpha=1, ncol=2)
    
    def result(self, ax, region, color=None):
        self.calculate_column_density()
        mean_log_Ntot, error_log_Ntot = self.error()

        # fig, ax = plt.subplots(figsize=(10, 6))
        self.plot(ax, region, color)

        # print(f"Column Density: {self.Ntot}")
        # print(f"Log Column Density: {self.log_Ntot:.3f}")
        
        print(f"Log Column Density for {self.absline} ({region}): {mean_log_Ntot:.3f} \pm {error_log_Ntot:.3f}")

# --------------------------------------------------------------------
#                       Optical Depth Comparison
# --------------------------------------------------------------------
xvals = np.linspace(-1500,1500,106) * u.km/u.s

'''
we don't want to lose any data, and 106 points is around the upper limit 
in the (-1500,1500) range so it is easier to map data for all ionization 
lines to the same x-axis.
'''
def abslines_od(absline,norm_data,mask=None): # mask tuple of (min,max)
    od = np.log(1/norm_data.value)
    yinterp = np.interp(xvals, velo(absline), od)
    if mask==None:
        return yinterp
    else:
        window = (xvals.value<mask[0]) | (xvals.value>mask[1])
        masked_arr = np.where(window, yinterp, np.nan)
        
        return masked_arr

