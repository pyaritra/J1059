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
# from globalfunc import *

sns.set_style("ticks", {"xtick.direction": u"in", "ytick.direction": u"in"})
fontsize = 12
plt.rc("font", **{"family": "sans-serif", "serif": ["Arial"]})
plt.rc("xtick", labelsize=fontsize)
plt.rc("ytick", labelsize=fontsize)

files = ("J1059_tworegions_norm_old.dat","J1059_tworegions_norm.dat")
title = ('Blue region','Red region')
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
        ("FeIIb", 1143),
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
        plt.xlabel('relative velocity (km/s)')
        plt.subplots_adjust(hspace=0)
        plt.savefig('velprof_%f.png'%absline[1])
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
        

        # make the labels look cleaner
        plt.setp(ax[i][0].get_xticklabels()[0], visible=False)    
        plt.setp(ax[i][0].get_xticklabels()[-1], visible=False)
        plt.subplots_adjust(hspace=0)

    def namestr(obj, namespace):
        return [name for name in namespace if namespace[name] is obj]
    plt.suptitle('%sization'%(namestr(dict,globals())[0]),y=0.9)
        
    plt.show()
    # plt.savefig('avg_prof.png')


# --------------------------------------------------------------------
#                       Equivalent width
# --------------------------------------------------------------------
def measure_absline(wave, flux, err, line):
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

    vmin = -1000 * u.km / u.s
    vmax = 500 * u.km / u.s
    
    down = (vmin/c.c.to(u.km/u.s) + 1) * line * u.AA
    up = (vmax/c.c.to(u.km/u.s) + 1) * line * u.AA
    downer = np.where(wave > down.value)[0][0]
    upper = np.where(wave < up.value)[0][-1]
    #-----------------------------------------------------------------------------
    # alternative to adding another arg for the err spectrum
    # def namestr(obj, namespace):
    #     return [name for name in namespace if namespace[name] is obj]
    # l=namestr(flux,globals())
    # tr = l[0]+'_err'
    # err = eval(tr)
    
    conterr = np.median(err[(wave > 1250) & (wave < 1430)])
    ew = disp * np.sum((1 + np.random.normal(0, conterr, 1)[0]) - flux[downer:upper])
    cent = np.sum(disp * (1 - flux[downer:upper]) * wave[downer:upper]) / ew
    vel = (cent - line) / line * c.c.to(u.km / u.s)
    maxvel = (wave[downer] - line) / line * c.c.to(u.km / u.s)

    ew = ew * u.angstrom
    return ew, vel, maxvel, wave[downer], wave[upper]
    
# --------------------------------------------------------------------
#                       EW plots + uncertainties
# --------------------------------------------------------------------

def ew_calc(wave, flux, err, absline):

    fig, ax = plt.subplots(len(flux),1, figsize=(30, len(flux)*7.5),squeeze=False,dpi=90)

    for i in range(len(flux)):
        lines_to_measure = list(absline.values())
        #     lines_to_measure = list(bluelines.values())

        # initializing the arrays
        ew = np.zeros(len(lines_to_measure)) * u.angstrom
        minwave = np.zeros(len(lines_to_measure))
        maxwave = np.zeros(len(lines_to_measure))
        vel = np.zeros(len(lines_to_measure)) * u.km / u.s
        maxvel = np.zeros(len(lines_to_measure)) * u.km / u.s
        for j in range(len(lines_to_measure)):
            linewave = lines_to_measure[j]
            ew[j], vel[j], maxvel[j], minwave[j], maxwave[j] = measure_absline(
                wave, flux[i], err[i], linewave    
            )

        minwave = minwave * u.AA
        maxwave = maxwave * u.AA

        ax[i][0].plot(wave, flux[i], color=col[i], drawstyle="steps-mid", label = '%s region'%col[i])
        
        # filling in the regions
        for j in range(len(minwave)):
            ax[i][0].fill_between(
                [minwave[j].value, maxwave[j].value],
                [-0.1, -0.1],
                y2=[1, 1],
                color=sns.xkcd_rgb["pink"],
                alpha=0.35,
                zorder=1,
            )
        # plotting the lines
        #-----------------------------------------------------------------------------
        ax[i][0].axhline(y=1, color='gray',ls = '--')
        ax[i][0].set_xlim(1250,1425)
        ax[i][0].set_ylim(-0.1,1.5)
        ax[i][0].legend(loc='lower right')

        for line in lines_to_measure:
            ax[i][0].axvline(line, color="xkcd:dark grey", linestyle="--", zorder=2)
        for blueline in list(bluelines.values()):
            ax[i][0].axvline(blueline, color="xkcd:deep sky blue", linestyle="--", zorder=2)
        for lymanline in lymanlinelist:
            ax[i][0].axvline(lymanline, color="xkcd:faded red", linestyle="--", zorder=2)
        for intervening_line in foreground_lymanlines:
            ax[i][0].axvline(intervening_line, color="xkcd:medium green", linestyle="--", zorder=2)
        
        
        for k in lines_to_measure:
            ax[i][0].text(k-1, 1.3, r'%0.2f$\AA$'%-ew[lines_to_measure.index(k)].value, {'ha': 'center'}, rotation=90)

        ax[i][0].vlines(x=[1247.38,1323.93,1324.31,1343.35,1417.24,1501.76,1718.55] * u.AA, ymin=1.3, ymax=1.5, colors='hotpink', ls='-', lw=5)
        #-----------------------------------------------------------------------------
        # calculate the uncertainties in the EW values
        #-----------------------------------------------------------------------------
        n_iter = 500 # number of iterations
        ew = np.zeros(len(lines_to_measure)) * u.angstrom
        minwave = np.zeros(len(lines_to_measure))
        maxwave = np.zeros(len(lines_to_measure))
        vel = np.zeros(len(lines_to_measure)) * u.km / u.s
        maxvel = np.zeros(len(lines_to_measure)) * u.km / u.s
        ew1 = np.zeros((n_iter,len(lines_to_measure)))
        for n in range(n_iter):
            flux_new = flux[i] + np.random.normal(0,err[i].value,len(err[i])) * err[i].unit
            for j in range(len(lines_to_measure)):
                linewave = lines_to_measure[j]
                ew[j], vel[j], maxvel[j], minwave[j], maxwave[j] = measure_absline(wave, flux_new, err[i], linewave)
            ew1[n] = ew
        del_ew = np.std(ew1,axis=0) * ew.unit
        m_ew = np.mean(ew1,axis=0) * ew.unit
        print("-----------------------------------------------------")
        for ii in range(len(lines_to_measure)):
            print('EW for', (list(absline.items())[ii]), 'is', f"{m_ew.value[ii]:0.3f}",'+/-',f"{del_ew[ii]:0.3f}")
        print("-----------------------------------------------------")
        #-----------------------------------------------------------------------------
        #-----------------------------------------------------------------------------

    plt.suptitle('Red region vs Blue region',y=0.9)
    plt.xlabel("Rest wavelength (Å)", fontsize=fontsize)
    plt.ylabel("F$_{\\lambda}$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)", fontsize=fontsize)
    plt.show()


# --------------------------------------------------------------------
#                       Column Density
# --------------------------------------------------------------------
def cold_prof2(absline,f,norm_data,region,co=None): #specify absorption line| f-value| data| region(blue/red)| color
    #f-values from Donald C. Morton 10.1086/377639
    od = np.log(1/norm_data.value) #optical depth
    N_a = 3.768e14 * od / (absline[1] * f) / u.cm / u.cm / u.km * u.s #
    vel = (nwavelength.value - absline[1])/(absline[1])* c.c.to(u.km/u.s)
    # indx = lines_to_measure.index(absline)
    
    maxvel = 500 * vel.unit
    minvel = -1000 * vel.unit
    
    down = np.abs(vel-minvel)
    downer = int(np.where(down == down.min())[0])
    up =  np.abs(vel-maxvel)
    upper = int(np.where(up == up.min())[0])
    
    Ntot = np.trapz(N_a[downer:upper], vel[downer:upper])
    log_Ntot = np.log10(Ntot.value)
    # ------------------------------------------------------------------
    
    # calculate error
    n_iter = 500
    # call the error file
    # ------------------------------------------------------------------
    def namestr(obj, namespace):
        return [name for name in namespace if namespace[name] is obj]
    l=namestr(norm_data,globals())
    tr = l[0]+'_err'
    err = eval(tr)
    conterr = np.median(err[(nwavelength.value > 1250) & (nwavelength.value < 1430)])
    # ------------------------------------------------------------------
    Ntot_arr = np.zeros(n_iter)
    log_Ntot_arr = np.zeros(n_iter)
    for n in range(n_iter):
        # norm_new = norm_data + np.random.normal(0,err.value,len(err)) * err.unit
        norm_new = norm_data + np.random.normal(0, conterr.value, 1)[0] * err.unit
        od = np.log(1/norm_new.value)
        N_a = 3.768e14 * od / (absline[1] * f) / u.cm / u.cm / u.km * u.s
        vel = (nwavelength.value - absline[1])/(absline[1])* c.c.to(u.km/u.s)
        # indx = lines_to_measure.index(absline)
        
        maxvel = 500 * vel.unit
        minvel = -1000 * vel.unit
        
        down = np.abs(vel-minvel)
        downer = int(np.where(down == down.min())[0])
        up =  np.abs(vel-maxvel)
        upper = int(np.where(up == up.min())[0])
        
        Ntot = np.trapz(N_a[downer:upper], vel[downer:upper])
        log_Ntot = np.log10(Ntot.value)
        Ntot_arr[n] = Ntot.value
        log_Ntot_arr[n] = log_Ntot
    del_Ntot = np.nanstd(Ntot_arr)
    m_Ntot = np.nanmean(Ntot_arr)
    del_log_Ntot = np.nanstd(log_Ntot_arr)
    m_log_Ntot = np.nanmean(log_Ntot_arr)
    # ------------------------------------------------------------------

    # print("log(N) for %s (%s region) is %f"%(list(abslines)[indx],region,log_Ntot))
    print("log(N) for %s (%s region) is %f +/- %f"%(absline[0],region,log_Ntot,del_log_Ntot))
    
    # plot the profiles
    # ------------------------------------------------------------------
    if co==1:
        plt.step(vel,np.log10(N_a.value) * N_a.unit, lw=0.5,
                 color="%s"%region,
                #  label = "%s %s region"%(list(abslines)[indx],region))
                label = "%s %s region"%(absline[0],region))
    else:
        plt.step(vel,np.log10(N_a.value) * N_a.unit,lw=0.5, label = "%s %s region"%(absline[0],region))
        plt.title("%s region"%region)
    plt.xlabel('relative velocity(km/s)')
    plt.ylabel(r'$log N_a(v)[cm^{-2}/(km s^{-1}]$')
    plt.xlim(-1000,500)
    plt.ylim(9,)
    plt.legend()
# ------------------------------------------------------------------

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