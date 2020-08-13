import scipy
import numpy as np
import matplotlib 
from matplotlib import colors, ticker, cm
from scipy.integrate import quad
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as gs 
from matplotlib.colors import LogNorm


# box dimentions [cm]
H = 30
W = 35
L = 56
# bulb position from 0
h = 15
# bulb lenght
l = 18.5
# radiation power
P = 5.5

# experimental data

# distance [cm]
ex_d = np.array([23, 17, 15, 11])
# intensity values for any distance [mW/cm2]
ex_I = np.array([0.0975, 0.51, 0.627, 1.875])

#-------------------------------------------

def plot_2D(array):
    fig1 = plt.figure(figsize=(10, 10))
    gsf = gs(1, 1, figure=fig1)  # creating 1 by 1 figure
    ax0 = fig1.add_subplot(gsf[0])
    pos = ax0.imshow(array, cmap='jet',norm = LogNorm(), vmin=10, vmax=1e2)#, extent=[0., 90., 0., 22.5])
    ax0.set_xlabel(r'Width(cm)', fontsize=20)
    ax0.set_ylabel(r'Lenght(cm)', fontsize=20)
    cbar = fig1.colorbar(pos, ax=ax0)
    cbar.ax.set_ylabel(r'$mJ/cm^2$', fontsize=22)
    
# define function
def integrand(c, x, y, z):
    numerator = P/l
    denominator = 4*np.pi*((c-x)**2 +(H-y)**2+(h-z)**2)
    return numerator/denominator

def reflect(c, x, y, z):
    numerator = P/l
    denominator = 4*np.pi*((c-x)**2 + (H-y)**2+(h-z)**2)
    return numerator/denominator

#fitting function
def fit_func (x1, a, b):
    return a / x1**2 + b

# bulb center position
x = 28
z = 17
inf = (L-l)/2
sup = (L+l)/2

# time of irradiation
time = 1*60 # time in [s]

# create two list for put theory data
res = [] # intensity result
yi=[] # distance

for y in range(0,20,1):
    yi.append(30 - y)
    I = quad(integrand, inf, sup, args = (x,y,z))+5*(quad(reflect, inf, sup, args = (x,y,z)))
    #in mW
    res.append(I[0] * 1000)
    
# for single calculation uncomment
# distance where will calculate intensity
y=(30-20)
'''
I = quad(integrand, inf, sup, args = (x,y,z))+5*(quad(reflect, inf, sup, args = (x,y,z)))
print ("intensidade = ", I[0]*1000, "mW/cm^2" )
dose = I[0]*time
dose_mJ = dose*1000
print ("dose = ", dose, "J/cm^2")
print ("dose = ", dose_mJ, "mJ/cm^2")
'''

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('Distance to bulb, cm')
ax.set_ylabel('Intensity, mW/cm2')
ax.grid()

# plot theory curve
ax.plot(yi, res, 'bo-', label='Theory curve')

yfit = np.asarray(yi)
# calculate coeffs for fitting
popt, pcov = curve_fit(fit_func, ex_d, ex_I)
# plot experimental data
ax.plot(ex_d, ex_I, 'ro-', label='Experimental data')

# plot fitting curve
ax.plot(yfit, fit_func(yfit, popt[0], popt[1]), 'go-', label='Fit experimental')
ax.legend()
plt.show()

# for plot 2d graph uncomment
'''
# 2d figure for dose at certain distance
y = 7 # coordinate from bottom of the box
Is = np.zeros((L-1,W-1))
for x in range(0,L-1):
    for z in range (0,W-1):
        I = quad(integrand, inf, sup, args = (x,y,z))+5*(quad(reflect, inf, sup, args = (x,y,z)))
        Is[x,z] = I[0]*time*1000


plot_2D(Is)
'''
