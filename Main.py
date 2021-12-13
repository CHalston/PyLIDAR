#Simulation of Lidar measuring wind speed array

import scipy.integrate as integrate
from scipy.integrate import quad
import scipy.special as special
import numpy as np
import math
import warnings
from matplotlib import pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def myround(x, prec=2, base=.05):
  return round(base * round(float(x)/base),prec)

#This is in case v,w != 0
#def get_uLOS(u,v,w,phi):
#    uLOS = np.sqrt((u**2)+(v**2)+(w**2))*np.cos(phi)
#    return uLOS

def get_uLOS_approx(u,phi):
    uLOS = u*np.cos(phi)
    return uLOS

def get_Weight_func (R,kN,F,Rr):
    W_fun = kN/((R**2)+(((1-(R/F))**2)*(Rr**2)))
    return W_fun

def get_Integrand(R,F,kN,Rr,phi,u):
    W = get_Weight_func(R,kN,F,Rr)
    Integrand = u*W
    return Integrand

def get_uLOSprime(F,kN,Rr,phi,u):
    uLOSprime = quad(get_Integrand, -np.inf, np.inf, args=(F,kN,Rr,phi,u))
    return uLOSprime

def get_kN(F,R,Rr):
    denom = integrate.quad((1/((R**2)+(((1-(R/F))**2)*(Rr)**2))), 0, F)
    kN = 1/denom
    return kN

def get_y_index(y):
    y_index = ((y+40)/6.667)
    return y_index

def get_z_index(z):
    z_index = ((-z+40)/6.667)
    return z_index

def get_3d_array(component):
    # Read text file into array
    comp_file = open("TurbSim." + component, "r")
    # u_data saved as list
    comp_data = comp_file.read().splitlines()
    comp_file.close()
    comp_array_raw = comp_data

    # removing all 'empty' rows from u_array
    x = 0
    for row in comp_array_raw:
        if row == '':
            comp_array_raw = np.delete(comp_array_raw, (x), axis=0)
        elif row != '':
            x += 1

    # remove non essential header data up to 'Z'
    x = 0
    while x < 3:
        comp_array_raw = np.delete(comp_array_raw, (0), axis=0)
        x += 1
    # NOTE: could potentially delete up to the row with Z as first value, but this works well for now

    # Next convert the strings(rows) to arrays using .split()
    x = 0
    # u_array will be an array of lists, with each list item being separately indexed
    comp_array = []
    for row in comp_array_raw:
        row = row.split()
        comp_array.append(row)
        x += 1

    # Now build two arrays, one 2d array with the (Z,Y) coords, the other a 3d array with the u data
    # First for loop builds the 2d ZY array containing a row of z coords and a row of y coords
    zy_array = []
    x = 0
    for row in comp_array:
        if comp_array[x][0] == 'Z':
            zy_array.append(comp_array[x + 1])
        if comp_array[x][0] == 'Y':
            zy_array.append(comp_array[x + 1])
        x += 1

    # Second for loop builds the 3d array of u data
    # Need to start by removing first 4 rows
    x = 0
    while x < 4:
        comp_array = np.delete(comp_array, (0), axis=0)
        x += 1

    # Convert u_array entries to floats in u_floats
    comp_floats = []
    for row in comp_array:
        comp_floats.append([float(x) for x in row])

    # Input data from 2d array into 3d array using timestamps as index reference
    x = 0.000
    y = 0.050
    comp_3d = []


    tempArray = []
    index = 0
    for row in comp_floats:
        if (row[0] == round(x, 3)) and len(row) == 2:
            continue
        if row[0] != (round(y, 3) or round(x, 3)) or len(row) > 2:
            tempArray.append(row)
            if row == comp_floats[-1]:
                comp_3d.append(tempArray)
            continue
        if (row[0] == round(y, 3)) and len(row) ==2:
            comp_3d.append(tempArray)
            tempArray = []
            x += 0.050
            y += 0.050
            continue
    return comp_3d

def get_full_3d_array(u_3d,v_3d,w_3d):
    full_3d = []
    x = 0
    # Iterates over each value in 3d array
    for array in u_3d:
        y = 0
        tempArray = []
        for row in array:
            z = 0
            # tempRow is the row filled with lists of u,v,w data
            tempRow = []
            for item in row:
                # tempList is the u,v,w data for one specific 3D index location
                tempList = []
                tempList.append(u_3d[x][y][z])
                tempList.append(v_3d[x][y][z])
                # appending w to the temp list seems to leave errors
                tempList.append(w_3d[x][y][z])
                tempRow.append(tempList)
                z += 1
            tempArray.append(tempRow)
            y += 1
        full_3d.append(tempArray)
        x += 1
    return full_3d

def get_points(u,D_index,z,y):
    #creates a list of four triplets: (z,y,value)
    #input u as array, D_index as int, z and y as floats
    z1 = math.floor(z)
    z2 = math.ceil(z)
    y1 = math.floor(y)
    y2 = math.ceil(y)
    val11 = u[D_index][z1][y1]
    val12 = u[D_index][z1][y2]
    val21 = u[D_index][z2][y1]
    val22 = u[D_index][z2][y2]
    point1 = [y1,z1,val11]
    point2 = [y2,z1,val12]
    point3 = [y1,z2,val21]
    point4 = [y2,z2,val22]
    points = [point1,point2,point3,point4]
    return points

def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

# Build 3 3D arrays of wind velocity vectors from TurbSim output files
# The 3D arrays are each made up of separate components u, v, and w
u_3d = get_3d_array("u")
v_3d = get_3d_array("v")
w_3d = get_3d_array("w")

# Combine u,v,w arrays into one 3d array
full_3d = get_full_3d_array(u_3d,v_3d,w_3d)


#Mean u velocity is detailed in the TurbSim output file, but for ease of use is set manually here
u_mean = 18.20

#D is the distance upwind from the rotor being examined
# This is a user input value
D = input("Please input Distance upwind in meters:\n")
print("Your input: " + D + " m")
D = float(D)
#From here, F must be converted to an array index
#First, get the time step value for the user set distance
D_adj = D/u_mean
#Next find the index value for F based on the 0.05 second time step set in TurbSim
D_index = myround(D_adj/0.05)
D_index = int(D_index)

#r is the radius of the LIDAR scanned circle
#This value is input by the user
r = input("Please input radius of scanned circle in meters:\n")
print("Your input: " + r + " m")
r = float(r)

# F is the focal distance from the Lidar to the point being measured
# F is equal to the root of (D^2)+(r^2), which are the x and radial distances from the lidar to the
# point being measured
F = np.sqrt((D**2)+(r**2))

#Hub height (m) is set within TurbSim, and for simplicity it is set manually here
Hhub = 84.29

# according to the NREL LIDAR document, "For the commercially available Doppler LIDAR system modeled,
# Rr is approximately 1570m."
Rr = 1570

#In order to solve for Kn, first need to find initial W(F,R)
W = quad(get_Weight_func, 0,np.inf, args=(1,F,Rr))
kN = 1/(W[0])

#Next solve for phi/theta, assuming u >> v,w
phi = np.arctan(r/D)

#For code simplification, LIDAR will take measurements of 50 equally spaced points
#This is due to the LIDAR model having a 50Hz sampling rate @ 60rpm
#n is the number of points
n = 50

#Next find the azimuth angle between points ie radians between points
azi = 2*np.pi/50

#Next iterate through each point the LIDAR reads in a single rotation
#Save the resultant values to an array/list
x = 0
azi_ref = 0
mainLidar = []
lH = float(input("Please input second Lidar station height:\n"))
doubleLidar = []
uCircle = []
azimuthAngle = []

while x < n:
    y = r * np.cos(azi_ref)
    z = r * np.sin(azi_ref)
    y_ind = get_y_index(y)
    z_ind = get_z_index(z)
    points = get_points(u_3d,D_index,z_ind,y_ind)
    uInterpolated = bilinear_interpolation(y_ind,z_ind,points)
    uCircle.append(uInterpolated)
    azimuthAngle.append(azi_ref)

    #For main LIDAR station
    uLOSprime = get_uLOSprime(F, kN, Rr, phi, uInterpolated)
    mainLidar.append(uLOSprime[0])

    #For second LIDAR station
    zFLidar = z + Hhub - lH
    yFLidar = y
    FLidar = np.sqrt((D**2)+(yFLidar**2)+(zFLidar**2))
    phi2 = math.acos(D/FLidar)
    uLOSprime2 = get_uLOSprime(FLidar, kN, Rr, phi2, uInterpolated)
    uLOSprimemean = (uLOSprime2[0] + uLOSprime[0])/2
    doubleLidar.append(uLOSprimemean)

    azi_ref += azi
    x += 1

#Now to plot our results
x = np.linspace(0, 2*np.pi)
plt.xlim(0,2*np.pi)
plt.plot(x,uCircle,label = 'u actual')
plt.plot(x,mainLidar,label = 'single LIDAR')
plt.plot(x,doubleLidar,label = 'double LIDAR')
plt.xlabel('Azimuth angle A (rads)')
plt.ylabel('Wind speed u (m/s)')
plt.title('Comparison of single and double LIDAR setups with actual windspeed')
plt.legend()
plt.grid()
plt.show()

#print(azimuthAngle)
#print(uCircle)
#print(mainLidar)
#print(doubleLidar)

x = 0
perErrorMain = []
perErrorDoub = []
for val in uCircle:
    mainerror = 100*((mainLidar[x]-uCircle[x])/(uCircle[x]))
    perErrorMain.append(mainerror)
    douberror = 100*((doubleLidar[x]-uCircle[x])/(uCircle[x]))
    perErrorDoub.append(douberror)
    x += 1

x = np.linspace(0, 2*np.pi)
plt.xlim(0,2*np.pi)
plt.plot(x,perErrorMain,label='% error Single LIDAR')
plt.plot(x,perErrorDoub,label='% error Double LIDAR')
plt.xlabel('Azimuth angle A (rads)')
plt.ylabel('% error')
plt.title('Percent error for single and double LIDAR setups',y=1.05)

plt.legend()
plt.grid()
plt.show()

avgErrorMain = sum(perErrorMain)/len(perErrorMain)
avgErrorDoub = sum(perErrorDoub)/len(perErrorDoub)
print("The average error of the Single Lidar setup = ",avgErrorMain)
print("The average error of the Double Lidar setup = ",avgErrorDoub)
perErrorDiff = avgErrorMain - avgErrorDoub
print("The percent error improvement of double LIDAR setup = ",perErrorDiff)