##################################################
# import
##################################################
import numpy as np
import math
import sympy as sy
import matplotlib.pyplot as plt

def emma(Ax, Ay, Bx, By, Cx, Cy, Px, Py):
    ######################################################
    ## define triangle
    ######################################################
    # create arrays
    a = np.array([Ax,Ay])
    b = np.array([Bx,By])
    c = np.array([Cx,Cy])
    p = np.array([Px,Py])
    # calculate triangle sides - vector direction
    AB = b - a
    BC = c - b
    CA = a - c
    # calculate triangle sides - vector magnitude
    lengthAB=math.sqrt(AB[0]**2+AB[1]**2)
    lengthBC=math.sqrt(BC[0]**2+BC[1]**2)
    lengthCA=math.sqrt(CA[0]**2+CA[1]**2)

    # calculate triangle height - vector magnitude
    s = (lengthAB+lengthBC+lengthCA)/2
    ha = (2/lengthBC) * math.sqrt(s*(s-lengthAB)*(s-lengthBC)*(s-lengthCA))
    hb = (2/lengthCA) * math.sqrt(s*(s-lengthAB)*(s-lengthBC)*(s-lengthCA))
    hc = (2/lengthAB) * math.sqrt(s*(s-lengthAB)*(s-lengthBC)*(s-lengthCA))
    # calculate triangle height - vector direction
    # height A:
    t = sy.S('t')
    pha = b+t*BC
    phaA = a - pha
    y = sy.solve( sy.Eq((BC[0]*phaA[0])+(BC[1]*phaA[1]), 0))
    pha = b+y*BC
    phaA= a - pha
    # height B:
    t = sy.S('t')
    phb = c+t*CA
    phbB = b - phb
    y = sy.solve( sy.Eq((CA[0]*phbB[0])+(CA[1]*phbB[1]), 0))
    phb = c+y*CA
    phbB= b - phb
    # height C:
    t = sy.S('t')
    phc = a+t*AB
    phcC = c - phc
    y = sy.solve( sy.Eq((AB[0]*phcC[0])+(AB[1]*phcC[1]), 0))
    phc = a+y*AB
    phcC= c - phc

    ######################################################
    ## triangle calculations
    ######################################################
    # calculate point of intersection of triangle height at right angles to P
    # point of intersection SPA:
    t = sy.S('t')
    SPA = pha+t*phaA
    SPAP = p - SPA
    y = sy.solve( sy.Eq((phaA[0]*SPAP[0])+(phaA[1]*SPAP[1]), 0))
    SPA = pha+y*phaA
    SPAP= p - SPA
    # point of intersection SPB:
    t = sy.S('t')
    SPB = phb+t*phbB
    SPBP = p - SPB
    y = sy.solve( sy.Eq((phbB[0]*SPBP[0])+(phbB[1]*SPBP[1]), 0))
    SPB = phb+y*phbB
    SPBP= p - SPB
    # point of intersection SPC:
    t = sy.S('t')
    SPC = phc+t*phcC
    SPCP = p - SPC
    y = sy.solve( sy.Eq((phcC[0]*SPCP[0])+(phcC[1]*SPCP[1]), 0))
    SPC = phc+y*phcC
    SPCP= p - SPC

    # calculate distance from point of intersection to triangle corner
    # distance to A:
    SPAA= a - SPA
    lengthSPAA=math.sqrt(SPAA[0]**2+SPAA[1]**2)
    # distance to B:
    SPBB= b - SPB
    lengthSPBB=math.sqrt(SPBB[0]**2+SPBB[1]**2)
    # distance to C:
    SPCC= c - SPC
    lengthSPCC=math.sqrt(SPCC[0]**2+SPCC[1]**2)

    # percentages of components
    # A:
    phaSPA = SPA- pha
    lengthphaSPA=math.sqrt(phaSPA[0]**2+phaSPA[1]**2)
    z = 100/ha*lengthphaSPA
    print('percentage of component A',z)
    # B:
    phbSPB = SPB- phb
    lengthphbSPB=math.sqrt(phbSPB[0]**2+phbSPB[1]**2)
    k = 100/hb*lengthphbSPB
    print('percentage of component B', k)
    # C:
    phcSPC = SPC- phc
    lengthphcSPC=math.sqrt(phcSPC[0]**2+phcSPC[1]**2)
    q = 100/hc*lengthphcSPC
    print('percentage of component C', q)

    #####################################################
    ## Plot
    #####################################################
    # mixing triangle with graphic representation of the calculation
    # of the percentage of component A
    fig, ax1 = plt.subplots()
    # corners
    plt.plot(a[0], a[1], marker='.', color='black')
    plt.plot(b[0], b[1], marker='.', color='black')
    plt.plot(c[0], c[1], marker='.', color='black')
    plt.plot(p[0], p[1], marker='.', color='black')
    # label corners
    ax1.text(a[0], a[1], 'A', ha='right')
    ax1.text(b[0], b[1], 'B', ha='left')
    ax1.text(c[0], c[1], 'C', ha='right')
    ax1.text(p[0], p[1], 'P', ha='right')
    # triangle sides
    # AB:
    plt.plot([a[0], b[0]], [a[1], b[1]], color='black')
    # BC:
    plt.plot([b[0], c[0]], [b[1], c[1]], color='black')
    # CA:
    plt.plot([c[0], a[0]], [c[1], a[1]], color='black')
    # point of intersection  pha:
    plt.plot(pha[0], pha[1], marker='x', color='blue')
    ax1.text(pha[0], pha[1], 'pha', ha='right', color='blue')
    # height A (phaA):
    plt.plot([pha[0], a[0]], [pha[1], a[1]], label='ha')
    # point of intersection SPA:
    plt.plot(SPA[0], SPA[1], marker='x', color='blue')
    ax1.text(SPA[0], SPA[1], 'SPA', ha='right', color='blue')
    # point of intersection of triangle height at right angles to P (SPAP):
    plt.plot([SPA[0], p[0]], [SPA[1], p[1]])
    # formatting
    plt.ylabel('y-axis')
    plt.xlabel('x-axis')
    plt.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)

    return z, k, q