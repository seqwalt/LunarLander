import numpy as np
from numpy import cos, sin, pi
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.patches import Polygon
import os

def VisualizeLander(X,Y,ANG,THRUST,TORQUE,TIME,REF,meta_data):

    # Arguments are vectors over time

    # points of lander body (B)
    a = 1.2     # width of enclosing box of body
    b = 1       # height of enclosing box of body
    wa = .7     # length of top/bottom side of body
    wb = .5     # length of left/right side of body
    abv = 0.25  # dist bottom of body is above ground
    Bp1 = [-wa/2,0+abv]; Bp2 = [wa/2,0+abv]; Bp3 = [a/2,(b-wb)/2+abv]; Bp4 = [a/2,(b+wb)/2+abv]
    Bp5 = [wa/2,b+abv]; Bp6 = [-wa/2,b+abv]; Bp7 = [-a/2,(b+wb)/2+abv]; Bp8 = [-a/2,(b-wb)/2+abv]
    # points of lander legs (L1 and L2 for left and right leg)
    n = 0.3; m = 2  #leg thinkness (n) and slope (m)
    c1 = (0.5+n)*(a-wa)/2; c2 = (0.5-n)*(b-wb)/2
    c3 = (0.5-n)*(a-wa)/2; c4 = (0.5+n)*(b-wb)/2
    b1 = -m*(c1-a/2)+c2+abv  # y = mx + b1
    b2 = -m*(c3-a/2)+c4+abv  # y = mx + b2
    L1p1 = [c1-a/2, c2+abv]; L1p2 = [c3-a/2, c4+abv]
    L1p3 = [-b2/m,0]; L1p4 = [-b1/m,0]
    L2p1 = [-c1+a/2, c2+abv]; L2p2 = [-c3+a/2, c4+abv]
    L2p3 = [b2/m,0]; L2p4 = [b1/m,0]
    # points of thruster (T)
    Tp1 = [0.1,abv];  Tp2 = [0.2,abv-0.1]
    Tp4 = [-0.1,abv]; Tp3 = [-0.2,abv-0.1]
    # initial points of thrust fire (F)
    Fp1 = [-0.14,abv-0.1]; Fp2 = [0.14,abv-0.1]
    Fp3 = [0, abv-0.1]

    # Collect points into arrays, and center about rotation point
    rotPnt = np.array(([0],[b/2+abv]))   # Point of rotation
    BodyPts = np.array((Bp1,Bp2,Bp3,Bp4,Bp5,Bp6,Bp7,Bp8)).T - rotPnt # Center about rotation point
    Leg1Pts = np.array((L1p1,L1p2,L1p3,L1p4)).T - rotPnt
    Leg2Pts = np.array((L2p1,L2p2,L2p3,L2p4)).T - rotPnt
    TPts    = np.array((Tp1,Tp2,Tp3,Tp4)).T - rotPnt

    # Size of figure
    ratio = (max(X)+1 - (min(X)-1))/(max(Y)+2 - (min(Y)-0.5))
    y_inches = (max(Y)+2 - (min(Y)-0.5))/1.15
    x_inches = ratio*y_inches

    os.system("rm temp_imgs/*.png")
    fps = meta_data[3]    # frames/second
    T_aft = 2 #time to keep video going after reaching goal

    print('Creating '+str(fps)+' FPS movie...')
    print()

    for j in range(int(fps*(TIME[-1]+T_aft))):
        # set figure size
        fig = plt.gcf()
        fig.set_size_inches(x_inches,y_inches)

        # want i-th data point
        i = j*int(len(TIME)/int(fps*TIME[-1]))

        # States and controls
        if i < len(TIME):
            x   = X[i]
            y   = Y[i] + b/2 + abv
            pos = np.array(([x],[y]))
            ang = ANG[i]
            th  = THRUST[i]
        elif y < b/2 + abv + 5e-3: # thrust off after landing
            th = 0

        # Collect control graphics, and center about rotation pt.
        Fp3 = [0, abv-0.1 - th/200]
        FPts  = np.array((Fp1,Fp2,Fp3)).T - rotPnt  # thrust fire

        # Rotation of lander
        R = np.array(([cos(ang), -sin(ang)],  # Rotation matrix
                      [sin(ang), cos(ang)]))
        BPTS = (R@BodyPts + pos).T # Rotate, then translate to original location
        L1PTS = (R@Leg1Pts + pos).T
        L2PTS = (R@Leg2Pts + pos).T
        TPTS  = (R@TPts + pos).T
        FPTS  = (R@FPts + pos).T

        # Plot Polygons
        bod = Polygon(BPTS, fc="w", ec="k")
        l1 = Polygon(L1PTS, fc="w", ec="k")
        l2 = Polygon(L2PTS, fc="w", ec="k")
        thr = Polygon(TPTS, fc="k", ec="k")
        fire = Polygon(FPTS, fc="r", ec="k")
        ax = plt.gca()
        ax.add_patch(bod)
        ax.add_patch(l1)
        ax.add_patch(l2)
        ax.add_patch(thr)
        ax.add_patch(fire)
        #plt.plot(rotPnt[0]+pos[0],rotPnt[1]+pos[1],'ko') # pt of rotation
        plt.plot(X,Y + b/2 + abv,'g') # trajectory
        plt.plot(x,y,'go') # current position
        plt.plot([min(X)-1,max(X)+1],[0,0],"k")
        plt.plot(REF[0],REF[1] + b/2 + abv,"bo")
        ax.set_xlim(min(X)-1,max(X)+1)
        ax.set_ylim(min(Y)-0.5,max(Y)+2)
        plt.savefig("temp_imgs/image" + str(j).zfill(4) + ".png")
        plt.close()
    filename = meta_data[2]
    os.system("ffmpeg -hide_banner -loglevel error -y -framerate "+str(fps)+" -i temp_imgs/image%04d.png \
        -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" -vcodec libx264 -crf 20 -pix_fmt \
        yuv420p movies/"+filename+".mp4")

    if meta_data[1] == 1:
        if meta_data[0] == "linux":
            os.system("vlc -q movies/"+filename+".mp4 2> /dev/null")
        elif meta_data[0] == "mac":
            os.system("open movies/"+filename+".mp4")
        else:
            raise Exception("Incorrect op_sys value.")
