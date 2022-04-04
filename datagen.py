import numpy as np
import matplotlib.pyplot as plt
import math
from dill import dump, loads, load

def myrandnormal(begin,end,repeat):
    return np.sum(np.random.rand(repeat)*(end-begin)+begin)/repeat

# write dill file
def writenormalizedfile(data, filename):
  # datanormal = data/np.linalg.norm(data, axis=1, ord=1)
  data /=  data.sum(axis=1)[:,np.newaxis]
  data = data.tolist()
  with open(filename, 'wb') as f:
    dump(data, f)

"""
n - number of items
m - dimensions
dist - distribution (u, z, n)
correlation - i,a,c
option
"""
def genData(n,m,count,dist='u',correlation='i',option=None):
    zipfA = 1.5
    if dist=='u' or dist == 'z':
        for c in range(count):
            if correlation =='a':    # a stands for anti-correlated
                #d=np.concatenate((np.random.rand(n, 1), (np.ones(n*(m-1))*0.5).reshape(n,m-1)), axis=1)
                d = (np.ones(n*(m))*0.5).reshape(n,m)
                r = np.random.rand(n,m) if dist == 'u' else np.random.zipf(zipfA, size=(n,m)).astype(float)
                for j in range(m):
                    for i in range(n):
                        b1 = d[i,j] if d[i,j] <= 0.5 else 0.5 - d[i,j];
                        b2 = d[i,(j+1)%m] if d[i,(j+1)%m] <= 0.5 else 0.5 - d[i,(j+1)%m];
                        if b2 < b1: b1 = b2;
                        v = (r[i,j] * 2 * b1) - b1;
                        d[i,j]+=v; d[i,(j+1)%m]-=v;
                for j in range(m):
                    for i in range(n):
                        d[i,j] = abs(d[i,j])
                # np.save('data/anti_' + dist + '_'+str(n)+'_'+str(m)+'_'+str(c)+'.npy',d)
                writenormalizedfile(d, 'data/' + correlation + '_' + dist + '_'+str(n)+'_'+str(m)+'_'+str(c)+'.dill')
            elif correlation == 'c': # c stands for correlated
                d = np.ones(n*m).reshape(n,m)
                base = np.random.rand(n,1) if dist == 'u' else np.random.zipf(zipfA, size=(n,1)).astype(float)
                if dist == 'z':
                  import pdb
                  # pdb.set_trace()
                halfrange = 0.2;
                repeat = 10 if option is None else option;
                base[base<halfrange] = 0.;
                base[base > 0.] -= halfrange
                for j in range(m):
                    for i in range(n):
                        d[i,j] = myrandnormal(base[i],2*halfrange,repeat);
                # np.save('data/anti_' + dist + '_'+str(n)+'_'+str(m)+'_'+str(option)+'_'+str(c)+'.npy',d)
                writenormalizedfile(d, 'data/' + correlation + '_' + dist + '_'+str(n)+'_'+str(m)+'_'+str(c)+'.dill')
            else: # independent
                d = np.random.rand(n,m) if dist == 'z' else np.random.zipf(zipfA, size=(n,m)).astype(float)
                # np.save('data/anti_' + dist + '_'+str(n)+'_'+str(m)+'_'+str(c)+'.npy', d)
                writenormalizedfile(d, 'data/' + correlation + '_' + dist + '_'+str(n)+'_'+str(m)+'_'+str(c)+'.dill')

def plotpoly(poly):
    #plt.plot(poly[:, 0], poly[:, 1], 'ok', markersize=8)
    for p in poly:
        plt.plot(p[0], p[1], 'ro', markersize=10)
    plt.show()

def point_in_poly(x,y,poly):

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


def getNextinShape(poly,dist='u'):
    while(True):
        p=np.random.rand(1,2)
        if point_in_poly(p[0,0],p[0,1],poly): return p;

def genCurvedData(n,count,corners=1,dist='u'):
    if corners > 0:
        poly =[(0,0)];
        if corners==1:
            poly.append((1,0)); poly.append((1,1)); poly.append((0,1))
        else:
            for i in range(corners+2):
                tetha = i*math.pi/(2*corners+2)
                if i == 1:
                    poly.append((1,math.sin(tetha)))
                elif i==corners:
                    poly.append((math.cos(tetha),1))
                else:
                    poly.append((math.cos(tetha),math.sin(tetha)))
        #plotpoly(poly)
        for c in range(count):
            d = np.zeros(n*2).reshape(n,2)
            for j in range(n):
                d[j] = getNextinShape(poly,dist);
            #plotpoly(d)
            np.save('data/curved_u_'+str(n)+'_'+str(corners)+'corners_'+str(c)+'.npy',d)


if __name__ == '__main__':
    points = 1000
    nodatasets = 1
    dims = list(range(5, 14))
    correlations = ['a', 'i', 'c']
    dists = ['z'] #['u', 'z']
    for dim in dims:
        for distval in dists:
            for corr in correlations:
                genData(points, dim, nodatasets, dist=distval,correlation=corr)
