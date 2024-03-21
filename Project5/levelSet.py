# % Class to perform level set segmentation
# % gradientNB, curvatureNB, DuDt2 functions need to be added
# % ECE 8396: Medical Image Segmentation
# % Spring 2024

import numpy as np
from fastMarching import *
import skimage.filters
import skimage.feature
from laplacianSmoothing import *

class levelSetParams:
    def __init__(self, method=None, alpha=.2, mindist=3.1, sigma=.5, inflation=1, tau=1, beta=0.1,
                 epsilon=1e-7, maxiter=1000, convthrsh=1e-2, reinitrate=1, visrate=0, dtt=None,
                 lmbda=1, mu=1, gvft=.9):
        if method is None:
            self.method='CS'
        else:
            self.method=method

        self.alpha = alpha
        self.mindist = mindist
        self.sigma = sigma
        self.v = inflation
        self.tau = tau
        self.beta = beta
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.convthrsh = convthrsh
        self.reinitrate = reinitrate
        self.visrate = visrate
        if dtt is None:
            self.dtt = np.ones(maxiter)
        else:
            self.dtt = dtt

        self.lmbda = lmbda
        self.mu = mu
        self.gvft = gvft


class levelSet:
    def __init__(self):
        self.fm = fastMarching()
        self.r=self.c=self.d=0
        self.normG = None

    def DuDt1(self,v,tau,speed,nG,kappa,G,gradspeed):
        return speed * nG * (kappa + v) + tau * np.sum(G * gradspeed,axis=0)

    def DuDt4(self, mu, tau, kappa, G, gvf):
        return mu*kappa + tau * np.sum(G * gvf, axis=0)

    def gradientImage(self, img):

        gradim = np.zeros((3,self.r,self.c,self.d))
        gradim[0,1:self.r - 1,:,:] = (img[2:,:,:] - img[0:self.r - 2,:,:]) / 2
        gradim[0,0,:,:] = img[1,:,:] - img[0,:,:]
        gradim[0,-1,:,:] = img[-1,:,:] - img[-2,:,:]
        gradim[1,:,1:self.c - 1,:] = (img[:,2:,:] - img[:,0:self.c - 2,:]) / 2
        gradim[1,:,0,:] = img[:,1,:] - img[:,0,:]
        gradim[1,:,-1,:] = img[:,-1,:] - img[:,-2,:]
        if self.d>1:
            gradim[2,:,:,1:self.d - 1] = (img[:,:,2:] - img[:,:,0:self.d - 2]) / 2
            gradim[2,:,:,0] = img[:,:,1] - img[:,:,0]
            gradim[2,:,:,-1] = img[:,:,-1] - img[:,:,-2]
        return gradim



    def segment(self, img, dmap_init, params=levelSetParams(), ax=None):
        self.r,self.c,self.d = np.shape(img)
        self.normG = np.zeros((3,self.r,self.c,self.d))

        if params.method != 'CV':
            if params.sigma > 0:
                imblur = skimage.filters.gaussian(img, params.sigma)
            else:
                imblur = img

            gradim = self.gradientImage(imblur)
            ngradimsq = np.sum(gradim*gradim, axis=0)
            speed = np.exp(-ngradimsq / (2 * params.alpha * params.alpha)) - params.beta
            speed[speed<params.epsilon] = params.epsilon
            gradspeed = self.gradientImage(speed)

        if params.method == 'GVF':
            X,Y,Z = np.meshgrid(range(self.r), range(self.c), range(self.d), indexing='ij')
            X = np.ravel(X, order='F')
            Y = np.ravel(Y, order='F')
            Z = np.ravel(Z, order='F')
            ngradspeed = np.linalg.norm(gradspeed, axis=0)
            mx = np.max(ngradspeed)
            msk = np.ravel(ngradspeed>params.gvft*mx, order='F')
            imsk = 1-msk
            dirn = X[msk] + self.r*(Y[msk] + self.c*Z[msk])
            intn = X[imsk] + self.r*(Y[imsk] + self.c*Z[imsk])
            gvfx = laplacianSmoothing(self.r,self.c,self.d,dirn,gradspeed[0,X[msk],Y[msk],Z[msk]],
                                      intn,gradspeed[0, X[imsk], Y[imsk], Z[imsk]],
                                      (ngradspeed[X[imsk],Y[imsk],Z[imsk]] / (params.gvft * mx))**2)
            gvfy = laplacianSmoothing(self.r,self.c,self.d,dirn,gradspeed[1,X[msk],Y[msk],Z[msk]],
                                      intn,gradspeed[1,X[imsk],Y[imsk],Z[imsk]],
                                      (ngradspeed[X[imsk],Y[imsk],Z[imsk]] / (params.gvft * mx))**2)
            gvfz = laplacianSmoothing(self.r,self.c,self.d,dirn,gradspeed[2,X[msk],Y[msk],Z[msk]],
                                      intn,gradspeed[2,X[imsk],Y[imsk],Z[imsk]],
                                      (ngradspeed[X[imsk],Y[imsk],Z[imsk]] / (params.gvft * mx))**2)
            gvf = np.concatenate((gvfx[np.newaxis,:,:,:], gvfy[np.newaxis,:,:,:], gvfz[np.newaxis,:,:,:]), axis=0)
            if params.visrate>0:
                f = plt.gcf()
                fig, axgvf = plt.subplots(2,2)
                axgvf[0,0].imshow(gvfx[:,:,self.d//2].T, 'gray')
                plt.axes(axgvf[0,0])
                plt.xlabel('GVF(x)')
                axgvf[0,1].imshow(gvfy[:,:,self.d//2].T, 'gray')
                plt.axes(axgvf[0,1])
                plt.xlabel('GVF(y)')
                axgvf[1,1].imshow(img[:,:,self.d//2].T, 'gray')
                plt.axes(axgvf[1,1])
                axgvf[1,0].imshow(img[:,:,self.d // 2].T,'gray')
                plt.axes(axgvf[1,0])
                plt.figure(f)

        iter=0
        self.fm.dmap = np.array(dmap_init, dtype=np.float64)
        self.fm.voxsz = np.array([1.,1.,1.])
        delta=params.convthrsh+1

        while iter < params.maxiter:
            if iter%params.reinitrate==0:
                self.fm.update(nbdist=params.mindist)

                nbin, nbout = self.fm.getNB()
                nbinone = nbin
                nboutone = nbout
                for i in range(len(nbin)):
                    if nbin[i].d>1:
                        nbinone = nbin[0:i]
                        break
                for i in range(len(nbout)):
                    if nbout[i].d>1:
                        nboutone = nbout[0:i]
                        break
                if iter>0:
                    delta = 0
                    for i in range(len(nbinold)):
                        delta += np.abs(-self.fm.dmap[ nbinold[i].x,  nbinold[i].y,  nbinold[i].z] -  nbinold[i].d)
                    for i in range(len(nboutold)):
                        delta += np.abs(self.fm.dmap[nboutold[i].x, nboutold[i].y, nboutold[i].z] - nboutold[i].d)
                    N = len(nbinold) + len(nboutold)
                    if N==0:
                        break
                    delta /= len(nbinold) + len(nboutold)
                    if delta<params.convthrsh:
                        break

                nbinold = nbinone
                nboutold = nboutone

            kappa, G, nG, xyz = self.curvatureNB(params.epsilon)
            if np.sum(np.isnan(kappa))>0:
                self.fm.update(nbdist=params.mindist)
                kappa,G,nG,xyz = self.curvatureNB(params.epsilon)

            if (params.method != 'CV') and (params.method != 'GVF'):
                dudt = self.DuDt1(params.v,params.tau,speed[xyz[0,:], xyz[1,:], xyz[2,:]],nG,kappa, G,
                                  np.concatenate((
                                      gradspeed[0,xyz[0,:],xyz[1,:],xyz[2,:]][np.newaxis,:],
                                      gradspeed[1,xyz[0,:],xyz[1,:],xyz[2,:]][np.newaxis,:],
                                      gradspeed[2,xyz[0,:],xyz[1,:],xyz[2,:]][np.newaxis,:]), axis=0))
            elif params.method == 'CV':
                msk = self.fm.dmap<=0
                c1 = np.mean(img[msk])
                c2 = np.mean(img[1-msk])
                dudt = self.DuDt2(params.mu, params.lmbda, kappa, img[xyz[0,:], xyz[1,:], xyz[2,:]], c1, c2)

            elif params.method == 'GVF':
                dudt = self.DuDt4(params.mu, params.tau, kappa, G, gvf)

            if len(dudt)>0:
                dt = params.dtt[iter] / np.max(np.abs(dudt))
                self.fm.dmap[xyz[0,:], xyz[1,:], xyz[2,:]] += dt * dudt
                if params.visrate>0 and iter%params.visrate==0:
                    plt.axes(ax[0])
                    plt.cla()
                    ax[0].imshow(img[:,:,self.d//2].T, 'gray')
                    X,Y = np.meshgrid(range(self.r),range(self.c),indexing="ij")
                    plt.contour(X,Y,self.fm.dmap[:,:,self.d // 2],levels=[0.0],colors='red')
                    plt.title(f'Iteration {iter}, Delta={delta}')
                    plt.axes(ax[1])
                    plt.cla()
                    plt.gca().imshow(self.fm.dmap[:,:,self.d//2].T, 'gray', vmin=-10, vmax=20)
                    plt.contour(X,Y,self.fm.dmap[:,:,self.d // 2],levels=[0.0],colors='red')
                    plt.title('Distance map')
                    plt.gcf().canvas.draw_idle()
                    plt.gcf().canvas.start_event_loop(0.01)
            iter += 1

        self.fm.update(nbdist=params.mindist)
        return self.fm.dmap



