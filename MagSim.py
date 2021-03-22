######################################################################
#
# MagSim – Magnetic Simulation software
#
# Copyright (C) 2021  Antoine Coulon (Institut Curie - CNRS).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact: software@coulonlab.org - www.coulonlab.org
#
######################################################################

__version__='2.0.5'

from scipy import *
from matplotlib import pyplot as plt, cm


######################################################################
###### Class 'Model' #################################################

class Model():
    """Represents the physical environment of the simulation and the set of magnetic objects in it."""

#==============================================#
# Define the physical simulation environemt
#==============================================#
    def __init__(self, T, mu=4*pi*1e-7, Bext=None, magSatMNP=3e-20):
        """Defines the situation to simulate.
        T: Temperature (in K)
        mu: Magnetic permeability (in T.m/A). Default: vacuum
        Bext: external magnetic field. Array of z, y and x coordinates
           (in Tesla). Default is no field.
        magSatMNP: Magetic moment of particles at saturation (in A.m^2),
           e.g. from SQUID measurement. Default: 3e-20"""
        self.T=T
        self.kB_T=1.38065e-23*T
        self.mu=mu
        if type(Bext)==type(None): self.Bext=r_[0,0,0]
        else: self.Bext=Bext
        self.permDipoles=[]
        self.magSatMNP=magSatMNP
    
#==============================================#
# Add magnetic objects in the simulation
#==============================================#
    def addBlock(self,orig,size,M,nbDipoles=20,yxOrient=None,model='perm',Ms=None,a=None,chi=None,k=None,alpha=None,c=None):
        """Adds to the model a list of dipoles representing a block of material.
          - orig:      3D coordinate of the origin corner (ZYX in meters)
          - size:      Size of the block (ZYX in meters)
          - M:         Magnetization vector, i.e. magnetic moment per
                       unit volume (ZYX in A/m)
          - nbDipoles: Number of dipoles in every dimension
          - yxOrient: Optional base for orienting the block. yxOrient[0] and
                       yxOrient[1] are the local y and x axes, expressed in the
                       absolute frame of reference./
          - model:     {'perm', 'paramag', 'ferromag'}
            => 'perm' (no additional parameters required)
            => 'paramag': [NOT IMPLEMENTED YET]
               - Ms:    Saturation magnetization (in A/m)
               - a:     Magentizing field at 1/3 magnetization (in A/m)
               - chi:   Magnetic susceptibility (in 1). Alternative to a.
            => 'ferromag' [NOT IMPLEMENTED YET]. Include parameters for 'paramag', as well as:
               - k:     Field to break pinning site (in A/m)
               - alpha: Inter-domain coupling (in 1)
               - c:     Magnetic reversibility (in range 0..1)
          """
        if type(size)==float: size=size*r_[1,1,1]
        if type(nbDipoles)==int: nbDipoles=(size/((prod(size)/nbDipoles)**(1./3))).round().astype(int)
        if type(yxOrient)==type(None): zyx=eye(3)
        else:
            zyx=eye(3); zyx[1:]=yxOrient
            zyx[2]=zyx[2]/sum(zyx[2]**2)**.5 # Normalize x
            zyx[1]-=zyx[2]*sum(zyx[2]*zyx[1]); zyx[1]=zyx[1]/sum(zyx[1]**2)**.5 # Make y ortho to x and normalize it
            zyx[0]=-cross(zyx[2],zyx[1]) # Calculate z ortho to x and y

        if   model in ['perm','permanent']: materialProperties=()
        elif model in ['para','param','paramag','paramagnetic','f','ferro','ferromag','ferromagnetic']:
            raise ValueError("This model is not implemented yet.")
            #if not (Ms!=None and ((a!=None) != (chi!=None))): raise ValueError("Need to define Ms and either a or chi.")
            #if a==None: a=Ms/(3*chi)
            #if model in ['para','param','paramag','paramagnetic']: materialProperties=(Ms,a)
            #else: materialProperties=(Ms,a,k,alpha,c)
        else: raise ValueError("'model' parameter not recognized.")

        self.permDipoles.extend([ (orig+dot((r_[iz,iy,ix]+.5)*size/nbDipoles,zyx), M*prod(size/nbDipoles)) + materialProperties
          for ix in r_[:nbDipoles[2]] for iy in r_[:nbDipoles[1]] for iz in r_[:nbDipoles[0]]])


#==============================================#
# Calculation of the fields
#==============================================#

    def calcAfield(s,pos):
        dipPos=array([dip[0] for dip in s.permDipoles])
        m     =array([dip[1] for dip in s.permDipoles])
        r,r_norm=calcAllDist(pos,dipPos); r_unit=(r.T/r_norm.T).T
        A=sum(((s.mu/r_norm.T**3)*cross(m,r).T).T,-2)/(4*pi)
        return A

    def calcBfield(s,pos):
        dipPos=array([dip[0] for dip in s.permDipoles])
        m     =array([dip[1] for dip in s.permDipoles])
        r,r_norm=calcAllDist(pos,dipPos); r_unit=(r.T/r_norm.T).T
        B=sum(((s.mu/r_norm.T**3)*(3*(r_unit.T*sum(m*r_unit,-1).T).T-m).T).T,-2)/(4*pi)
        # Source: Coey, JMD "Magnetism and magnetic materials" (Cambridge
        #         University Press), 2010. ISBN-13 978-0-511-67743-4
        #         [eq. 2.13]
        return B


    def calcFieldsInView(s,orig,u,v,uStep,vStep,uRange,vRange,calcForce=True):
        """Calculates the B, U, F and fluo fields within a given plane and field of view.
        Returns an object of class 'View'."""
        dipoles=s.permDipoles

        u=u/sum(u**2)**.5; v=v/sum(v**2)**.5; w=-cross(u,v)

        uvGrid_ZYX=orig+array([[uStep*u*i + vStep*v*j for j in r_[vRange[0]:vRange[1]]] for i in r_[uRange[0]:uRange[1]]])
        if calcForce: # Add positions in uvGrid for calculation of gradient
            dWVU=  0.01  *min(uStep,vStep)
            uvGrid_ZYX=moveaxis(array([
                    uvGrid_ZYX,           # Original positions
                    uvGrid_ZYX+w*dWVU,    # ... +dw
                    uvGrid_ZYX+v*dWVU,    # ... +dv
                    uvGrid_ZYX+u*dWVU,    # ... +du
                   ]),0,-2); # The extra dimension is the second-to-last axis

        uCoo=uStep*r_[uRange[0]:uRange[1]]; vCoo=vStep*r_[vRange[0]:vRange[1]]

        ## Calculate magnetic field over UV grid, expressed in XYZ coordinates
        Bpillar_ZYX   = s.calcBfield(uvGrid_ZYX)     # Magnetic field from all dipoles
        Bexternal_ZYX = s.Bext;                      # External magnetic field
        B_ZYX         = Bpillar_ZYX + Bexternal_ZYX  # Total magentic field

        # Norm and direction of magentic field
        Bnorm=(B_ZYX**2).sum(-1)**.5
        Bdir_ZYX=(B_ZYX.T/Bnorm.T).T

        # Magetic moment per particle
        m_ZYX = Bdir_ZYX*s.magSatMNP # (Note: Where B<100mT, we should include the whole magnetization curve)

        # Potential energy [Coey 2010, eq. 2.73]
        mDotB=Bnorm*s.magSatMNP; # or mDotB=sum(m_ZYX*B_ZYX,-1);
        U=-mDotB+s.magSatMNP*sum(s.Bext**2)**.5 # Offset so that U=0 at long distance
        fluo=exp((-U/s.kB_T).clip(-3e2,3e2)); # Predicted fluorescence from Boltzman distribution

        # Force
        if calcForce:
            F_WVU=-(U.T[1:]-U.T[0]).T/dWVU # Gradient of U [Coey 2010, eq. 2.74]
            Fnorm=(F_WVU**2).sum(-1)**.5
            Fdir_WVU=(F_WVU.T/Fnorm.T).T

        #------------

        # Coordinates of dipoles
        dipPos_ZYX=array([a[0]-orig for a in dipoles])

        # Remove positions used for calculation of gradient
        if calcForce: 
            uvGrid_ZYX  =uvGrid_ZYX.T[:,0].T
            Bpillar_ZYX =Bpillar_ZYX.T[:,0].T
            B_ZYX       =B_ZYX.T[:,0].T
            Bnorm       =Bnorm.T[0].T
            Bdir_ZYX    =Bdir_ZYX.T[:,0].T
            m_ZYX       =m_ZYX.T[:,0].T
            U           =U.T[0].T
            fluo        =fluo.T[0].T 

        ## Convert coordinates from ZYX to UVW
        B_WVU      = array([dot(B_ZYX     ,w).T,dot(B_ZYX     ,v).T,dot(B_ZYX     ,u).T]).T
        Bdir_WVU   = array([dot(Bdir_ZYX  ,w).T,dot(Bdir_ZYX  ,v).T,dot(Bdir_ZYX  ,u).T]).T
        m_WVU      = array([dot(m_ZYX     ,w).T,dot(m_ZYX     ,v).T,dot(m_ZYX     ,u).T]).T
        dipPos_WVU = array([dot(dipPos_ZYX,w).T,dot(dipPos_ZYX,v).T,dot(dipPos_ZYX,u).T]).T

        ## Create 'View' object.
        view=View(uCoo,vCoo,uStep,vStep,uRange,vRange,s.kB_T)
        view.B     =B_WVU
        view.Bnorm =Bnorm
        view.Bdir  =Bdir_WVU
        view.U     =U
        view.fluo  =fluo
        if calcForce:
            view.F     =F_WVU
            view.Fnorm =Fnorm
            view.Fdir  =Fdir_WVU
        else: view.F=None
        view.dipPos=dipPos_WVU

        return view

#==============================================#
# Display
#==============================================#

    def plotDipoleMap(s):
        fig=plt.figure(figsize=(4,4),dpi=72)
        tmp=array([a[0] for a in s.permDipoles])
        plt.scatter(tmp[:,2]*1e6,tmp[:,1]*1e6,marker='.',s=2,alpha=.1)
        plt.gca().set_aspect('equal', 'box'); plt.xlabel("x (µm)"); plt.ylabel("y (µm)")
        plt.show()
        fig=plt.figure(figsize=(4,4),dpi=72)
        tmp=array([a[0] for a in s.permDipoles])
        plt.scatter(tmp[:,2]*1e6,tmp[:,0]*1e6,marker='.',s=2, alpha=.1)
        plt.gca().set_aspect('equal', 'box'); plt.xlabel("x (µm)"); plt.ylabel("z (µm)")
        plt.show()

        print("Total number of dipoles:",len(s.permDipoles))


###### End of class 'Model' ##########################################
######################################################################



######################################################################
###### Class 'View' ##################################################

class View():
    """Class to store and display the different fields (B, U, F, ...) within a given plane and field of view."""

    def __init__(self,uCoo,vCoo,uStep,vStep,uRange,vRange,kB_T):
        #self.orig=orig
        #self.u=u
        #self.v=v
        #self.w=w
        self.uCoo=uCoo
        self.vCoo=vCoo
        self.uStep=uStep
        self.vStep=vStep
        self.uRange=uRange
        self.vRange=vRange
        self.kB_T=kB_T
        
        
#============================================================
    def show_B(s,rangeLogVal=[-1,0], showArrows=True,contours=[],showDipoleMap=True,fig=None,fileName=None,cmap=cm.jet):
        if fig==None: plt.figure(figsize=(8,6),dpi=72)
        #else: fig.add_subplot(221);

        plt.title("Magnetic field $\mathbf{B}$")

        if len(contours):
            plt.contour(s.uCoo*1e6,s.vCoo*1e6,s.Bnorm.T,levels=contours,colors='lightgray',linestyles='solid',linewidths=.8);

        #plt.pcolormesh(s.uCoo*1e6,s.vCoo*1e6,s.Bnorm.T,cmap=cm.jet,
        #               norm=cm.colors.LogNorm(vmin=10** ( -1.5 ) , vmax=10** ( .5 ) ),shading='gouraud');
        plt.pcolormesh(s.uCoo*1e6,s.vCoo*1e6,s.Bnorm.T,cmap=cmap,
                       norm=cm.colors.LogNorm(vmin=10**rangeLogVal[0], vmax=10**rangeLogVal[1]),
                       shading='gouraud');
        plt.colorbar(shrink=.5,label='Tesla')

        if showDipoleMap:
            plt.scatter(s.dipPos[:,2]*1e6,s.dipPos[:,1]*1e6,marker='.',s=20,alpha=1,c='k',zorder=100)

        if showArrows:
            plt.quiver(s.uCoo*1e6,s.vCoo*1e6,s.Bdir[:,:,2].T,s.Bdir[:,:,1].T,s.Bdir[:,:,0].T,
                       cmap=cm.bwr,scale=60,width=.05,headwidth=4,minshaft=2,minlength=.1,pivot='mid')
            plt.clim(-.8,.8); #plt.colorbar(fraction=0.04)

        plt.xlim(s.uRange[0]*s.uStep*1e6,(s.uRange[1]-1)*s.uStep*1e6);
        plt.ylim(s.vRange[0]*s.vStep*1e6,(s.vRange[1]-1)*s.vStep*1e6);
        plt.gca().set_aspect('equal', 'box'); plt.xlabel("µm"); plt.ylabel("µm")

        if fig==None:
            if fileName!=None: plt.savefig(fileName)
            else: plt.show()
                
#============================================================
    def show_U(s,rangeVal=[-5,0], showArrows=True,contours=[],showDipoleMap=True,fig=None,fileName=None,cmap=cm.pink_r):
        if fig==None: plt.figure(figsize=(8,6),dpi=72)
        #else: fig.add_subplot(222);
        
        plt.title("Potential energy $U=-\mathbf{m}\cdot\mathbf{B}$")

        if len(contours):
            plt.contour(s.uCoo*1e6,s.vCoo*1e6,s.U.T/s.kB_T,levels=contours,colors='lightgray',linestyles='solid',linewidths=.8);

        plt.pcolormesh(s.uCoo*1e6,s.vCoo*1e6,s.U.T/s.kB_T,cmap=cmap,vmin=rangeVal[0],vmax=rangeVal[1],shading='gouraud');
        plt.colorbar(shrink=.5,label='$k_BT$')

        if showDipoleMap:
            plt.scatter(s.dipPos[:,2]*1e6,s.dipPos[:,1]*1e6,marker='.',s=20,alpha=1,c='k',zorder=100)

        plt.xlim(s.uRange[0]*s.uStep*1e6,(s.uRange[1]-1)*s.uStep*1e6);
        plt.ylim(s.vRange[0]*s.vStep*1e6,(s.vRange[1]-1)*s.vStep*1e6);
        plt.gca().set_aspect('equal', 'box'); plt.xlabel("µm"); plt.ylabel("µm")
        
        if fig==None:
            if fileName!=None: plt.savefig(fileName)
            else: plt.show()

#============================================================
    def show_F(s,rangeLogVal=[-1.5,1], showArrows=True,contours=[],showDipoleMap=True,fig=None,fileName=None,cmap=cm.gist_heat):
        if fig==None: plt.figure(figsize=(8,6),dpi=72)
        #else: fig.add_subplot(223);
        
        plt.title("Force field $\mathbf{F}$")

        if len(contours):
            plt.contour(s.uCoo*1e6,s.vCoo*1e6,s.Fnorm.T/1e-15,levels=contours,
                        colors='lightgray',linestyles='solid',linewidths=.8);

        plt.pcolormesh(s.uCoo*1e6,s.vCoo*1e6,s.Fnorm.T/1e-15,cmap=cmap,
                       norm=cm.colors.LogNorm(vmin=10**rangeLogVal[0],vmax=10**rangeLogVal[1]),shading='gouraud');
        plt.colorbar(shrink=.5,label='fN / MNP')

        if showDipoleMap:
            plt.scatter(s.dipPos[:,2]*1e6,s.dipPos[:,1]*1e6,marker='.',s=20,alpha=1,c='k',zorder=100)

        if showArrows:
            plt.quiver(s.uCoo*1e6,s.vCoo*1e6,s.Fdir[:,:,2].T,s.Fdir[:,:,1].T,s.Fdir[:,:,0].T,
                       cmap=cm.bwr,scale=60,width=.05,headwidth=4,minshaft=2,minlength=.1,pivot='mid')
            plt.clim(-.8,.8); #plt.colorbar(fraction=0.04)

        plt.xlim(s.uRange[0]*s.uStep*1e6,(s.uRange[1]-1)*s.uStep*1e6);
        plt.ylim(s.vRange[0]*s.vStep*1e6,(s.vRange[1]-1)*s.vStep*1e6);
        plt.gca().set_aspect('equal', 'box'); plt.xlabel("µm"); plt.ylabel("µm")
        
        if fig==None:
            if fileName!=None: plt.savefig(fileName)
            else: plt.show()

#============================================================
    def show_fluo(s,rangeVal=[0,20], showArrows=True,contours=[],showDipoleMap=True,fig=None,fileName=None):
        if fig==None: plt.figure(figsize=(8,6),dpi=72)
        #else: fig.add_subplot(224);
        
        plt.title("Fluorescence")

        plt.pcolormesh(s.uCoo*1e6,s.vCoo*1e6,s.fluo.T,vmin=rangeVal[0],vmax=rangeVal[1],
                       cmap=cmap_BkGn,shading='gouraud');
        plt.colorbar(shrink=.5,label='a.u.')

        if showDipoleMap:
            plt.scatter(s.dipPos[:,2]*1e6,s.dipPos[:,1]*1e6,marker='.',s=20,alpha=1,c='k',zorder=100)

        plt.xlim(s.uRange[0]*s.uStep*1e6,(s.uRange[1]-1)*s.uStep*1e6);
        plt.ylim(s.vRange[0]*s.vStep*1e6,(s.vRange[1]-1)*s.vStep*1e6);
        plt.gca().set_aspect('equal', 'box'); plt.xlabel("µm"); plt.ylabel("µm")
        
        if fig==None:
            if fileName!=None: plt.savefig(fileName)
            else: plt.show()
                
#============================================================
    def show(s,showArrows=True,showDipoleMap=True,
             B_rangeLogVal =[-1,0],   B_contours=[],
             U_rangeVal    =[-5,0],   U_contours=[],
             F_rangeLogVal =[-1.5,1], F_contours=[],
             fluo_rangeVal =[0,20],
             fileName=None):

        fig=plt.figure(figsize=(16,12),dpi=72)

        fig.add_subplot(221);
        s.show_B(rangeLogVal=B_rangeLogVal,showArrows=showArrows,contours=B_contours,showDipoleMap=True,fig=fig)
        fig.add_subplot(222);
        s.show_U(rangeVal=U_rangeVal,      showArrows=showArrows,contours=U_contours,showDipoleMap=True,fig=fig)
        if type(s.F)!=type(None):
            fig.add_subplot(223);
            s.show_F(rangeLogVal=F_rangeLogVal,showArrows=showArrows,contours=F_contours,showDipoleMap=True,fig=fig)
        fig.add_subplot(224);
        s.show_fluo(rangeVal=fluo_rangeVal, showArrows=showArrows,showDipoleMap=True,fig=fig)

        if fileName!=None: plt.savefig(fileName)
        else: plt.show()


#============================================================
    def show_B_components(s,rangeVal=[-.5,.5],showDipoleMap=True,fileName=None):

        fig=plt.figure(figsize=(12,9),dpi=72)
        #============================================================
        fig.add_subplot(131); plt.title("Magnetic field $B_u$")

        plt.pcolormesh(s.uCoo*1e6,s.vCoo*1e6,s.B.T[2],cmap=cm.jet,vmin=rangeVal[0],vmax=rangeVal[1],shading='gouraud');
        plt.colorbar(shrink=.2,label='Tesla')

        if showDipoleMap:
            plt.scatter(s.dipPos[:,2]*1e6,s.dipPos[:,1]*1e6,marker='.',s=20,alpha=1,c='k',zorder=100)

        plt.xlim(s.uRange[0]*s.uStep*1e6,(s.uRange[1]-1)*s.uStep*1e6);
        plt.ylim(s.vRange[0]*s.vStep*1e6,(s.vRange[1]-1)*s.vStep*1e6);
        plt.gca().set_aspect('equal', 'box'); plt.xlabel("µm"); plt.ylabel("µm")
        #============================================================
        fig.add_subplot(132); plt.title("Magnetic field $B_v$")

        plt.pcolormesh(s.uCoo*1e6,s.vCoo*1e6,s.B.T[1],cmap=cm.jet,vmin=rangeVal[0],vmax=rangeVal[1],shading='gouraud');
        plt.colorbar(shrink=.2,label='Tesla')

        if showDipoleMap:
            plt.scatter(s.dipPos[:,2]*1e6,s.dipPos[:,1]*1e6,marker='.',s=20,alpha=1,c='k',zorder=100)

        plt.xlim(s.uRange[0]*s.uStep*1e6,(s.uRange[1]-1)*s.uStep*1e6);
        plt.ylim(s.vRange[0]*s.vStep*1e6,(s.vRange[1]-1)*s.vStep*1e6);
        plt.gca().set_aspect('equal', 'box'); plt.xlabel("µm"); plt.ylabel("µm")
        #============================================================
        fig.add_subplot(133); plt.title("Magnetic field $B_w$")

        plt.pcolormesh(s.uCoo*1e6,s.vCoo*1e6,s.B.T[0],cmap=cm.jet,vmin=rangeVal[0],vmax=rangeVal[1],shading='gouraud');
        plt.colorbar(shrink=.2,label='Tesla')

        if showDipoleMap:
            plt.scatter(s.dipPos[:,2]*1e6,s.dipPos[:,1]*1e6,marker='.',s=20,alpha=1,c='k',zorder=100)

        plt.xlim(s.uRange[0]*s.uStep*1e6,(s.uRange[1]-1)*s.uStep*1e6);
        plt.ylim(s.vRange[0]*s.vStep*1e6,(s.vRange[1]-1)*s.vStep*1e6);
        plt.gca().set_aspect('equal', 'box'); plt.xlabel("µm"); plt.ylabel("µm")
        #============================================================
        plt.tight_layout()

        if fileName!=None: plt.savefig(fileName)
        else: plt.show()


#============================================================
    def show_F_components(s,rangeVal=[-1,1],showDipoleMap=True,fileName=None):

        fig=plt.figure(figsize=(12,9),dpi=72)
        #============================================================
        fig.add_subplot(131); plt.title("Force field $F_u$")

        plt.pcolormesh(s.uCoo*1e6,s.vCoo*1e6,s.F.T[2]/1e-15,cmap=cmap_BlBkOr,
                       vmin=rangeVal[0],vmax=rangeVal[1],shading='gouraud');
        plt.colorbar(shrink=.2,label='fN per particle')

        if showDipoleMap:
            plt.scatter(s.dipPos[:,2]*1e6,s.dipPos[:,1]*1e6,marker='.',s=20,alpha=1,c='k',zorder=100)

        plt.xlim(s.uRange[0]*s.uStep*1e6,(s.uRange[1]-1)*s.uStep*1e6);
        plt.ylim(s.vRange[0]*s.vStep*1e6,(s.vRange[1]-1)*s.vStep*1e6);
        plt.gca().set_aspect('equal', 'box'); plt.xlabel("µm"); plt.ylabel("µm")
        #============================================================
        fig.add_subplot(132); plt.title("Force field $F_v$")

        plt.pcolormesh(s.uCoo*1e6,s.vCoo*1e6,s.F.T[1]/1e-15,cmap=cmap_BlBkOr,
                       vmin=rangeVal[0],vmax=rangeVal[1],shading='gouraud');
        plt.colorbar(shrink=.2,label='fN per particle')

        if showDipoleMap:
            plt.scatter(s.dipPos[:,2]*1e6,s.dipPos[:,1]*1e6,marker='.',s=20,alpha=1,c='k',zorder=100)

        plt.xlim(s.uRange[0]*s.uStep*1e6,(s.uRange[1]-1)*s.uStep*1e6);
        plt.ylim(s.vRange[0]*s.vStep*1e6,(s.vRange[1]-1)*s.vStep*1e6);
        plt.gca().set_aspect('equal', 'box'); plt.xlabel("µm"); plt.ylabel("µm")
        #============================================================
        fig.add_subplot(133); plt.title("Force field $F_w$")

        plt.pcolormesh(s.uCoo*1e6,s.vCoo*1e6,s.F.T[0]/1e-15,cmap=cmap_BlBkOr,
                       vmin=rangeVal[0],vmax=rangeVal[1],shading='gouraud');
        plt.colorbar(shrink=.2,label='fN per particle')

        if showDipoleMap:
            plt.scatter(s.dipPos[:,2]*1e6,s.dipPos[:,1]*1e6,marker='.',s=20,alpha=1,c='k',zorder=100)

        plt.xlim(s.uRange[0]*s.uStep*1e6,(s.uRange[1]-1)*s.uStep*1e6);
        plt.ylim(s.vRange[0]*s.vStep*1e6,(s.vRange[1]-1)*s.vStep*1e6);
        plt.gca().set_aspect('equal', 'box'); plt.xlabel("µm"); plt.ylabel("µm")
        #============================================================
        plt.tight_layout()

        if fileName!=None: plt.savefig(fileName)
        else: plt.show()        
        
###### End of class 'View' ###########################################
######################################################################




#==============================================#
# Low-level routines
#==============================================#
    
def calcAllDist(p1,p2):
    """Calculates all the vectors and distances between the all pairs
    of elements of two arrays p1 and p2 of arbitrary shapes, assuming
    that the last dimension of both arrays represents the the spatial
    coordinates (spatial coordinates an have arbitrary dimensions).
    Example in 3D:
      Input  => p1.shape=(i,j,k,3) p2.shape=(l,m,3)
      Output => vector.shape=(i,j,k,l,m,3) distance.shape=(i,j,k,l,m)"""
    r=((zeros(p1.shape[:-1]+p2.T.shape)+p2.T).T-p1.T).T
    r=moveaxis(r,r_[-len(p2.shape):0],r_[-len(p2.shape):0][::-1])
    r_norm=sum(r**2,-1)**.5
    return r, r_norm

    
#==============================================#
# Custom colormaps
#==============================================#
    
cmap_BkGnSat=cm.colors.LinearSegmentedColormap('BkGnSat',
                {'red':   [[0.,0.,0.],[.95,0.,.7],[1.,.7,.7]],
                 'green': [[0.,0.,0.],[.95,1.,.7],[1.,.7,.7]],
                 'blue':  [[0.,0.,0.],[.95,0.,.7],[1.,.7,.7]]})
cmap_BkGn=cm.colors.LinearSegmentedColormap('BkGnSat',
                {'red':   [[0.,0.,0.],[1.,0.,0.]],
                 'green': [[0.,0.,0.],[1.,1.,1.]],
                 'blue':  [[0.,0.,0.],[1.,0.,0.]]})
cmap_BlBkOr=cmap=cm.colors.LinearSegmentedColormap.from_list('BlBkOr',[[.5,.5,1],[0,0,1],[0,0,0],[1,.7,0],[1,.85,.5]])

