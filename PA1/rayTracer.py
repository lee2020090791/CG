#!/usr/bin/env python3
# -*- coding: utf-8 -*
# sample_python aims to allow seamless integration with lua.
# see examples below

import os
import sys
import pdb  # use pdb.set_trace() for debugging
import code # or use code.interact(local=dict(globals(), **locals()))  for debugging.
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image 
class Color:
    def __init__(self, R, G, B):
        self.color=np.array([R,G,B]).astype(np.float64)

    # Gamma corrects this color.
    # @param gamma the gamma value to use (2.2 is generally used).
    def gammaCorrect(self, gamma):
        inverseGamma = 1.0 / gamma
        self.color=np.power(self.color, inverseGamma)

    def toUINT8(self):
        return (np.clip(self.color, 0,1)*255).astype(np.uint8)
def normalize(x):
    x=x/np.sqrt(np.dot(x,x))
    return x
class Sphere:
    def __init__(self, center, radius,shader):
        self.center=center
        self.radius=radius
        self.shader=shader
    def intersection(self, ray, viewPoint):
        oc = viewPoint - self.center
        a=np.dot(ray,ray)
        b=-np.dot(oc, viewPoint)
        c=np.dot(oc,oc)-self.radius*self.radius
        D = b*b-a*c
        if D<0:
            return -1
        else :
            t1=(-b-np.sqrt(D))/a
            t2=(-b+np.sqrt(D))/a
            return min(t1,t2)
class Camera:
    def __init__(self, viewPoint,viewDir,projNormal,viewUp,projDistance,viewWidth,viewHeight,intensity):
        self.viewPoint=viewPoint
        self.viewDir=viewDir
        self.projNormal=projNormal
        self.viewUp=viewUp
        self.projDistance=projDistance
        self.viewWidth=viewWidth
        self.viewHeight=viewHeight
        self.intensity=intensity
    def getRay(self,img0,img1,ix,iy): #return ray direction
        pix_x=self.viewWidth/img0
        pix_y=self.viewHeight/img1
    
        w=self.viewDir
        u=np.cross(w,self.viewUp)
        v=np.cross(w, u)
        w=normalize(w)
        u=normalize(u)
        v=normalize(v)

        pix_00=w*self.projDistance-u*pix_x*((img0/2)+1/2)-v*pix_y*((img1/2)+1/2)
        start=pix_00+self.viewPoint
        end=self.viewPoint+w*pix_x*(img0)+v*pix_y*(img1)
        ray=pix_00+u*ix*pix_x+v*iy*pix_y
        return ray
        
class Light:
    def __init__(self, position, intensity):
        self.position=position
        self.intensity=intensity
class Shader:
    def __init__(self, type):
        self.type=type
class Phong(Shader):
    def __init__(self,diffuse,specular,exponent):
        self.diff=diffuse
        self.spec=specular
        self.expo=exponent
class Lambertian(Shader):
    def __init__(self,diffuse):
        self.diff=diffuse  
        
def rayTrace(sphere,ray,viewPoint):
    min_d=sys.maxsize #closet point
    id_c=-1 #closest index
    temp=0
    for i in sphere:
        oc = viewPoint-i.center
        a=np.dot(ray,ray)
        b=np.dot(oc, ray)
        c=np.dot(oc,oc)-(i.radius*i.radius)
        if b*b-a*c>=0:
            xp=(-b+np.sqrt(b*b-a*c))/a
            xm=(-b-np.sqrt(b*b-a*c))/a
            if -b+np.sqrt(b*b-a*c)>=0:
                if min_d>=xp:
                    min_d=xp
                    id_c=temp
            if -b-np.sqrt(b*b-a*c)>=0:
                if min_d>=xm:
                    min_d=xm
                    id_c=temp
        temp+=1
    return [min_d,id_c]

    
def shade(ray,cam,sphere,light,m,i):
    # arr=rayTrace(sphere,ray,cam.viewPoint)
    # m=arr[0]
    # i=arr[1].astype(np.int32)
    x,y,z=0,0,0
    v=-m*ray
    if(i==-1):
        return np.array([0,0,0])    
    n=np.array([0,0,0])
    fsh=sphere[i]#firstshader
    n=cam.viewPoint+m*ray-fsh.center #intersection point to center
    n=normalize(n)
    for temp in light:
        #reflected light
        l=v+temp.position-cam.viewPoint

        l=normalize(l)
        k=max(0,np.dot(n,l))
        tmp=rayTrace(sphere,-l,temp.position)
        
        if tmp[1] == i:
            if(fsh.shader.__class__.__name__=="Lambertian"):
                x=x+fsh.shader.diff[0]*k*temp.intensity[0]
                y=y+fsh.shader.diff[1]*k*temp.intensity[1]
                z=z+fsh.shader.diff[2]*k*temp.intensity[2]
            elif(fsh.shader.__class__.__name__=="Phong"):
                v=-m*ray
                v=normalize(m)
                h=v+l
                h=normalize(h)
                x=x+fsh.shader.diff[0]*k*temp.intensity[0]+fsh.shader.spec[0]*temp.intensity[0]*pow(max(0,np.dot(n,h)),fsh.shader.expo[0])
                y=y+fsh.shader.diff[1]*k*temp.intensity[1]+fsh.shader.spec[1]*temp.intensity[1]*pow(max(0,np.dot(n,h)),fsh.shader.expo[0])
                z=z+fsh.shader.diff[2]*k*temp.intensity[2]+fsh.shader.spec[2]*temp.intensity[2]*pow(max(0,np.dot(n,h)),fsh.shader.expo[0])
    res=Color(x,y,z)
    res.gammaCorrect(2.2)
    return res.toUINT8()
        
            
            
def main():
    tree = ET.parse(sys.argv[1])
    root = tree.getroot()

    # set default values
    viewDir=np.array([0,0,-1]).astype(np.float64)
    viewUp=np.array([0,1,0]).astype(np.float64)
    viewProjNormal=-1*viewDir  # you can safely assume this. (no examples will use shifted perspective camera)
    viewWidth=1.0
    viewHeight=1.0
    projDistance=1.0
    position=np.array([0,0,0]).astype(np.float64) # light position
    intensity=np.array([1,1,1]).astype(np.float64)  # how bright the light is.

    imgSize=np.array(root.findtext('image').split()).astype(np.int32)
    
    for c in root.findall('camera'):
        viewPoint=np.array(c.findtext('viewPoint').split()).astype(np.float64)
        viewDir=np.array(c.findtext('viewDir').split()).astype(np.float64)
        viewUp=np.array(c.findtext('viewUp').split()).astype(np.float64)
        viewProjNormal=-1*viewDir
        viewWidth=float(c.findtext('viewWidth'))
        viewHeight=float(c.findtext('viewHeight'))
        if(c.findtext('projDistance')):
            projDistance=np.array(c.findtext('projDistance').split()).astype(np.float64)        
    
        
    #code.interact(local=dict(globals(), **locals()))  

    # Create an empty image
    channels=3
    img = np.zeros((imgSize[1], imgSize[0], channels), dtype=np.uint8)
    img[:,:]=0
    # replace the code block below!
    # camera
    cam = Camera(viewPoint,viewDir,viewProjNormal,viewUp,projDistance,viewWidth,viewHeight,intensity)
    sphere=[] #sphere list
    light=[] #light list 
    # get surface
    for c in root.findall('surface'):
        if(c.get('type')=='Sphere'):
            center = np.array(c.findtext('center').split()).astype(np.float64)
            radius = np.array(c.findtext('radius')).astype(np.float64)
            ref=''
            for child in c:
                if child.tag=='shader':
                    ref=child.get('ref')
            for s in root.findall('shader'):
                if s.get('name')==ref:
                    diffuse_s = np.array(s.findtext('diffuseColor').split()).astype(np.float64)
                    type_s=s.get('type')
                    if type_s == 'Lambertian':
                        shader = Lambertian(diffuse_s)
                        sphere.append(Sphere(center,radius,shader))
                    elif type_s == 'Phong':
                        specular=np.array(s.findtext('specularColor').split()).astype(np.float64)
                        exponent=np.array(s.findtext('exponent').split()).astype(np.float64)
                        shader = Phong(diffuse_s,specular,exponent)
                        sphere.append(Sphere(center,radius,shader))
    # get light
    for c in root.findall('light'):
        position=np.array(c.findtext('position').split()).astype(np.float64)
        intensity=np.array(c.findtext('intensity').split()).astype(np.float64)
        light.append(Light(position,intensity))
    # run raytrace    
    for x in np.arange(imgSize[0]):
        for y in np.arange(imgSize[1]):
            ray=cam.getRay(imgSize[0],imgSize[1],x,y)
            arr=rayTrace(sphere,ray,cam.viewPoint)
            img[y][x]=shade(ray,cam,sphere,light,arr[0],arr[1])
                
    rawimg = Image.fromarray(img, 'RGB')
    #rawimg.save('out.png')
    rawimg.save(sys.argv[1]+'.png')
    
if __name__=="__main__":
    main()
