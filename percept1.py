# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:20:00 2018

@author: asus
"""
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import glob
import math
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import glob
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
#matplotlib.use('TkAgg')
dt=""
def setinter():
    global dt
    interface=tk.Tk()
    interface.title('單層感知機')
    interface.geometry('1100x1000')
    def selectfile():
        global dt
        file=tk.filedialog.askopenfilename()
        file=open(file,'r')
        data=file.read()
       # print(data)
        dt=data.split('\n')
        dt=[i.split(' ')for i in dt]
        dt=np.array(dt)
        
        return dt
    FileButton=tk.Button(interface,text="fileselect",command=selectfile)
    FileButton.grid(row=0,sticky=W)
    def _initial_():
        weight=[-1,0,1]
        outy=-1
        return weight,outy
    
    epoch=1
    LR=0.5
    
    LRlabel=tk.Label(interface,text="Learnig Rate",bg='red',font=('Arial',12),width=15,height=2)
    LRlabel.grid(row=1,sticky=W)
    LRentry=tk.Entry(interface)
    LRentry.grid(row=2,sticky=W)
    #######################################Learning rate
    EPlabel=tk.Label(interface,text="Epoch",bg='red',font=('Arial',12),width=15,height=2)
    EPlabel.grid(row=3,sticky=W)
    Epochentry=tk.Entry(interface)
    Epochentry.grid(row=4,sticky=W)
   
    # plt.plot(x1,x2,color='green',linewidth=10,linestyle='--')
    def prt():
        global dt
        x1,x2,d=splitdata(dt)
        epoch=1#initial
        LR=str(LRentry.get())
        LR=float(LR)
        epoch=int(Epochentry.get())
        train(epoch,LR)
        
    printbt=tk.Button(interface,text="train",command=prt)
    printbt.grid(row=5,sticky=W)
   
    
     
    
    def splitdata(data):#dt
        x1=[]
        x2=[]
        d=[]
        for i in range(len(data)):#shape=row
            if(len(data[i][0])>0):
                x1.append(float(data[i][0]))
                x2.append(float(data[i][1]))
                if(int(data[i][2])!=1):
                    d.append(0)
                else:
                    d.append(1)

        return x1,x2,d
    
    def sgn(y):
       
        if y>=0:
             return 1
        elif y<0:
            return 0
    def adjustweight(x1,x2,d,outy,LR,wt):
        bias=-1
        rt=np.zeros(3,dtype=float)
        rt=np.array(rt)
        if(outy==d):
            rt=wt
            return rt
        elif(outy==0):
            rt=(wt[0]+LR*bias,wt[1]+LR*x1,wt[2]+LR*x2)
        else:
            rt=(wt[0]-LR*bias,wt[1]-LR*x1,wt[2]-LR*x2)
        return rt
    
    
    def train(epoch,LR):
        global dt
        bias=-1
       
        x1=[1,2]
        x1,x2,d=splitdata(dt)
        XMAX=-1000
        XMIN=1000
        YMAX=-100
        YMIN=100
        for i in range(len(x1)):
            XMAX=max(XMAX,x1[i])
            XMIN=min(XMIN,x1[i])
        for i in range(len(x2)):
            YMAX=max(YMAX,x2[i])
            YMIN=min(YMIN,x2[i])
        testindex=[]
        trainindex=[]
        traindata=[]
        testdata=[]
        d1tr=[]
        d0tr=[]
        d1ts=[]
        d0ts=[]
        
        if(len(x1)>0):
            trainindex=np.random.choice(len(x1),size=int(len(x1)*2/3)+1,replace=False)
            
        for i in range(len(x1)):
            testindex.append(i)
        for i in range(len(trainindex)):
            trx=[x1[trainindex[i]],x2[trainindex[i]]]
            traindata.append(trx)#####################traindata
            if(d[trainindex[i]]==0):
                d0tr.append(trx)
            elif(d[trainindex[i]]==1):
                d1tr.append(trx)
            trx=[]
        #traindatashow
        testindex=set(testindex)-set(trainindex)
        testindex=list(testindex)
        for i in range(len(testindex)):
            tsx=[x1[testindex[i]],x2[testindex[i]]]
            testdata.append(tsx)
            if(d[testindex[i]]==1):
                d1ts.append(tsx)
            elif(d[testindex[i]]==0):
                d0ts.append(tsx)
            tsx=[]

        weight,outy=_initial_()
        errortr=0.0
        ######################################training
        for j in range(epoch):
                
            errortr=0.0
            for i in range(len(trainindex)):
                outy=0
                outy+=bias*float(weight[0])+float(x1[trainindex[i]])*float(weight[1])+float(x2[trainindex[i]])*float(weight[2])
                outy=sgn(outy) 
               
                if(outy!=int(d[trainindex[i]])): 
                    #
                    errortr+=1
                    weight=adjustweight(x1[trainindex[i]],x2[trainindex[i]],d[trainindex[i]],outy,LR,weight)
        #print(weight)            
        errortr=(errortr/len(trainindex))*100
        errortr=100-errortr
        trerr="the training accuracy:"+str(errortr)+"%"
        errorts=0.0
        for i in range(len(testindex)):
            ty=float(bias*float(weight[0]))+float(x1[testindex[i]]*float(weight[1]))+float(x2[testindex[i]]*float(weight[2]))
            ty=sgn(ty)
#                print(d[testindex[i]],outy)
            if(ty==int(d[testindex[i]])):
                errorts=errorts+1
        errorts=(errorts/len(testindex))
        errorts*=100
            
        tserr="the testing accuracy:"+str(errorts)+"%"
            
        TRERLBL=tk.Label(interface,text=trerr,bg='white',font=('Arial',12),width=50,height=2)
        TRERLBL.grid(row=1,column=2)
        TSERLBL=tk.Label(interface,text=tserr,bg='white',font=('Arial',12),width=50,height=2)
        TSERLBL.grid(row=2,column=2)
        print("weight")
        print(weight)
            
        def draw():
           try:
               f =Figure(figsize=(10,10), dpi=50)
               a=f.add_subplot(111)
               
               canvas =FigureCanvasTkAgg(f, master=interface)
               #canvas.show()
               if(len(d0ts)>=1):
                   for i in range(len(d0ts)):
                       a.plot(d0ts[i][0],d0ts[i][1],'bx')
               if(len(d1ts)>=1):
                   for i in range(len(d1ts)):
                       a.plot(d1ts[i][0],d1ts[i][1],'mx')
               
              
               if(len(d0tr)>=1):
                   for i in range(len(d0tr)):
                       a.plot(d0tr[i][0],d0tr[i][1],'go')
               if(len(d1tr)>=1):
                   for i in range(len(d1tr)):
                       a.plot(d1tr[i][0],d1tr[i][1],'ro')
               
               if(weight[2]!=0.000000):
                   x=np.linspace(int(XMIN)-1,int(XMAX)+1,100)
                   y=-float((weight[1]/weight[2]))*x-float(weight[0]/weight[2])
                   a.plot(x,y)
               elif(weight[1]!=0.00000):
                   y=np.linspace(int(YMIN)-1,int(YMAX)+1,100)
                   x=-weight[0]+0*y
                   a.plot(x,y)
               else:
                   noline="無法畫線"
                   print(noline)
               Wstr="weight ["+str(round(weight[0],3))+"] ["+str(round(weight[1],3))+"] ["+str(round(weight[2],3))+"]"
               WLBL=tk.Label(interface,text=Wstr,bg='white',font=('Arial',12),width=50,height=2)
               WLBL.grid(row=3,column=2)
               
               canvas.get_tk_widget().grid(row=6,sticky=E)
               canvas._tkcanvas.grid(row=6,sticky=E)
            
           except Exception as e:
               print(e)
        draw() 
        
    
########################################################################################################   
   
    interface.mainloop()
setinter()



