
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat, unumpy
import scipy.stats as stats
import uncertainties.umath
import scipy.constants

def linearFit(x, A, B):
    return np.multiply(A,x) + B

def zeroFunc(x):
    return [0]*(len(x))


class plotter:
    def __init__(self,files, X, Y, UncertX = -1, UncertY = -1, xLabel="x", yLabel="y",fileType="csv"):
        # X -> Col for independent variable
        # Y -> Col for dependent variable
        # UncertX -> Col for X uncertainty
        # UncertY -> Col for Y uncertainty
        self.files = files
        self.XCol = X
        self.YCol = Y
        self.UncertX = UncertX
        self.UncertY = UncertY
        self.XData = []
        self.YData = []
        self.XVal = []
        self.YVal = []
        self.Xerror = []
        self.Yerror = []

        self.pcov = []
        self.popt = []

        self.plt = plt

        
        self.xLabel = xLabel
        self.yLabel = yLabel


        if fileType.casefold() == "csv":
            self.split = ","
        elif fileType.casefold() == "tsv":
            self.split = "\t"
        elif fileType.casefold() == "raw":
            self.XVal = X
            self.Xerror = UncertX
            self.YVal = Y
            self.Yerror = UncertY

        else:
            self.split = " "



    def loadVals(self):
        for i in range(0,len(self.files)):
            f = open(dataFiles[i])
            lines = f.readlines()
            for line in lines:
                lineA = line.split('\n')
                RawData = lineA.split(self.split)

                self.XVal = self.XVal + [float((RawData[self.XCol]))]
                self.YVal = self.YVal + [float((RawData[self.YCol]))]


                if self.UncertX == -1:
                    self.XData =  self.XData + [ufloat(float(RawData[self.XCol]), 0.0)]
                    self.Xerror = self.Xerror + [0.0]
                else:
                    self.XData =  self.XData + [ufloat(float(RawData[self.XCol]), float(RawData[self.XUncert]))]
                    self.Xerror =  self.Xerror + [float(RawData[self.XUncert])]

                if self.YData == -1:
                    self.YData =  self.YData + [ufloat(float(RawData[self.YCol]), 0.0)]
                    self.Yerror = self.Yerror + [0.0]
                else:
                    self.YData =  [ufloat(float(RawData[self.YCol]), float(RawData[self.YUncert]))]
                    self.Yerror =  self.Yerror + [float(RawData[self.Yncert])]
        return 0






    def plotErrorBars(self,title = "X vs Y",color='red',errorbarColor='lightgray',format='.',capsize = 1):
        print(self.Xerror)
        print(self.Yerror)
        self.plt.errorbar(self.XVal,self.YVal,self.Yerror, self.Xerror,fmt =format, ecolor = errorbarColor,capsize = capsize)
        self.plt.xLabel = self.xLabel
        self.plt.yLabel = self.yLabel
        self.plt.title = title
        return 0

    def fitData(self, func = linearFit):

        popt, pcov = curve_fit(func, self.XVal, self.YVal)

        if len(popt) >= 1:
            d_N = [np.sqrt(pcov[i][i]) for i in range(0, len(popt),1) ]
        else:
            d_N = [ np.sqrt(pcov[0]) ]

        self.popt = popt
        self.pcov = pcov

        return d_N
    def graphFunc(self,func=linearFit,color='purple',legend="Fit"):
        print(self.popt)
        y = func(self.XVal,*self.popt )

        self.plt.plot(self.XVal,y,color=color)
        self.plt.legend(legend)

        return 0
    

    def show(self):
        self.plt.show()
        return 0;


X = range(0,10,1)
Y = range(1,21,2)

Error = (np.ones(10)*0.5)

pltr = plotter("",X,Y,Error,Error,xLabel = "x", yLabel= "y", fileType="raw")
pltr.plotErrorBars()
pltr.fitData()
pltr.graphFunc()
pltr.show()











# assume magnetic field is only pointing in the z direction
#2pi * f/ Bz = gyromagnetic ratio
# lande g factor = gamma/ (e/2m)

#Frequency = (1/2pi)*gamma*B


#dataFiles = ["Coil E.csv", "Coil F.csv", "Coil G.csv"]

#Current = []
#Frequency = []




#plot =  plotter(dataFiles,X=0,Y=2,UncertY=1 ,UncertX= 3,xLabel="current",yLabel="Frequency",fileType="csv") 
#plot.loadVals()
#plot.plotErrorBars()
#plot.plt.show()


#for i in range(0,3):
#    f = open(dataFiles[i])
#    lines = f.readlines()
#    for line in lines:
#        RawData = line.split(',')
#        print(RawData)
#        Current = Current + [ufloat(float(RawData[0]), float(RawData[1]))] #Current + uncertainty (in A)
#        Frequency = Frequency + [ufloat(float(RawData[2]),float(RawData[2].split('\n')[0]))*1000*1000] # Frequency in Hz
        
#Bfield = BFunc(Current)

##Gyro = GyroFunc(Frequency,Bfield)

#B_val = [val.n for val in Bfield]
#B_uncert = [val.s for val in Bfield]

#Current_val = [val.n for val in Current]
#Current_uncert = [val.s for val in Current]


#Freq_val = [val.n for val in Frequency]
#Freq_uncert = [val.s for val in Frequency]

##Gyro_val = [val.n for val in Gyro]
##Gyro_uncert = [val.s for val in Gyro]


##Plotting Frquency vs Current

#plt.errorbar(Current_val,Freq_val,Freq_uncert,Current_uncert,fmt='.',ecolor="lightgray",capsize=1)

#popt, pcov = curve_fit(curveFit,Current_val,Freq_val)
#d_A = np.sqrt(pcov[0])


#A = ufloat(popt,d_A)



#equation = 'Frequency = ' + str(A) + " * I" 
#plt.plot(Current_val,curveFit(Current_val,*popt),color="green",label = equation)
#plt.legend()
#plt.title(" Current vs  Frequency ")
#plt.xlabel("Current (A)")
#plt.ylabel("Frequency (Hz)")



#BSlopeFact = (math.pow(.8,1.5)*mu_0*n/R)
#FreqSlopeFact = (1/(2*math.pi))

#Gyro = np.divide(A,BSlopeFact*FreqSlopeFact)


#print("Gyromagnetic Ratio = " + str(Gyro))


#G_fact = Gyro/(e/(2*m_e))
#print("Lande g Factor = " + str(G_fact))

#print(G_fact/2.002)


#plt.show()


#yval = curveFit(Current_val,*popt)- Freq_val
#plt.plot(Current_val,zeroFunc(Current_val),color='red')
#plt.scatter(Current_val,yval,marker='d',color='blue')


#chi_0 = np.power(np.divide(yval,Freq_uncert),2)
#chisq = np.sum(chi_0)/(len(yval)-2) #One fit param
#print("Reduced Chisquared Val For Outer Circle is: " +str(chisq))


#plt.show()










    
