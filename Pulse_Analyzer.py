from logging import error
from tkinter import font
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import trapz, simps
import os
from tslearn.clustering import TimeSeriesKMeans
import h5py
import similaritymeasures

#uncomment data files that you want to analyze. Uncomment one to get the full analysis. Uncomment 2 or  more to compare multiple data sets
datafiles = [ 
    'ExamplePulseData.csv',
    #'ExamplePulseData2.csv',
     ] 

#####Code Engine Below here#####
for q in range(len(datafiles)):
    run = datafiles[q]
    df = pd.read_csv(run)
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(df)
    x=df['X']
    y_5=df['Y'].rolling(5).mean() #smoth the curve with a moving average of five sample points
    print(y_5)
    y_5 = y_5.div(60) #convert
    #print(y_5)
    ##find peaks
    peaks, peak_props = find_peaks(y_5, prominence=5, width=10, height=0.1, distance = 10) #Adjust prominence as needed to find your peaks. Remember to adjust one below too!

    if len(datafiles) == 1:

        plt.subplot(4, 1, 1) 
        plt.plot(x, df['Y'], color='blue') #plot peak points
        plt.title(run + ' Raw Data')
        plt.xlabel('X')
        plt.ylabel('Y') 
        
        plt.subplot(4, 1, 2) 
        plt.plot(x, y_5, color='blue') #plot peak points
        plt.title(run + ' 5 sample rolling mean')
        plt.xlabel('X')
        plt.ylabel('Y') 
    
        
        
        y_width_heights = peak_props['width_heights'].round()
        x_min = peak_props["left_ips"].round()
        x_max = peak_props["right_ips"].round()
        plt.subplot(4, 1, 3) 
        plt.plot(x[peaks], y_5[peaks], "x", color='C1') #plot peak points
        plt.vlines(x=x[peaks], ymin=y_5[peaks] - peak_props["prominences"], ymax = y_5[peaks], color = "C1") #plot peak prominences (heights)
        plt.hlines(y=y_width_heights, xmin=x[x_min], xmax=x[x_max], color = "C1") #plot half-peak widths
        plt.plot(x, y_5, color = 'blue') #plot y_5

    #Find bottom peak 
    ind_curves = []
    slicepeak = [] 
    ind_curve = []
    ind_x = []
    ind_xs = []
    for k in range(1,len(peak_props["right_ips"])-1): #this part finds the bottom of each curve by determining where the slope changes from going down to up!
        i = peak_props["right_ips"][k].round()
        j = peak_props["left_ips"][k].round()
        while y_5[i] > y_5[i + 20]: #dont just get the first value where the slope changes from going down to up, look ahead this many points (adjust as needed)
            i = i+1
        #print('i:', i)
        while y_5[j] > y_5[j - 20]: #dont just get the first value where the slope changes from going down to up, look behind this many points (adjust as needed)
            j = j-1
        #print('j:', j)

        left = j-1
        right = i+1
        slicepeak = y_5[(y_5.index >= left) & (y_5.index <= right)]
        slice_x = x[(y_5.index >= left) & (y_5.index <= right)]
        ind_curve = y_5[(y_5.index >= left) & (y_5.index <= right)]
        ind_x = x[(y_5.index >= left) & (y_5.index <= right)]
        ind_curve.index = (ind_curve.index - ind_curve.index[0])
        ind_x.index = (ind_x.index - ind_x.index[0])
        ind_x = (ind_x - ind_x[0])

        ind_curves.append(ind_curve)
        ind_xs.append(ind_x)
        
        if len(datafiles) == 1:
                plt.subplot(4, 1, 3) 
                plt.plot(slice_x, slicepeak, color = 'red') #plot y_5
                plt.title(run + ' Peak Finder')
                plt.xlabel('X')
                plt.ylabel('Y') 
    if len(datafiles) == 1:
        plt.subplot(4, 1, 3) 
        
    
    ind_curves = pd.DataFrame(ind_curves).transpose()
    ind_xs = pd.DataFrame(ind_xs).transpose()

    file = list(range(0,len(ind_curves.columns)))
    file_numbs_x = list(range(0,len(ind_xs.columns)))
    ind_curves.columns = file
    ind_xs.columns = file_numbs_x

    pd.set_option('display.max_columns', 100)  # or 1000
    pd.set_option('display.max_rows', 100)  # or 1000

    color = 'C' + str(q+1)
    #print(color)
    if len(datafiles) == 1: 
        plt.subplot(4, 1, 4) 
    plt.plot(ind_xs, ind_curves, color = 'grey', label='_Hidden', alpha =0.1,)



    #clustering
    ind_xs=ind_xs.values
    ind_xs=np.transpose(ind_xs)
    ind_curves=ind_curves.values
    ind_curves=np.transpose(ind_curves)
    km_sdtw_y = TimeSeriesKMeans(n_clusters=1, metric="softdtw", max_iter=30, max_iter_barycenter=30, random_state=0, metric_params={"gamma": 0.5}).fit(ind_curves)
    km_sdtw_x = TimeSeriesKMeans(n_clusters=1, metric="softdtw", max_iter=30, max_iter_barycenter=30, random_state=0, metric_params={"gamma": 0.5}).fit(ind_xs)
    DTWcurve = pd.DataFrame(km_sdtw_y.cluster_centers_[0]) #convert to dataframe
    DTWcurve[1] = pd.DataFrame(km_sdtw_x.cluster_centers_[0]) #convert to dataframe

    #find true mean curve
    minflow = DTWcurve[0].max()*0.12 #correction factor applied to find end of curve. Adjust as needed
    DTWcurve = DTWcurve[DTWcurve[0] > minflow].dropna()   

    #print(peak_props_avg)
    peaks_avg, peak_props_avg = find_peaks(DTWcurve[0], prominence=5, width=10, height=0.1, distance = 10, rel_height=1,) #dont forget to adjust this prominence to the same value as above!         
    y_width_heights_avg = int(peak_props_avg['width_heights'].round())
    x_min_avg = int(peak_props_avg["left_ips"].round())
    x_max_avg = int(peak_props_avg["right_ips"].round())
    c = km_sdtw_x.cluster_centers_[0]

    curve_dist = 'NA: Method only works with two curves datafiles.'
    if len(datafiles) == 2:  #calcualte distance between two curves if only two are being analyzed                    
        print('q equals: ' + str(q))                
        if q == 0:
            curvecomp_A = DTWcurve.to_numpy()
        if q == 1:
            curvecomp_B = DTWcurve.to_numpy()
            curve_dist = similaritymeasures.frechet_dist(curvecomp_A, curvecomp_B)
            curve_dist= round(curve_dist, 4)

    peak_width = DTWcurve[1].iloc[-1] - DTWcurve[1].iloc[0]
    mean_peak_width ="{:.3f}".format(peak_width)
    mean_max_y = "{:.3f}".format(DTWcurve[0][peaks_avg[0]])
    peak_x_position = "{:.3f}".format(DTWcurve[1][peaks_avg[0]])
    area_under_curve = "{:.3f}".format(trapz(DTWcurve[0], DTWcurve[1]))


    print(run + " mean peak width: " + str(peak_width) + " x")
    print(run + " mean max y: " + str(DTWcurve[0][peaks_avg[0]]) +" y")
    print(run + " peak x position: " + str(DTWcurve[1][peaks_avg[0]]) +" x")
    print(run + " predicted area under curve: " + str(area_under_curve))
    print("Discrete Frechet Distance: " + str(curve_dist))


    #Plot peak and width if only looking at one curve
    if len(datafiles) == 1:

        plt.subplot(4, 1, 4) 
        plt.title('Dynamic Time Warping k-means')
        plt.plot(DTWcurve[1][peaks_avg], DTWcurve[0][peaks_avg], "x", color='C1') #plot peak points
        plt.vlines(x=c[peaks_avg], ymin=DTWcurve[0][peaks_avg] - peak_props_avg["prominences"], ymax = DTWcurve[0][peaks_avg], color = "C1") #plot peak prominences (heights)
        plt.hlines(y=y_width_heights_avg, xmin=c[x_min_avg], xmax=c[x_max_avg], color = "C1") #plot half-peak widths
        plt.plot(km_sdtw_x.cluster_centers_[0], km_sdtw_y.cluster_centers_[0], color = 'blue') #plot km_sdtw_y
        plt.tight_layout(pad=0.1)
        plt.show()



    if len(datafiles) > 1:
        f = plt.plot(DTWcurve[1],DTWcurve[0], color = color, label=run)



if len(datafiles) > 1:
    plt.title('Dynamic Time Warping k-means')
    plt.xlabel('X')
    plt.ylabel('Y')    
    plt.legend()

    f = plt.show()
    
