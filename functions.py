import numpy as np
import matplotlib.pyplot as plt

# ============ IMPORT DATA FROM .CSV FILE ============

def impt(path: str,skip: int=1):      #path is a file path in quotes, skip is the number of rows to skip at the top
    return np.transpose(np.genfromtxt(path, delimiter=',', skip_header=skip)) 

# ========= REMOVE NAN VALUES FROM AN ARRAY ==========

def nan(array):
    return array[~np.isnan(array)]

# ==== TAKE THE WEIGHTED MEAN OF A LIST OF VALUES ====

def weighted_mean(list):        #takes a list of ordered pairs of the form (mean, std), returns an ordered pair (mean, std)
    sigma2=0
    for i in list:
        sigma2 += 1/i[1]**2
    avg=0
    for i in list:
        avg += i[0]/i[1]**2
    avg = avg/sigma2
    return avg, 1/(sigma2)**0.5

# ================ QUICKLY PLOT DATA =================

def fastplot(xdata: None | list | np.typing.NDArray=None,
              ydata: None | list | np.typing.NDArray=None,
              xlab: str="x axis",
              ylab: str="y axis",
              xrange: None | tuple=None,
              yrange: None | tuple=None,
              Title: None | str=None,
              filename: str="plot",
              download: bool=False,
              datalabel: str="Data",
              legend: bool=True,
              grid: bool=False,
              yerr: None | list | np.typing.NDArray=None,
              index: int=1,
              figsize: tuple=(4,4),
              margin: float=0.1):
    if ydata is None:
       if xdata is None:
          raise TypeError('Please specify data to be plotted!')
       else: 
          ydata = xdata
          xdata = np.linspace(1, len(ydata), len(ydata), dtype=int)
    if xdata is None:
       xdata = np.linspace(1, len(ydata), len(ydata), dtype=int)
    figure = plt.figure(index,figsize=figsize)
    plt.errorbar(xdata, ydata, yerr=yerr, c="b", fmt="o", markersize=3, mfc="white", label=datalabel, capsize=2)
    if legend == True:
        plt.legend()
    if xrange == None:
        xrange = min(xdata) - (max(xdata) - min(xdata)) * margin, max(xdata) + (max(xdata)-min(xdata)) * margin
    if yrange == None:
        if yerr == None:
            miny = min(ydata)
            maxy = max(ydata)
        else:
            miny = min(np.array(ydata) - np.array(yerr))
            maxy = max(np.array(ydata) + np.array(yerr))
        yrange = miny - (maxy - miny) * margin, maxy + (maxy - miny) * margin
    ax = plt.gca()
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    plt.grid(grid)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if Title != None:
        plt.title(Title)
    if download == True:
        plt.savefig(filename, bbox_inches="tight")


