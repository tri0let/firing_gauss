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

def fastplot(data,
             xlab = "x axis",
             ylab = "y axis",
             xrange = None,
             yrange = None,
             Title = None,
             filename = "plot",
             download = False,
             datalabel = "Data",
             legend = True,
             yerr = None,
             index = 1,
             figsize = (4,4),
             margin = 0.05,
             x = 0,
             y = 1):
    plt.figure(index,figsize=figsize)
    plt.errorbar(data[x], data[y], yerr=yerr, c="b", fmt="o", markersize=3, mfc="white", label=datalabel, capsize=2)
    if legend == True:
        plt.legend()
    if xrange == None:
        xrange = min(data[x]), max(data[x])
    else:
        xrange = min(data[x]) - (max(data[x]) - min(data[x])) * margin, max(data[x]) + (max(data[x])-min(data[x])) * margin
    if yrange == None:
        yrange = min(data[y]), max(data[y])
    else:
        yrange = min(data[y]) - (max(data[y]) - min(data[y])) * margin, max(data[y]) + (max(data[y]) - min(data[y])) * margin
    ax = plt.gca()
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if Title != None:
        plt.title(Title)
    if download == True:
        plt.savefig(filename, bbox_inches="tight")
        # files.download(f'{filename}.png')