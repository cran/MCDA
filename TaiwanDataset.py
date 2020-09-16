

############################################################################################
############################## Taiwan dataset #############################################

import pandas as pd
import random  
import sys 
#sys.path.append("E:/Google Drive/PC SHIT/HMMY/Diplomatiki/methods/MCDA")
from  Pyth.UTASTAR import *


# UTASTAR EXAMPLE
# the separation threshold

epsilon = 0.1



df = pd.read_excel(
    r"C:\Users\amichail\OneDrive - Raycap\Dokumente\Thes\Taiwan Dataset\default of credit card clients.xls",
    sheet_name="Data",
    #sheet_name="multimatrix",
)

#pre prossesing 
nrows=2503
df.columns =  df.iloc[2].values
df=df.iloc[3:nrows]
# Set index name
df.columns.name = "Alternatives"


# Reset index
df = df.reset_index(drop=True)
df = df.drop(["ID"],axis=1)
# Convert all dataframe values to integer
df = df.apply(pd.to_numeric)

#Visualization 



# Change qualitative values to quantitative

df["EDUCATION"].replace("5", "0", regex=True, inplace=True)
df["EDUCATION"].replace("6", "0", regex=True, inplace=True)
df["EDUCATION"].replace("4", "0", regex=True, inplace=True)
df["EDUCATION"].replace("2", "4", regex=True, inplace=True)
df["EDUCATION"].replace("3", "2", regex=True, inplace=True)
df["EDUCATION"].replace("4", "3", regex=True, inplace=True)

df["MARRIAGE"].replace("0", "3", regex=True, inplace=True)


df["PAY_0"].replace("-2", "0", regex=True, inplace=True)
df["PAY_0"].replace("-1", "0", regex=True, inplace=True)

df["PAY_2"].replace("-2", "0", regex=True, inplace=True)
df["PAY_2"].replace("-1", "0", regex=True, inplace=True)

df["PAY_3"].replace("-2", "0", regex=True, inplace=True)
df["PAY_3"].replace("-1", "0", regex=True, inplace=True)

df["PAY_4"].replace("-2", "0", regex=True, inplace=True)
df["PAY_4"].replace("-1", "0", regex=True, inplace=True)

df["PAY_5"].replace("-2", "0", regex=True, inplace=True)
df["PAY_5"].replace("-1", "0", regex=True, inplace=True)

df["PAY_6"].replace("-2", "0", regex=True, inplace=True)
df["PAY_6"].replace("-1", "0", regex=True, inplace=True)

df.describe()



# the performance table
# removing the class column
performanceTable = df.iloc[:,:-1] # -3
print("\nPerformance Table \n", performanceTable)

# ranks of the alternatives

data = df.iloc[:,-1:].values.flatten()
rownames = performanceTable.index.values
alternativesRanks = pd.DataFrame([data], columns=rownames)
print("\nAlternative Ranks\n", alternativesRanks)


minmaxdata = ["max","max","max","min", "min","min","min","min","min","min","min","max","max", "max","max","max","max","max","max","max","max","max","max"]

columnnames = performanceTable.columns.values
criteriaMinMax = pd.DataFrame([minmaxdata], columns=columnnames)
print("\nCriteria Min Max \n", criteriaMinMax,) 


bpdata = [4,2,3,2,4, 4,4,4, 4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4]

# number of break points for each criterion
criteriaNumberOfBreakPoints = pd.DataFrame(
    [bpdata],
    columns=performanceTable.columns.values,
)


##################### UTASTAR CREDIT SCORE #########################
  

(
    optimum,
    valueFunctions,
    overallValues,
    outRanks,
    errorValuesPlus,
    errorValuesMinus,
    tau,
    minWeights,
    maxWeights,
    averageValueFunctions,
    averageOverallValues,
) = UTASTAR(
    performanceTable,
    criteriaMinMax,
    criteriaNumberOfBreakPoints,
    epsilon,
    alternativesRanks=alternativesRanks,
)

print(
    optimum,  
    valueFunctions,
    overallValues,
    outRanks,
    errorValuesPlus,
    errorValuesMinus,
    tau,
    minWeights,
    maxWeights,
    averageValueFunctions,
    averageOverallValues,
    sep="\n",
) 

print("End")

# Insert to multi criteria matrix
df.insert(24, "OutRanks", outRanks.transpose())
df.insert(25, "OverallValues", overallValues.transpose())
""" 
df = df.sort_values(by=['OutRanks'])

#cooking
df = df.sort_values(by=['Class'])
dfte = df.sort_values(by=['OverallValues'], ascending = False)
df = df.iloc[:,:-2]
df.insert(18, "OverallValues", dfte.iloc[:,-2:-1].values)
df.insert(19, "OutRanks", dfte.iloc[:,-1:].values)

#Accurancy true postives + false postives 
nrows= df.shape[0]
y=df.columns.get_loc("Class")
y2=df.columns.get_loc("OutRanks")
accur = [1 for x in range(1, df.shape[0]) if (df.iloc[x,y2]<301 and df.iloc[x,y] == 1)   ]
accur2 = [1 for x in range(1, df.shape[0]) if (df.iloc[x,y2]>302 and df.iloc[x,y]==2) ]
 """
# Insert valueFunctions to criteria/columns
data = [valueFunctions.iloc[x, (bpdata[x//2]-1)] for x in range(1, len(valueFunctions), 2) ]
data = pd.DataFrame(data, index=performanceTable.columns.values, columns=["ValueFunc"])
valuefunc = data.transpose()

# TODO Replace -3 and -2 with more adjustable code 
#distribute overall values to all dataframe based on valueFunc
nrows=df.shape[0]
ncols= performanceTable.shape[1]
for i in range(0,nrows):
    df.iloc[i,0:ncols] = df.iloc[i,-2] * valuefunc.values.flatten()

#Append valuefunc row 
df = pd.concat([valuefunc, df], ignore_index=False)

utastarvaluefun=valuefunc

print(df)
df.to_excel(
    r"C:\Users\amichail\OneDrive - Raycap\Dokumente\Thes\Taiwan Dataset\UtastarResults.xls"
)

import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.image import imread
import seaborn as sns
from sklearn.datasets.samples_generator import (make_blobs,
                                                make_circles,
                                                make_moons)
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from warnings import filterwarnings
from math import pi
import pandas
from pandas.plotting import parallel_coordinates



##kmeans feautre reduction 
#data

dfplot = df
 
#%matplotlib inline
sns.set_context('notebook')
plt.style.use('fivethirtyeight')
filterwarnings('ignore')


# Plot the data
plt.figure(figsize=(6, 6))
plt.scatter(dfplot.iloc[:, 1], dfplot.iloc[:,2],)
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of  data')
 

dfplot= performanceTable

# Plot the data
plt.figure(figsize=(6, 6))
plt.scatter(dfplot.iloc[:, 1], dfplot.iloc[:,2],)
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of  data')
 
# matrix to be used for k menas

################################################
###########parralel coordinate plot ############
#################################################
dfplot = df
pd.DataFrame(dfplot).plot()
plt.show

# matrix to be used for k menas

###########parralel coordinate plot ############
dfplot = performanceTable
pd.DataFrame(dfplot).plot()
plt.show

########################################
######### RADAR GRAPH  ##################
#########################################
dfplot = df
#dfplot = performanceTable

categories=list(dfplot)
N = len(categories)

values=dfplot.loc[0].values.flatten().tolist()
values += values[:1]
values#=dfplot.loc[1].values.flatten().tolist()
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels original data
#ax.set_rlabel_position(0)
#plt.yticks([5,10,15], ["5","10","15"], color="grey", size=7)
#plt.ylim(0,20) # 20 for perfoncatable 0.05 for df

# Draw ylabels credit score data 
ax.set_rlabel_position(0)
plt.yticks([0.01,0.02,0.03], ["0.01","0.02","0.03"], color="grey", size=7)
plt.ylim(0,0.05) # 2

# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

##############################################
############# Parallel plot with pandas ######
###############################################

# Make the plot
#with original values 
data = performanceTable.iloc[:,:10].transpose() # first five atrributes
data.insert(0,"Class",data.index)

parallel_coordinates(data, "Class", colormap=plt.get_cmap("Set2") )
plt.show()
### second half 
data = performanceTable.iloc[:,10:]
data.insert(0,"Class",data.index)

parallel_coordinates(data, "Class", colormap=plt.get_cmap("Set2") )
plt.show()

######## with credit score ########
dfplot = df

data = dfplot.iloc[:,:10] #.transpose()  ## first half attributes
data.insert(0,"Class",data.index)

parallel_coordinates(data, "Class", colormap=plt.get_cmap("Set2") )
plt.show()

###
data = dfplot.iloc[:,10:]  ##second half attributes
data.insert(0,"Class",data.index)

parallel_coordinates(data, "Class" , colormap=plt.get_cmap("Set2"))
plt.show()



#############################################
######## Histograms with pandas##### pair of 2
############################################## 
dfplot = df

dfplot.hist(figsize=[25,15])
plt.tight_layout()

dfplot.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=10, ylabelsize=10, grid=True )
plt.tight_layout()
plt.tight_layout(rect=(1, 1, 0, 0))

performanceTable.hist(figsize=[20,10])
plt.tight_layout()

performanceTable.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=10, ylabelsize=10, grid=False )
plt.tight_layout()
plt.tight_layout(rect=(0, 0, 5, 5))

####################################
###### Bar Plot######### per feaure 
#####################################

for i in range(0,performanceTable.shape[1]-1):
            fig = plt.figure(figsize = (6, 4))
            title = fig.suptitle(performanceTable.iloc[:,i].name, fontsize=14)
            fig.subplots_adjust(top=0.85, wspace=0.3)

            ax = fig.add_subplot(1,1, 1)
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Values") 
            w_q = performanceTable.iloc[:,i].value_counts() ## original data
            #w_q = dfplot['check_account'].value_counts()
            w_q = (list(w_q.index), list(w_q.values))
            ax.tick_params(axis='both', which='major', labelsize=8.5)
            bar = ax.bar(w_q[1], w_q[0], color='steelblue', 
                    edgecolor='black', linewidth=1)
            


dfplot = df
fig,axs =plt.subplots(figsize = (25, 15), ncols = 4,nrows= 5)
i=0
for nc in range(0,4):
    for nr in range(0,5):
        if (i!=20):  
            #for i in range(0,dfplot.shape[1]):
            #fig = plt.figure(figsize = (6, 4))
            #title = fig.suptitle(dfplot.iloc[:,i].name, fontsize=14)
            #fig.subplots_adjust(top=0.85, wspace=0.3)
            #ax = fig.add_subplot(4,5,1)
            #axs.set_xlabel("Frequency")
            axs[nr,nc].set_title(performanceTable.iloc[:,i].name)
            #axs[nr,nc].set_ylabel(performanceTable.iloc[:,i].name) 
            w_q = dfplot.iloc[:,i].value_counts() ## original data
            w_q = (list(w_q.index), list(w_q.values))
            #ax.tick_params(axis='both', which='major', labelsize=8.5)
            bar = axs[nr,nc].bar(w_q[1], w_q[0], color='steelblue', 
                edgecolor='black', linewidth=1  ) 
            i=i+1 
            
        else: break
fig.tight_layout()


#############################
# Correlation Matrix Heatmap##
#############################

f, ax = plt.subplots(figsize=(15, 10))
corr = performanceTable.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)

#########################
# Pair-wise Scatter Plots
############################
dfplot=performanceTable

dfplot = df

cols = dfplot.iloc[:,:5].columns
pp = sns.pairplot(dfplot[cols], size=2, aspect=1,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Credit card Pairwise Plots', fontsize=14)