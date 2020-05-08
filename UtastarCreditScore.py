import pandas as pd
import random  
import sys 
sys.path.append("E:/Google Drive/PC SHIT/HMMY/Diplomatiki/methods/MCDA")
from  Pyth.UTASTAR import *

# UTASTAR EXAMPLE
# the separation threshold

epsilon = 0.05

### German Credit Card Approval Dataset
# Use Utastar to create Credit Score
# 20 criteria we can use the value functions the will be created for feautre reduction

df = pd.read_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\multicriteria matrix.xlsx",
    #sheet_name="main",
    sheet_name="multimatrix",
)
# Data Processing


# Set correct column names from import
df.columns = df.iloc[0]

# Remove junk row,column
df = df.iloc[1:, 1:]

df["purpose"].replace("Α40", "11", regex=True, inplace=True)
# df["purpose"].replace('Α410','10',regex=True,inplace=True)


# Change qualitative values to quantitative
df.replace("(.)(?!$)", "", regex=True, inplace=True)

# Per columnc change values if needed
# Replace true zero

df["check_account"].replace("4", "0", regex=True, inplace=True)
df["savings_account"].replace("5", "0", regex=True, inplace=True)
df["employment"].replace("1", "0", regex=True, inplace=True)
df["debtors_guarantors"].replace("1", "0", regex=True, inplace=True)
df["property"].replace("4", "0", regex=True, inplace=True)
df["installment_plans"].replace("3", "0", regex=True, inplace=True)
df["job"].replace("1", "0", regex=True, inplace=True)
df["telephone"].replace("1", "0", regex=True, inplace=True)


# Remove index column if needed
# df = df.set_index(list(df)[0])

# Set index name
df.columns.name = "Alternatives"

# Reset index
df = df.reset_index(drop=True)

# Convert all dataframe values to integer
df = df.apply(pd.to_numeric)

df.describe()

# the performance table
# removing the class column
performanceTable = df.iloc[:, :-1]
print("\nPerformance Table \n", performanceTable)

# ranks of the alternatives

data = df["Class"].values  
rownames = performanceTable.index.values
alternativesRanks = pd.DataFrame([data], columns=rownames)
print("\nAlternative Ranks\n", alternativesRanks)

#Find min breakpoint block 
""" 
k=0
minoptimum = 10 
while True:

# criteria to minimize or maximize
choices = ["max","min"]
minmaxdata = [random.choice(choices) for i in range(df.shape[1]-1)]
 """

minmaxdata = ['min',
 'min',
 'max',
 'max',
 'min',
 'min',
 'max',
 'min',
 'min',
 'max',
 'min',
 'max',
 'max',
 'max',
 'max',
 'max',
 'max',
 'min',
 'max',
 'min']
columnnames = performanceTable.columns.values
criteriaMinMax = pd.DataFrame([minmaxdata], columns=columnnames)
print("\nCriteria Min Max \n", criteriaMinMax,)

bpdata = [4, 3, 3, 4, 4, 4, 3, 4, 2, 3, 3, 3, 4, 4, 3, 2, 4, 4, 3, 3]
#choices = [2,3,4]
#bpdata = [random.choice(choices) for i in range(df.shape[1]-1)]

# number of break points for each criterion
criteriaNumberOfBreakPoints = pd.DataFrame(
    [bpdata],
    columns=performanceTable.columns.values,
)
# print("\nCriteriaNumofBP\n", criteriaNumberOfBreakPoints)


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

###Find optimum break points 
""" k = k+1
print(k,"<--iteration")
    


    if float(optimum.values) < minoptimum:
        minoptimum = float(optimum.values)
        print(float(optimum.values),"min optimum ")
        minbpdata = bpdata
        minminmaxdata = minmaxdata

        tempminoptimum = pd.DataFrame([minoptimum])
        tempminbpdata = pd.DataFrame([minbpdata])
        tempminminmaxdata = pd.DataFrame([minminmaxdata])

        tempminoptimum.to_excel(r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\minoptimum.xlsx")
        tempminbpdata.to_excel(r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\minbpdata.xlsx")
        tempminminmaxdata.to_excel(r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\minminmaxdata.xlsx")

    if float(optimum.values) == 0:
        print(bpdata)
        print(minmaxdata)
        break """


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

#Utastar with post optimality but not needed as optimality is not achivied
""" 
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
    kPostOptimality=0.01,
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
) """

print("End")


#Output to excel 
valueFunctions.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\ValueFunctions.xlsx"
)
overallValues.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\OverallValues.xlsx"
)
outRanks.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\OutRanks.xlsx"
)

# averageValueFunctions.to_excel(
#     r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\AverageValueFunctions.xlsx"
# )
# averageOverallValues.to_excel(
#     r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\AverageOverallValues.xlsx"
# )



# Results excel 
# Insert to multi criteria matrix

# Insert averageOverallValues as column  to Alternatives
    #df.insert(21, "AverageOverallValues", averageOverallValues.transpose())
# Insert overallValues values as column to Alternatives
df.insert(21, "OverallValues", overallValues.transpose())
# Inert outRanks values as column to Alternatives
df.insert(22, "OutRanks", outRanks.transpose())

#Accurancy true postives + false postives 
nrows= df.shape[0]
y=df.columns.get_loc("Class")
y2=df.columns.get_loc("OutRanks")
accur = [1 for x in range(1, df.shape[0]) if (df.iloc[x,y2]<nrows/2 and df.iloc[x,y] == 1)   ]
accur2 = [1 for x in range(1, df.shape[0]) if (df.iloc[x,y2]>nrows/2 and df.iloc[x,y]==2) ]
#print("Accuracy", sum(accur+accur2))

# Insert valueFunctions to criteria/columns
data = [valueFunctions.iloc[x, (bpdata[x//2]-1)] for x in range(1, len(valueFunctions), 2) ]
data = pd.DataFrame(data, index=performanceTable.columns.values, columns=["ValueFunc"])
valuefunc = data.transpose()

# TODO Replace -3 and -2 with more adjustable code 
#distribute overall values to all dataframe based on valueFunc
for i in range(0,nrows):
    #margin = df.iloc[i,-2] / valuefunc.values /100
    #df.replace("inf", 0 , regex=True, inplace=True)
    df.iloc[i,0:-3] = df.iloc[i,-2] * valuefunc.values.flatten()# margin.flatten() #
    #df.iloc[i,0:-3] = pd.DataFrame(df.iloc[i,0:-3].dot(valuefunc.values.flatten()))


#Append valuefunc row 
df = pd.concat([valuefunc, df], ignore_index=False)

# Insert averagevalueFunctions to criteria/columns block 
""" data = [
    averageValueFunctions.iloc[x, (bpdata[x//2]-1)] for x in range(1, len(averageValueFunctions), 2)
]
data = pd.DataFrame(
    data, index=performanceTable.columns.values, columns=["AverageValueFunc"]
)
avaluefunc = data.transpose()
# append average value func row 
df = pd.concat([avaluefunc, df], ignore_index=False)
 """

print(df)
df.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\Results.xlsx"
)

print("Accuracy", sum(accur+accur2)/nrows)


#kmeans feautre reduction 
#data
dfplot = df.iloc[1:,:-3]

### pari wise 
#imports
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.image import imread
import pandas as pd
import seaborn as sns
from sklearn.datasets.samples_generator import (make_blobs,
                                                make_circles,
                                                make_moons)
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
 
#%matplotlib inline
sns.set_context('notebook')
plt.style.use('fivethirtyeight')
from warnings import filterwarnings
filterwarnings('ignore')

dfplot = df.iloc[1:,:-3]

# Plot the data
plt.figure(figsize=(6, 6))
plt.scatter(dfplot.iloc[:, 1], dfplot.iloc[:,2],)
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of  data')
 
# matrix to be used for k menas
from math import pi




###########parralel coordinate plot ############
dfplot = df.iloc[1:,:-3]
dfplot=performanceTable

pd.DataFrame(dfplot).plot()
plt.show


######### radar graph ###############

categories=list(dfplot)
N = len(categories)

values=dfplot.loc[0].values.flatten().tolist()
values += values[:1]
values#=dfplot.values.flatten().tolist()
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.002,0.005,0.001], ["0.002","0.005","0.01"], color="grey", size=7)
plt.ylim(0,0.005) # 5 for perfoncatable
 
# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)


#############parallel plot with pandas #######
import pandas
from pandas.plotting import parallel_coordinates

# Make the plot
data = performanceTable.iloc[:,:5].transpose()
data.insert(0,"Class",data.index)

parallel_coordinates(data, "Class", colormap=plt.get_cmap("Set2") )
plt.show()

data = performanceTable.iloc[:,10:]
data.insert(0,"Class",data.index)

parallel_coordinates(data, "Class", colormap=plt.get_cmap("Set2") )
plt.show()

##### values 
dfplot = df.iloc[1:,:-3]

data = dfplot.iloc[:,:10] #.transpose()
data.insert(0,"Class",data.index)

parallel_coordinates(data, "Class", colormap=plt.get_cmap("Set2") )
plt.show()

###
data = dfplot.iloc[:,10:]
data.insert(0,"Class",data.index)

parallel_coordinates(data, "Class" , colormap=plt.get_cmap("Set2"))
plt.show()

######## Histograms with pandas##### pair of 2 
dfplot = df.iloc[1:,:-3]
dfplot.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=10, ylabelsize=10, grid=False )
plt.tight_layout()
plt.tight_layout(rect=(0, 0, 5, 5))

performanceTable.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=10, ylabelsize=10, grid=False )
plt.tight_layout()
plt.tight_layout(rect=(0, 0, 5, 5))

###### Bar Plot######### per feaure 


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

dfplot = df.iloc[1:,:-3]
for i in range(0,dfplot.shape[1]-1):
fig = plt.figure(figsize = (6, 4))
    title = fig.suptitle(dfplot.iloc[:,i].name, fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax = fig.add_subplot(1,1, 1)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Values") 
    w_q = dfplot.iloc[:,i].value_counts() ## original data
    #w_q = dfplot['check_account'].value_counts()
w_q = (list(w_q.index), list(w_q.values))
ax.tick_params(axis='both', which='major', labelsize=8.5)
    bar = ax.bar(w_q[1], w_q[0], color='steelblue', 
        edgecolor='black', linewidth=1)

# Correlation Matrix Heatmap with original matrix data 

f, ax = plt.subplots(figsize=(10, 6))
corr = performanceTable.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)

# Pair-wise Scatter Plots

#dfplot = performanceTable

dfplot=performanceTable
dfplot = df.iloc[1:,:-3]

cols = dfplot.iloc[:,:5].columns
pp = sns.pairplot(dfplot[cols], size=2, aspect=1,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Credit card Pairwise Plots', fontsize=14)

################### k means for feautre/alternatives  reduction ###############



#3 Using the elbow method to find out the optimal number of #clusters. 
#KMeans class from the sklearn library.
from sklearn.cluster import KMeans

wcss=[]
#dfk = performanceTable.values
dfk = df.iloc[1:,:-3].values 

for i in range(1,11): 
    kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )

    kmeans.fit(dfk)

    wcss.append(kmeans.inertia_)
#kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.
#4.Plot the elbow graph
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#5 According to the Elbow graph we deterrmine the clusters number as #5. Applying k-means algorithm to the X dataset.
kmeans = KMeans(n_clusters=4, init ='k-means++', max_iter=300, n_init=10,random_state=0 )

# We are going to use the fit predict method that returns for each #observation which cluster it belongs to. 
# The cluster to which #it belongs and it will return this cluster numbers into a 
# # single vector that is  called y K-means

X = dfk
y_kmeans = kmeans.fit_predict(X)


# #6 Visualising the clusters based on prediction 
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=10, c='red', label ='Cluster 1' )
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=10, c='blue', label ='Cluster 2')


#visual based on actual values values 
#plt.scatter(X[:,:10],X[:,10:],alpha = 0.5)

#Plot the centroid. This time we're going to use the cluster centres  
# attribute that returns here the coordinates of the centroid.
plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=30, c='yellow', label = 'Centroids' )
plt.title('Clusters k-means')
#plt.xlim(0.00175,0.0035)# predicted
#plt.ylim(0.0006,0.0015) #

#plt.xlim(0.00175,0.0035) #actual data 
#plt.ylim(0.0006,0.0015)

#plt.xlabel('Values)')
#plt.ylabel('Spending Score(1-100')
plt.show()