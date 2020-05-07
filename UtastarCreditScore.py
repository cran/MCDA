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
""" 
%matplotlib inline
sns.set_context('notebook')
plt.style.use('fivethirtyeight')
from warnings import filterwarnings
filterwarnings('ignore')


# Plot the data
plt.figure(figsize=(6, 6))
plt.scatter(dfplot.iloc[:, 1], dfplot.iloc[:,2],)
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of  data')
 """
# matrix to be used for k menas
from math import pi


#data
dfplot = df.iloc[1:,:-3]

###########parralel coordinate plot ############
pd.DataFrame(dfplot).plot()
plt.show


######### radar graph ###############
categories=list(dfplot)[1:]
N = len(categories)

values=dfplot.loc[0].drop('check_account').values.flatten().tolist()
values += values[:1]
values
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
plt.ylim(0,40)
 
# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)


#############parallel plot with pandas #######
import pandas
from pandas.plotting import parallel_coordinates

# Make the plot
data = dfplot.transpose()
data.insert(0,"Class",data.index)

parallel_coordinates(data, "Class" , colormap=plt.get_cmap("Set2"))
plt.show()


######## Histograms with pandas##### pair of 2 
dfplot.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)

plt.tight_layout(rect=(0, 0, 1.2, 1.2))