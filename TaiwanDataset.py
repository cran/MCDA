
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
nrows=440
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


############Plots ######

import matplotlib.pyplot as plt
import seaborn as sns
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



#############################
# Correlation Matrix Heatmap##
#############################

f, ax = plt.subplots(figsize=(15, 10))
corr = df.corr()  
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)



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