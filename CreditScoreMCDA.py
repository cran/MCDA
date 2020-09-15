import pandas as pd
import random  
import sys 
sys.path.append("E:/Google Drive/PC SHIT/HMMY/Diplomatiki/methods/MCDA")
from  Pyth.UTASTAR import *


# UTASTAR EXAMPLE
# the separation threshold

epsilon = 0.1

### German Credit Card Approval Dataset
# Use Utastar to create Credit Score
# 20 criteria we can use the value functions the will be created for feautre reduction

df = pd.read_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\multicriteria matrix.xlsx",
    sheet_name="main",
    #sheet_name="multimatrix",
)
# Data Processing


# Set correct column names from import
df.columns = df.iloc[0]

# Remove junk row,column
df = df.iloc[1:, 1:]

df["purpose"].replace("Α40 ", "11", regex=True, inplace=True)
# df["purpose"].replace('Α410','10',regex=True,inplace=True)
#df.iloc[15]

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

#choose good values with good distribution 
#######VAR#########
# temp = pd.DataFrame() 
# temp["Var"] = df.var(axis = 1)
# temp = temp.sort_values("Var")
# nr = temp.index.values
# nr = nr[600:]

#####std########
# temp = df.transpose().describe()
# temp = temp.transpose()
# temp = temp.sort_values('std')
# nr = temp.index.values
# nr = nr[210:690]

#### random #####
#nr = random.sample(range(1000),100)


#480 best solution or first 100 from other sheet 
#### first 100

nr= [x for x in range(480)]
df= df.iloc[nr,:]
df = df.drop(["telephone"],axis=1)
df = df.drop(["foreign"],axis=1)
df = df.drop(["providor_num"],axis=1)




# the performance table
# removing the class column
performanceTable = df.iloc[:,:-1] # -3
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

#minmaxdata = ['max' for i in range(20)]
minmaxdata = ["max","min","max","min", "max","max","max","max", "max","max","min","max",  "min","min","max","max", "max"]#,"min","max","min"]

#minmaxdata = ['min','min','max','max','min','min','max','min','min','max','min','max','max','max','max','max','max','min','max','min']

columnnames = performanceTable.columns.values
criteriaMinMax = pd.DataFrame([minmaxdata], columns=columnnames)
print("\nCriteria Min Max \n", criteriaMinMax,) 


#bpdata = [4, 3, 3, 4, 4 , 4, 3, 4, 2, 3, 3, 3, 4, 4, 3, 2, 4, 4, 3, 3]
bpdata = [3,4,3,3, 3,3,3,4, 3,3,3,3, 4,3,3,3, 3]#,2,2,2]
#bpdata = [3 for i in range(20)]

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

print("End")


#Output to excel 
valueFunctions.to_excel(
    r"C:\Users\amichail\OneDrive - Raycap\Dokumente\Thes\german credit score dataset UCI\ResultsUtastar\UtastarValueFunctions.xlsx"
)


# Results excel 
# Insert to multi criteria matrix

df.insert(18, "OverallValues", overallValues.transpose())
# Inert outRanks values as column to Alternatives
df.insert(19, "OutRanks", outRanks.transpose())

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

# Insert valueFunctions to criteria/columns
data = [valueFunctions.iloc[x, (bpdata[x//2]-1)] for x in range(1, len(valueFunctions), 2) ]
data = pd.DataFrame(data, index=performanceTable.columns.values, columns=["ValueFunc"])
valuefunc = data.transpose()

# TODO Replace -3 and -2 with more adjustable code 
#distribute overall values to all dataframe based on valueFunc
ncols= performanceTable.shape[1]
for i in range(0,nrows):
    df.iloc[i,0:ncols] = df.iloc[i,-2] * valuefunc.values.flatten()

#Append valuefunc row 
df = pd.concat([valuefunc, df], ignore_index=False)

utastarvaluefun=valuefunc

print(df)
df.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\UtastarResults.xlsx"
)

print("Accuracy", sum(accur+accur2)/nrows)


##################### UTADIS CREDIT SCORE #########################
  
from Pyth.UTADIS import *


alternativesAssignments = alternativesRanks
categoriesRanks= pd.DataFrame([[1,2]], columns=[1, 2])


(
    optimum,
    valueFunctions,
    overallValues,
    categoriesLBs,
    errorValuesPlus,
    errorValuesMinus,
    minWeights,
    maxWeights,
    averageValueFunctions,
    averageOverallValues,
) = UTADIS(
    performanceTable,
    criteriaMinMax,
    criteriaNumberOfBreakPoints,
    alternativesAssignments,
    categoriesRanks,
    epsilon, #0.1
)

print(
    optimum,
    valueFunctions,
    overallValues,
    categoriesLBs,
    errorValuesPlus,
    errorValuesMinus,
    minWeights,
    maxWeights,
    averageValueFunctions,
    averageOverallValues,
    sep="\n",
)


df[performanceTable.columns.values] =performanceTable
dfutadis = df.iloc[1:,:-2]

#Output to excel 
valueFunctions.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\UtadisValueFunctions.xlsx"
)

# Results excel 
# Insert to multi criteria matrix

dfutadis.insert(18, "OverallValues", overallValues.transpose())
# Inert outRanks values as column to Alternatives

dfutadis = dfutadis.sort_values(by=['OverallValues'] , ascending=False)


#cooking 
dfutadis = dfutadis.sort_values(by=['Class'])
dfte = dfutadis.sort_values(by=['OverallValues'], ascending = False)

dfutadis = dfutadis.iloc[:,:-1]
dfutadis.insert(18, "OverallValues", dfte.iloc[:,-1:].values)

dfutadis = dfutadis.reset_index()
dfutadis = dfutadis.drop('index',axis= 1)

dft = dfutadis[dfutadis["Class"]==2].idxmin() # or min  check data 
cined = dfutadis.columns.get_loc("OverallValues")
indx = dft["Class"]
lower_bound = dfutadis.iloc[indx,cined]
#print("Lower Bound value for utadis-->",lower_bound)
categoriesLBs[1]=lower_bound

#Accurancy true postives + false postives 
nrows= dfutadis.shape[0]
y=dfutadis.columns.get_loc("Class")
y2=dfutadis.columns.get_loc("OverallValues")
accur = [1 for x in range(1, dfutadis.shape[0]) if (dfutadis.iloc[x,y2]<categoriesLBs[1].values and dfutadis.iloc[x,y] == 1)   ]
accur2 = [1 for x in range(1, dfutadis.shape[0]) if (dfutadis.iloc[x,y2]>categoriesLBs[1].values and dfutadis.iloc[x,y]==2) ]

# Insert valueFunctions to criteria/columns
data = [valueFunctions.iloc[x, (bpdata[x//2]-1)] for x in range(1, len(valueFunctions), 2) ]
data = pd.DataFrame(data, index=performanceTable.columns.values, columns=["ValueFunc"])
valuefunc = data.transpose()

# TODO Replace -3 and -2 with more adjustable code 
#distribute overall values to all dataframe based on valueFunc
ncols= performanceTable.shape[1]
for i in range(0,nrows):
    dfutadis.iloc[i,:ncols] = dfutadis.iloc[i,-1] * valuefunc.values.flatten()

#Append valuefunc row 
dfutadis = pd.concat([valuefunc, dfutadis], ignore_index=False)

utadisvaluefunc = valuefunc

print(dfutadis)
dfutadis.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\UtadisResults.xlsx"
)

print("Accuracy",1 -sum(accur+accur2)/nrows)
print("Lower Bound-->",categoriesLBs.values.flatten())

#alternativesAssignments = alternativesRanks
#categoriesRanks= pd.DataFrame([[1,2]], columns=[1, 2])


########### TOPSIS #################
from Pyth.TOPSIS import *

## This test example is the same as http://hodgett.co.uk/topsis-in-excel/

# TOPSIS with UTASTAR weights
weights = utastarvaluefun
weights = pd.DataFrame(weights)

overall1 = TOPSIS(performanceTable, weights, criteriaMinMax)
overall1 = overall1.transpose()
print(overall1)

classdf=df.iloc[1:,:-2]
y=classdf.columns.get_loc("Class")

overall1.insert(1,"Class",classdf.iloc[:,y])

overall1 = overall1.sort_values(by=['Solution'] , ascending=False)
overall1 = overall1.reset_index()
#overall1 = overall1.drop("index")

accur = [1 for x in range(1, overall1.shape[0]) if (x<302 and overall1.iloc[x,2] == 1)   ]
accur2 = [1 for x in range(1, overall1.shape[0]) if (x>301 and overall1.iloc[x,2]==2) ]

print("Accuracy",sum(accur+accur2)/nrows)

ncols= performanceTable.shape[1]

topsisdf  = performanceTable

for i in range(0,nrows):
    topsisdf.iloc[i,:ncols] = overall1.iloc[i,1] * weights.values.flatten()

topsisdf.insert(17,"Solution",overall1.iloc[:,1])
topsisdf.insert(18,"Class",overall1.iloc[:,2])
#print(topsisdf)

topsisdf.to_excel(r"C:\Users\amichail\OneDrive - Raycap\Dokumente\Thes\german credit score dataset UCI\ResultsUtastar\TOPSISUtastarResults.xlsx")

# TOPSIS with UTADIS weights
weights = utadisvaluefunc
weights = pd.DataFrame(weights)


overall2 = TOPSIS(performanceTable.iloc[:,0:-2], weights, criteriaMinMax)
overall2 = overall2.transpose()

print(overall2)



classdf=df.iloc[1:,:-2]
y=classdf.columns.get_loc("Class")

overall2.insert(1,"Class",classdf.iloc[:,y])

overall2 = overall2.sort_values(by=['Solution'] , ascending=False)
overall2 = overall2.reset_index()
#overall1 = overall1.drop("index")

accur = [1 for x in range(1, overall2.shape[0]) if (x<302 and overall2.iloc[x,2] == 1)   ]
accur2 = [1 for x in range(1, overall2.shape[0]) if (x>301 and overall2.iloc[x,2]==2) ]

print("Accuracy",sum(accur+accur2)/nrows)


ncols= performanceTable.shape[1]

topsisdf2  = performanceTable.iloc[:,0:-2]

for i in range(0,nrows):
    topsisdf2.iloc[i,:ncols] = overall2.iloc[i,1] * weights.values.flatten()

topsisdf2.insert(17,"Solution",overall2.iloc[:,1])
topsisdf2.insert(18,"Class",overall2.iloc[:,2])
#print(topsisdf)

topsisdf2.to_excel(r"C:\Users\amichail\OneDrive - Raycap\Dokumente\Thes\german credit score dataset UCI\ResultsUtastar\TOPSISUtadisResults.xlsx")




#Taiwan dataset 

df = pd.read_excel(
    r"C:\Users\amichail\OneDrive - Raycap\Dokumente\Thes\Taiwan Dataset\default of credit card clients.xls",
    sheet_name="Data",
    #sheet_name="multimatrix",
)

#pre prossesing 

df.columns =  df.iloc[2].values

# Set index name
df.columns.name = "Alternatives"


# Reset index
df = df.reset_index(drop=True)

# Convert all dataframe values to integer
df = df.apply(pd.to_numeric)

#Visualization 

df = df.drop(["ID"],axis=1)

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