import pandas as pd
import random  
import sys 
#sys.path.append("E:/Google Drive/PC SHIT/HMMY/Diplomatiki/methods/MCDA")
#from UtadiskmeansCreditScore import utadisvaluefunc
#from UtastarkmeansCreditScore import utastarvaluefun

# UTASTAR EXAMPLE
# the separation threshold

epsilon = 0.1

### German Credit Card Approval Dataset
# Use Utastar to create Credit Score
# 20 criteria we can use the value functions the will be created for feautre reduction

df = pd.read_excel(
    r"C:\Users\amichail\OneDrive - Raycap\Dokumente\Thes\german credit score dataset UCI\multicriteria matrix.xlsx",
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


########### TOPSIS #################
from Pyth.TOPSIS import *

## This test example is the same as http://hodgett.co.uk/topsis-in-excel/

nrows = performanceTable.shape[0]
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









########### TOPSIS with UTADIS weights ############
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


