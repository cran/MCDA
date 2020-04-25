import pandas as pd
import random  
from Python.UTASTAR import *

# UTASTAR EXAMPLE
# the separation threshold

epsilon = 0.05

### German Credit Card Approval Dataset
# Use Utastar to create Credit Score
# 20 criteria we can use the value functions the will be created for feautre reduction

df = pd.read_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\multicriteria matrix.xlsx",
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

#k=0
#minoptimum = 10 
#while True:

# # criteria to minimize or maximize
# choices = ["max","min"]
# minmaxdata = [random.choice(choices) for i in range(df.shape[1]-1)]

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
print(
    "\nCriteria Min Max \n", criteriaMinMax,
)



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

#k = k+1
#print(k,"<--iteration")
    


    # if float(optimum.values) < minoptimum:
    #     minoptimum = float(optimum.values)
    #     print(float(optimum.values),"min optimum ")
    #     minbpdata = bpdata
    #     minminmaxdata = minmaxdata

    #     tempminoptimum = pd.DataFrame([minoptimum])
    #     tempminbpdata = pd.DataFrame([minbpdata])
    #     tempminminmaxdata = pd.DataFrame([minminmaxdata])

    #     tempminoptimum.to_excel(r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\minoptimum.xlsx")
    #     tempminbpdata.to_excel(r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\minbpdata.xlsx")
    #     tempminminmaxdata.to_excel(r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\minminmaxdata.xlsx")

    # if float(optimum.values) == 0:
    #     print(bpdata)
    #     print(minmaxdata)
    #     break

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



print("X ends Here")

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
)

print("End")


valueFunctions.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\ValueFunctions.xlsx"
)
overallValues.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\OverallValues.xlsx"
)
outRanks.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\OutRanks.xlsx"
)
averageValueFunctions.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\AverageValueFunctions.xlsx"
)
averageOverallValues.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\AverageOverallValues.xlsx"
)

# Insert to multi criteria matrix


# Insert averageOverallValues as column  to Alternatives
df.insert(21, "AverageOverallValues", averageOverallValues.transpose())
# Insert overallValues values as column to Alternatives
df.insert(22, "OverallValues", overallValues.transpose())
# Inert outRanks values as column to Alternatives
df.insert(23, "OutRanks", outRanks.transpose())

# Insert valueFunctions to criteria/columns
data = [valueFunctions.iloc[x, (bpdata[x//2]-1)] for x in range(1, len(valueFunctions), 2) ]
data = pd.DataFrame(data, index=performanceTable.columns.values, columns=["ValueFunc"])
data = data.transpose()

# df = df.append(data)
df = pd.concat([data, df], ignore_index=False)

# Insert averagevalueFunctions to criteria/columns
data = [
    averageValueFunctions.iloc[x, (bpdata[x//2]-1)] for x in range(1, len(averageValueFunctions), 2)
]
data = pd.DataFrame(
    data, index=performanceTable.columns.values, columns=["AverageValueFunc"]
)
data = data.transpose()
# df = df.append(data)
df = pd.concat([data, df], ignore_index=False)

print(df)
df.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\Results.xlsx"
)

