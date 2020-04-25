import pandas as pd
from Python.UTADIS import *

# UTADIS EXAMPLE
# the separation threshold

epsilon = 0.05

# the performance table
data = ([3, 10, 1], [4, 20, 2], [2, 20, 0], [6, 40, 0], [30, 30, 3])
performanceTable = pd.DataFrame(
    data,
    columns=["Price", "Time", "Comfort"],
    index=["RER", "METRO1", "METRO2", "BUS", "TAXI"],
)

print("\nPerformance Table \n", performanceTable)
# ranks of the alternatives
data = ["good", "medium", "medium", "bad", "bad"]
rownames = performanceTable.index.values
alternativesAssignments = pd.DataFrame([data], columns=rownames)
print("\nAlternative Assignments\n", alternativesAssignments)

# criteria to minimize or maximize
data = ["min", "min", "max"]
columnnames = performanceTable.columns.values
criteriaMinMax = pd.DataFrame([data], columns=columnnames)
print(
    "\nCriteria Min Max \n", criteriaMinMax,
)
# number of break points for each criterion

criteriaNumberOfBreakPoints = pd.DataFrame(
    [[3, 4, 4]], columns=performanceTable.columns.values
)
print("\nCriteriaNumofBP\n", criteriaNumberOfBreakPoints)


categoriesRanks = pd.DataFrame([[1, 2, 3]], columns=["good", "medium", "bad"])
# ranks of the categories
print("\nCategoriesRanks", categoriesRanks, sep="\n")

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
    0.1,
)

print("X=")
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

#k-post Optimality 


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
    0.3,
    kPostOptimality = 0.3
)

print("Post optimal X =")
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


# filtering out category "good" and assigment examples "RER" and "TAXI"

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
    0.1,
    categoriesIDs=["medium", "bad"],
    alternativesIDs=["METRO1", "METRO2", "BUS"],
)

print("Y=")
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
    0.1,
    criteriaIDs=["Comfort", "Time"],
)
print("Z=")
# stopifnot(x$optimum ==0 && y$optimum ==0 && z$optimum ==0)

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