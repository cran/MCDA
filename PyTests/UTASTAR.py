from Python.UTASTAR import UTASTAR

# UTASTAR EXAMPLE
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
data = [1, 2, 2, 3, 4]
rownames = performanceTable.index.values
alternativesRanks = pd.DataFrame([data], columns=rownames)
print("\nAlternative Ranks\n", alternativesRanks)

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


x = UTASTAR(
    performanceTable,
    criteriaMinMax,
    criteriaNumberOfBreakPoints,
    epsilon,
    alternativesRanks=alternativesRanks,
)

print(x)

x1 = UTASTAR(
    performanceTable,
    criteriaMinMax,
    criteriaNumberOfBreakPoints,
    epsilon,
    alternativesRanks=alternativesRanks,
    kPostOptimality=0.5,
)

print(x1)
# assert Kendall == 1

# let us try the same with the pairwise preferences to test if the results
# are the same

alternativesPreferences = pd.DataFrame(
    [["RER", "METRO1"], ["METRO2", "BUS"], ["BUS", "TAXI"]]
)
print(alternativesPreferences)
alternativesIndifferences = pd.DataFrame([["METRO1"], ["METRO2"]])
alternativesIndifferences = alternativesIndifferences.transpose()
print(alternativesIndifferences)

x1prime = UTASTAR(
    performanceTable,
    criteriaMinMax,
    criteriaNumberOfBreakPoints,
    epsilon,
    alternativesPreferences=alternativesPreferences,
    alternativesIndifferences=alternativesIndifferences,
)

print(x1prime)
# assert(all(x1$valueFunctions$Price == x1prime$valueFunctions$Price) & & all(x1$valueFunctions$Time == x1prime$valueFunctions$Time) & & all(x1$valueFunctions$Comfort == x1prime$valueFunctions$Comfort))

# now some filtering

x2 = UTASTAR(
    performanceTable,
    criteriaMinMax,
    criteriaNumberOfBreakPoints,
    epsilon,
    alternativesRanks=alternativesRanks,
    criteriaIDs=["Price", "Time"],
    alternativesIDs=["METRO1", "METRO2", "TAXI"],
)

print(x2)
# assert(x2$overallValues[1] == x2$overallValues[2])
