import pandas as pd
from utastarpy import UTASTAR

# import numpy as np

# the separation threshold

epsilon = 0.05

# the performance table
data = ([3, 10, 1], [4, 20, 2], [2, 20, 0], [6, 40, 0], [30, 30, 3])
performanceTable = pd.DataFrame(
    data,
    columns=["Price", "Time", "Comfort"],
    index=["RER", "METRO1", "METRO2", "BUS", "TAXI"],
)

print(performanceTable,"Performance Table")
# ranks of the alternatives
data = [1, 2, 2, 3, 4]
rownames = performanceTable.index.values
alternativesRanks = pd.DataFrame(
   [data] , columns=rownames
)
print(alternativesRanks,"Alternative Ranks")

# criteria to minimize or maximize
data = ["min", "min", "max"]
columnnames = performanceTable.columns.values
criteriaMinMax = pd.DataFrame(
    [data], columns=columnnames
)
print(criteriaMinMax,"Criteria MIN MAX")
# number of break points for each criterion

criteriaNumberOfBreakPoints = pd.DataFrame(
    [[3, 4, 4]], columns=performanceTable.columns.values
)
print(criteriaNumberOfBreakPoints, " criteriaNumofBP")

x1 = UTASTAR(performanceTable, criteriaMinMax, criteriaNumberOfBreakPoints, epsilon, alternativesRanks=alternativesRanks)

#assert(x1["Kendall"] == 1)

# let us try the same with the pairwise preferences to test if the results
# are the same

alternativesPreferences = pd.DataFrame([["RER", "METRO1"], ["METRO2", "BUS"], ["BUS", "TAXI"]])
print(alternativesPreferences)
alternativesIndifferences = pd.DataFrame(["METRO1", "METRO2"])
print(alternativesIndifferences)

x1prime =  UTASTAR(performanceTable, criteriaMinMax, criteriaNumberOfBreakPoints, epsilon,alternativesPreferences=alternativesPreferences, alternativesIndifferences=alternativesIndifferences)

# assert(all(x1$valueFunctions$Price == x1prime$valueFunctions$Price) & & all(x1$valueFunctions$Time == x1prime$valueFunctions$Time) & & all(x1$valueFunctions$Comfort == x1prime$valueFunctions$Comfort))

# now some filtering
x2 = UTASTAR(performanceTable, criteriaMinMax, criteriaNumberOfBreakPoints, epsilon,alternativesRanks=alternativesRanks, criteriaIDs=["Price", "Time"], alternativesIDs=["METRO1", "METRO2", "TAXI"])

# assert(x2$overallValues[1] == x2$overallValues[2])

