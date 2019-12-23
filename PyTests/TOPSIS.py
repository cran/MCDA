
import pandas as pd 
from Python.TOPSIS import *
## This test example is the same as http://hodgett.co.uk/topsis-in-excel/
data = [[5490,51.4,8.5,285],[6500,70.6,7,288],[6489,54.3,7.5,290]]
performanceTable = pd.DataFrame()
print(performanceTable)

performanceTable.index =  ["Corsa","Clio","Fiesta"]
performanceTable.columns = ["Purchase Price","Economy","Aesthetics","Boot Capacity"]

weights = [0.35,0.25,0.25,0.15]

criteriaMinMax = ["min", "max", "max", "max"]

positiveIdealSolutions = [0.179573776, 0.171636015, 0.159499658, 0.087302767]
negativeIdealSolutions = [0.212610118, 0.124958799, 0.131352659, 0.085797547]

weights.index = performanceTable.columns
criteriaMinMax.index = performanceTable.columns
positiveIdealSolutions.index = performanceTable.columns
negativeIdealSolutions.index = performanceTable.columns

overall1 = TOPSIS(performanceTable, weights, criteriaMinMax)

overall2 = TOPSIS(performanceTable, weights, criteriaMinMax, positiveIdealSolutions, negativeIdealSolutions)

overall3 = TOPSIS(performanceTable, weights, criteriaMinMax, alternativesIDs = ["Corsa","Clio"], criteriaIDs = ["Purchase Price","Economy","Aesthetics"])

overall4 = TOPSIS(performanceTable, weights, criteriaMinMax, positiveIdealSolutions, negativeIdealSolutions, alternativesIDs = ["Corsa","Clio"], criteriaIDs =["Purchase Price","Economy","Aesthetics"])

# s1 <- structure(c(0.4817, 0.5182, 0.1780
# ), .Names = c("Corsa", "Clio", "Fiesta"))

# stopifnot(round(overall1,4) == s1)

# s2<- structure(c(0.4817, 0.5182, 0.1780
# ), .Names = c("Corsa", "Clio", "Fiesta"))

# stopifnot(round(overall2,4) == s2)

# s3<- structure(c(0.4943, 0.5057), .Names = c("Corsa", "Clio"))

# stopifnot(round(overall3,4) == s3)

# s4 <- structure(c(0.5182, 0.5146), .Names = c("Corsa", "Clio"))

# stopifnot(round(overall4,4) == s4)

