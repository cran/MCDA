#library(MCDA)

# the separation threshold

epsilon <-0.05

# the performance table

performanceTable <- rbind(
  c(3,10,1),
  c(4,20,2),
  c(2,20,0),
  c(6,40,0),
  c(30,30,3))

rownames(performanceTable) <- c("RER","METRO1","METRO2","BUS","TAXI")

colnames(performanceTable) <- c("Price","Time","Comfort")

# ranks of the alternatives

alternativesRanks <- c(1,2,2,3,4)

names(alternativesRanks) <- row.names(performanceTable)

# criteria to minimize or maximize

criteriaMinMax <- c("min","min","max")

names(criteriaMinMax) <- colnames(performanceTable)

# number of break points for each criterion

criteriaNumberOfBreakPoints <- c(3,4,4)

names(criteriaNumberOfBreakPoints) <- colnames(performanceTable)

x1<-UTASTAR(performanceTable, criteriaMinMax, criteriaNumberOfBreakPoints, epsilon, alternativesRanks = alternativesRanks)

stopifnot(x1$Kendall ==1)

# let us try the same with the pairwise preferences to test if the results
# are the same

alternativesPreferences<-rbind(c("RER","METRO1"),
                               c("METRO2","BUS"),
                               c("BUS","TAXI"))

alternativesIndifferences<-rbind(c("METRO1","METRO2"))

x1prime <- UTASTAR(performanceTable, criteriaMinMax, criteriaNumberOfBreakPoints, epsilon, alternativesPreferences = alternativesPreferences, alternativesIndifferences = alternativesIndifferences)

stopifnot(all(x1$valueFunctions$Price == x1prime$valueFunctions$Price) && all(x1$valueFunctions$Time == x1prime$valueFunctions$Time) && all(x1$valueFunctions$Comfort == x1prime$valueFunctions$Comfort))

# now some filtering

x2<-UTASTAR(performanceTable, criteriaMinMax, criteriaNumberOfBreakPoints, epsilon, alternativesRanks = alternativesRanks, criteriaIDs = c("Price", "Time"), alternativesIDs = c("METRO1","METRO2","TAXI"))

stopifnot(x2$overallValues[1] == x2$overallValues[2])


