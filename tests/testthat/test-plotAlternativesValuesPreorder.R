#library(MCDA)
test_that("plotAlternativesValuesPreorder works", {
  alternativesValues <- c(10,1,8,3,8,3,4,4,8,5)
  
  names(alternativesValues) <- c("x10","x1","x9","x2","x8","x3","x7","x4","x6","x5")
  
  x <- plotAlternativesValuesPreorder(alternativesValues, 
                                 decreasing=TRUE, 
                                 alternativesIDs=c("x10","x3","x7","x4","x6","x5"), 
                                 silent=TRUE)
  expect_equal(x, c("x10","x6","x5","x7, x4","x3"))
})
