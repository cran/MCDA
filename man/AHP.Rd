% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/AHP.R
\name{AHP}
\alias{AHP}
\title{Analytic Hierarchy Process (AHP) method}
\usage{
AHP(criteriaWeightsPairwiseComparisons, alternativesPairwiseComparisonsList)
}
\arguments{
\item{criteriaWeightsPairwiseComparisons}{Matrix or data frame containing
the pairwise comparison matrix for the criteria weights. Lines and columns
are named according to the IDs of the criteria.}

\item{alternativesPairwiseComparisonsList}{A list containing a matrix or
data frame of pairwise comparisons (comparing alternatives) for each
criterion. The elements of the list are named according to the IDs of the
criteria. In each matrix, the lines and the columns are named according to
the IDs of the alternatives. If one criteria is already a score (i.e. it 
is a numeric value between 0 and 1 where higher values indicate better 
performance), then providing a nAlt-length vector, or a nAlt x 1 matrix
containing the score associated with each alternative will be enough, but
the vector or rows of the matrix must be named as the alternatives.}
}
\value{
The function returns a vector containing the AHP score for each
alternative.
}
\description{
AHP is a multi-criteria decision analysis method which was originally
developed by Thomas L. Saaty in 1970s.
}
\examples{
alts <- c("Corsa","Clio","Fiesta","Sandero")
style <- matrix(c(1.0, 1/4, 4.0, 1/6,
                  4.0, 1.0, 4.0, 1/4,
                  1/4, 1/4, 1.0, 1/5,
                  6.0, 4.0, 5.0, 1.0), 
                nrow=length(alts), ncol=length(alts), byrow=TRUE, 
                dimnames=list(alts,alts))
reliability <- matrix(c(1.0, 2.0, 5.0, 1.0,
                        1/2, 1.0, 3.0, 2.0,
                        1/5, 1/3, 1.0, 1/4,
                        1.0, 1/2, 4.0, 1.0), 
                      nrow=length(alts), ncol=length(alts), byrow=TRUE, 
                      dimnames=list(alts,alts))
fuel <- matrix(c(1.0, 2.0, 4.0, 1.0,
                 0.5, 1.0, 3.0, 2.0,
                 1/4, 1/3, 1.0, 1/5,
                 1.0, 1/2, 5.0, 1.0), 
               nrow=length(alts), ncol=length(alts), byrow=TRUE, 
               dimnames=list(alts,alts))
alternativesPairwiseComparisonsList <- list(style       = style, 
                                            reliability = reliability, 
                                            fuel        = fuel)
crit <- c("style","reliability","fuel")
criteriaWeightsPairwiseComparisons <- matrix(c(1.0, 1/2, 3.0,
                                               2.0, 1.0, 4.0,
                                               1/3, 1/4, 1.0), 
                                             nrow=length(crit), 
                                             ncol=length(crit), 
                                             dimnames=list(crit,crit))
# All attributes have pairwise comparisons
AHP(criteriaWeightsPairwiseComparisons, alternativesPairwiseComparisonsList)
# Fuel is a score
newFuel <- c(Corsa=34, Clio=27, Fiest=24, Sandero=28)
newFuel <- newFuel/sum(newFuel)
alternativesPairwiseComparisonsList$fuel <- newFuel
AHP(criteriaWeightsPairwiseComparisons, alternativesPairwiseComparisonsList)

}
\references{
The Analytic Hierarchy Process: Planning, Priority Setting
(1980), ISBN 0-07-054371-2, McGraw-Hill
}
