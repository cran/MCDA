#' Identification of profiles, weights, majority threshold and veto thresholds
#' for MRSort using a genetic algorithm.
#' 
#' MRSort is a simplification of the Electre TRI method that uses the
#' pessimistic assignment rule, without indifference or preference thresholds
#' attached to criteria. Only a binary discordance condition is considered,
#' i.e. a veto forbids an outranking in any possible concordance situation, or
#' not. The identification of the profiles, weights, majority threshold and
#' veto thresholds are done by taking into account assignment examples.
#' 
#' 
#' @param performanceTable Matrix or data frame containing the performance
#' table.  Each row corresponds to an alternative, and each column to a
#' criterion.  Rows (resp. columns) must be named according to the IDs of the
#' alternatives (resp. criteria).
#' @param assignments Vector containing the assignments (IDs of the categories)
#' of the alternatives to the categories. The elements are named according to
#' the alternatives.
#' @param categoriesRanks Vector containing the ranks of the categories. The
#' elements are named according to the IDs of the categories.
#' @param criteriaMinMax Vector containing the preference direction on each of
#' the criteria. "min" (resp. "max") indicates that the criterion has to be
#' minimized (maximized). The elements are named according to the IDs of the
#' criteria.
#' @param veto Boolean parameter indicating whether veto profiles are to be
#' used or not.
#' @param alternativesIDs Vector containing IDs of alternatives, according to
#' which the data should be filtered.
#' @param criteriaIDs Vector containing IDs of criteria, according to which the
#' data should be filtered.
#' @param timeLimit Allows to fix a time limit of the execution, in seconds
#' (default 60).
#' @param populationSize Allows to change the size of the population used by
#' the genetic algorithm (default 20).
#' @param mutationProb Allows to change the mutation probability used by the
#' genetic algorithm (default 0.1).
#' @return The function returns a list containing: \item{majorityThreshold}{The
#' inferred majority threshold (single numeric value).}
#' \item{criteriaWeights}{The inferred criteria weights (a vector named with
#' the criteria IDs).} \item{profilesPerformances}{The inferred category limits
#' (a matrix with the column names given by the criteria IDs and the rownames
#' given by the upper categories each profile delimits).}
#' \item{vetoPerformances}{The inferred vetoes (a matrix with the column names
#' given by the criteria IDs and the rownames given by the categories to which
#' each profile applies).} \item{fitness}{The classification accuracy of the
#' inferred model (from 0 to 1).}
#' @references Bouyssou, D. and Marchant, T. An axiomatic approach to
#' noncompen- satory sorting methods in MCDM, II: more than two categories.
#' European Journal of Operational Research, 178(1): 246--276, 2007.
#' 
#' no reference yet for the algorithmic approach; one should become available
#' in 2018
#' @keywords methods
#' @examples
#' 
#' \donttest{
#' performanceTable <- rbind(c(10,10,9), c(10,9,10), c(9,10,10), c(9,9,10), c(9,10,9), c(10,9,9),
#'                           c(10,10,7), c(10,7,10), c(7,10,10), c(9,9,17), c(9,17,9), c(17,9,9),
#'                           c(7,10,17), c(10,17,7), c(17,7,10), c(7,17,10), c(17,10,7), c(10,7,17),
#'                           c(7,9,17), c(9,17,7), c(17,7,9), c(7,17,9), c(17,9,7), c(9,7,17))
#' 
#' rownames(performanceTable) <- c("a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11",
#'                                 "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
#'                                 "a21", "a22", "a23", "a24")
#' 
#' colnames(performanceTable) <- c("c1","c2","c3")
#' 
#' assignments <-c("P", "P", "P", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F",
#'                 "F", "F", "F", "F", "F", "F", "F")
#' 
#' names(assignments) <- rownames(performanceTable)
#' 
#' categoriesRanks <- c(1,2)
#' 
#' names(categoriesRanks) <- c("P","F")
#' 
#' criteriaMinMax <- c("max","max","max")
#' 
#' names(criteriaMinMax) <- colnames(performanceTable)
#' 
#' set.seed(1)
#' 
#' x<-MRSortInferenceApprox(performanceTable, assignments, categoriesRanks, 
#'                          criteriaMinMax, veto = TRUE,
#'                          alternativesIDs = c("a1","a2","a3","a4","a5","a6","a7"))
#' }
#' 
#' @export MRSortInferenceApprox
MRSortInferenceApprox <- function(performanceTable, assignments, categoriesRanks, criteriaMinMax, veto = FALSE, alternativesIDs = NULL, criteriaIDs = NULL, timeLimit = 60, populationSize = 20, mutationProb = 0.1){
  
  ## check the input data
  
  if (!(is.matrix(performanceTable) || is.data.frame(performanceTable))) 
    stop("performanceTable should be a matrix or a data frame")
  
  if(is.null(colnames(performanceTable)))
    stop("performanceTable columns should be named")
  
  if (!(is.vector(assignments)))
    stop("assignments should be a vector")
  
  if(is.null(names(assignments)))
    stop("assignments should be named")
  
  if (!(is.vector(criteriaMinMax)))
    stop("criteriaMinMax should be a vector")
  
  if(!all(sort(colnames(performanceTable)) == sort(names(criteriaMinMax))))
    stop("criteriaMinMax should be named as the columns of performanceTable")
  
  if (!(is.vector(categoriesRanks)))
    stop("categoriesRanks should be a vector")
  
  if(is.null(names(categoriesRanks)))
    stop("categoriesRanks should be named")
  
  if(!all(assignments %in% names(categoriesRanks)))
    stop("some of the assignments reference a category which does not exist in categoriesRanks")
  
  if (!is.logical(veto))
    stop("veto should be a boolean")
  
  if (!(is.null(timeLimit)))
  {
    if(!is.numeric(timeLimit))
      stop("timeLimit should be numeric")
    if(timeLimit <= 0)
      stop("timeLimit should be strictly positive")
  }
  
  if (!(is.null(populationSize)))
  {
    if(!is.numeric(populationSize))
      stop("populationSize should be numeric")
    if(populationSize < 10)
      stop("populationSize should be at least 10")
  }
  
  if (!(is.null(mutationProb)))
  {
    if(!is.numeric(mutationProb))
      stop("mutationProb should be numeric")
    if(mutationProb < 0 || mutationProb > 1)
      stop("mutationProb should be between 0 and 1")
  }

  if (!(is.null(alternativesIDs) || is.vector(alternativesIDs)))
    stop("alternativesIDs should be a vector")
  
  if (!(is.null(criteriaIDs) || is.vector(criteriaIDs)))
    stop("criteriaIDs should be a vector")
  
  ## filter the data according to the given alternatives and criteria
  
  if (!is.null(alternativesIDs)){
    performanceTable <- performanceTable[alternativesIDs,]
    assignments <- assignments[names(assignments) %in% alternativesIDs]
  }
  
  if (!is.null(criteriaIDs)){
    performanceTable <- performanceTable[,criteriaIDs]
    criteriaMinMax <- criteriaMinMax[criteriaIDs]
  }
  
  if (is.null(dim(performanceTable))) 
    stop("less than 2 criteria or 2 alternatives")
  
  if (length(assignments) == 0) 
    stop("assignments is empty or the provided alternativesIDs have filtered out everything from within")
  
  # -------------------------------------------------------
  
  numAlt <- dim(performanceTable)[1]
  numCrit <- dim(performanceTable)[2]
  numCat <- length(categoriesRanks)
  minEvaluations <- apply(performanceTable, 2, min)
  maxEvaluations <- apply(performanceTable, 2, max)
  
  # -------------------------------------------------------
  
  getCategory <- function(alternativePerformances, criteriaWeights, majorityThreshold, profilesPerformances, vetoPerformances, criteriaMinMax){
    for (k in (numCat-1):1)
    {
      weightedSum <- 0
      
      for (i in 1:numCrit)
      {
        if (criteriaMinMax[i] == "min")
        {
          if (alternativePerformances[i] %<=% profilesPerformances[k,i])
            weightedSum <- weightedSum + criteriaWeights[i]
        }
        else
        {
          if (alternativePerformances[i] %>=% profilesPerformances[k,i])
            weightedSum <- weightedSum + criteriaWeights[i]
        }
      }
      
      vetoActive <- FALSE
      
      if(veto)
      {
        for (i in 1:numCrit)
        {
          if (criteriaMinMax[i] == "min")
          {
            if (alternativePerformances[i] %>=% vetoPerformances[k,i])
            {
              vetoActive <- TRUE
              break
            }
          }
          else
          {
            if (alternativePerformances[i] %<=% vetoPerformances[k,i])
            {
              vetoActive <- TRUE
              break
            }
          }
        }
      }
      
      # stopping condition
      if(weightedSum < majorityThreshold || vetoActive)
        return(k + 1)
    }
    # better than all profiles -> top categ
    return(1)
  }
  
  InitializePopulation <- function()
  {
    population <- list()
    for(i in 1:populationSize)
    {
      values <- c(0,sort(runif(numCrit-1,0,1)),1)
      weights <- sapply(1:numCrit, function(i) return(values[i+1]-values[i]))
      names(weights) <- colnames(performanceTable)
      
      majority <- runif(1,0.5,1)
      
      profiles <- NULL
      for(j in 1:numCrit)
      {
        if(criteriaMinMax[j] == 'max')
          profiles <- cbind(profiles,sort(runif(numCat - 1,minEvaluations[j],maxEvaluations[j]), decreasing = TRUE))
        else
          profiles <- cbind(profiles,sort(runif(numCat - 1,minEvaluations[j],maxEvaluations[j])))
      }
      colnames(profiles) <- colnames(performanceTable)
      
      vetoes <- NULL
      if(veto)
      {
        for(j in 1:numCrit)
        {
          if(criteriaMinMax[j] == 'max')
            vetoes <- cbind(vetoes,rep(minEvaluations[j] - 1, numCat - 1))
          else
            vetoes <- cbind(vetoes,rep(maxEvaluations[j] + 1, numCat - 1))
        }
        rownames(vetoes) <- c()
        colnames(vetoes) <- colnames(performanceTable)
      }
      
      population[[length(population)+1]] <- list(majorityThreshold = majority, criteriaWeights = weights, profilesPerformances = profiles, vetoPerformances = vetoes)
    }
    return(population)
  }
  
  Fitness <- function(individual)
  {
    ok <- 0
    for (alternative in names(assignments))
    {
      category <- getCategory(performanceTable[alternative,],individual$criteriaWeights, individual$majorityThreshold, individual$profilesPerformances, individual$vetoPerformances, criteriaMinMax)
      if(category == categoriesRanks[assignments[alternative]])
        ok <- ok + 1
    }
    return(ok/length(assignments))
  }
  
  Reproduce <- function(parents){
    children <- list()
    
    numPairs <- as.integer(length(parents)/2)
    
    pairings <- matrix(sample(1:length(parents),numPairs*2),numPairs,2)
    
    for(i in 1:numPairs)
    {
      parent1 <- parents[[pairings[i,1]]]
      
      parent2 <- parents[[pairings[i,2]]]
      
      # crossover bewtween profiles
      
      criteria <- sample(colnames(performanceTable), numCrit)
      
      pivot <- runif(1,1,numCrit - 1)
      
      profiles1 <- matrix(rep(0,numCrit*(numCat - 1)),numCat - 1,numCrit)
      profiles2 <- matrix(rep(0,numCrit*(numCat - 1)),numCat - 1,numCrit)
      
      colnames(profiles1) <- colnames(performanceTable)
      colnames(profiles2) <- colnames(performanceTable)
      
      for(k in 1:(numCat - 1))
        for(j in 1:numCrit)
        {
          if(j <= pivot)
          {
            profiles1[k,criteria[j]] <- parent1$profilesPerformances[k,criteria[j]]
            profiles2[k,criteria[j]] <- parent2$profilesPerformances[k,criteria[j]]
          }
          else
          {
            profiles1[k,criteria[j]] <- parent2$profilesPerformances[k,criteria[j]]
            profiles2[k,criteria[j]] <- parent1$profilesPerformances[k,criteria[j]]
          }
        }
      
      vetoes1 <- matrix(rep(NA,numCrit*(numCat - 1)),numCat - 1,numCrit)
      vetoes2 <- matrix(rep(NA,numCrit*(numCat - 1)),numCat - 1,numCrit)
      
      colnames(vetoes1) <- colnames(performanceTable)
      colnames(vetoes2) <- colnames(performanceTable)
      
      if(veto)
      {
        for(k in 1:(numCat - 1))
          for(j in 1:numCrit)
          {
            if(j <= pivot)
            {
              vetoes1[k,criteria[j]] <- parent1$vetoPerformances[k,criteria[j]]
              vetoes2[k,criteria[j]] <- parent2$vetoPerformances[k,criteria[j]]
            }
            else
            {
              vetoes1[k,criteria[j]] <- parent2$vetoPerformances[k,criteria[j]]
              vetoes2[k,criteria[j]] <- parent1$vetoPerformances[k,criteria[j]]
            }
          }
      }
      
      # child identical to first parent - will get mutated in the second step
      
      children[[length(children)+1]] <- list(majorityThreshold = parent1$majorityThreshold, criteriaWeights = parent1$criteriaWeights, profilesPerformances = parent1$profilesPerformances, vetoPerformances = parent1$vetoPerformances)
      
      # child identical to second parent
      
      children[[length(children)+1]] <- list(majorityThreshold = parent2$majorityThreshold, criteriaWeights = parent2$criteriaWeights, profilesPerformances = parent2$profilesPerformances, vetoPerformances = parent2$vetoPerformances)
      
      # child takes weights and threshold from first parent and profiles from second
      
      children[[length(children)+1]] <- list(majorityThreshold = parent1$majorityThreshold, criteriaWeights = parent1$criteriaWeights, profilesPerformances = parent2$profilesPerformances, vetoPerformances = parent2$vetoPerformances)
      
      # child takes weights and threshold from second parent and profiles from first
      
      children[[length(children)+1]] <- list(majorityThreshold = parent2$majorityThreshold, criteriaWeights = parent2$criteriaWeights, profilesPerformances = parent1$profilesPerformances, vetoPerformances = parent1$vetoPerformances)
      
      # child takes weights and threshold from first parent and profiles from first crossover
      
      children[[length(children)+1]] <- list(majorityThreshold = parent1$majorityThreshold, criteriaWeights = parent1$criteriaWeights, profilesPerformances = profiles1, vetoPerformances = vetoes1)
      
      # child takes weights and threshold from first parent and profiles from second crossover
      
      children[[length(children)+1]] <- list(majorityThreshold = parent1$majorityThreshold, criteriaWeights = parent1$criteriaWeights, profilesPerformances = profiles2, vetoPerformances = vetoes2)
      
      # child takes weights and threshold from second parent and profiles from first crossover
      
      children[[length(children)+1]] <- list(majorityThreshold = parent2$majorityThreshold, criteriaWeights = parent2$criteriaWeights, profilesPerformances = profiles1, vetoPerformances = vetoes1)
      
      # child takes weights from second parent and profiles from second crossover
      
      children[[length(children)+1]] <- list(majorityThreshold = parent2$majorityThreshold, criteriaWeights = parent2$criteriaWeights, profilesPerformances = profiles2, vetoPerformances = vetoes2)
    }
    
    # mutate children
    
    numChildren <- length(children)
    
    for(i in 1:numChildren)
    {
      if(runif(1,0,1) < mutationProb)
      {
        # mutate majority threshold
        
        children[[i]]$majorityThreshold <- runif(1,0.5,1)
      }
      
      for(j1 in 1:(numCrit-1))
      {
        for(j2 in (j1+1):numCrit)
        {
          if(runif(1,0,1) < mutationProb)
          {
            # mutate two criteria weights
            
            criteria <- c(colnames(performanceTable)[j1],colnames(performanceTable)[j2])
            
            minVal <- 0 - children[[i]]$criteriaWeights[criteria[1]]
            
            maxVal <- children[[i]]$criteriaWeights[criteria[2]]
            
            tradeoff <- runif(1,minVal,maxVal)
            
            children[[i]]$criteriaWeights[criteria[1]] <- children[[i]]$criteriaWeights[criteria[1]] + tradeoff
            
            children[[i]]$criteriaWeights[criteria[2]] <- children[[i]]$criteriaWeights[criteria[2]] - tradeoff
          }
        }
      }
      
      for(k in 1:(numCat - 1))
      {
        for(criterion in colnames(performanceTable))
        {
          if(runif(1,0,1) < mutationProb)
          {
            # mutate profile evaluation
            
            maxVal <- maxEvaluations[criterion]
            
            minVal <- minEvaluations[criterion]
            
            if(k < (numCat - 1))
            {
              if(criteriaMinMax[criterion] == 'max')
                minVal <- children[[i]]$profilesPerformances[k+1,criterion]
              else
                maxVal <- children[[i]]$profilesPerformances[k+1,criterion]
            }
            
            if(k > 1)
            {
              if(criteriaMinMax[criterion] == 'max')
                maxVal <- children[[i]]$profilesPerformances[k-1,criterion]
              else
                minVal <- children[[i]]$profilesPerformances[k-1,criterion]
            }
            
            if(veto)
            {
              if(criteriaMinMax[criterion] == 'max')
              {
                if(children[[i]]$vetoPerformances[k,criterion] %>=% minVal)
                  minVal <- children[[i]]$vetoPerformances[k,criterion] + 0.0000000001
              }
              else
              {
                if(children[[i]]$vetoPerformances[k,criterion] %<=% maxVal)
                  maxVal <- children[[i]]$vetoPerformances[k,criterion] - 0.0000000001
              }
            }
            
            children[[i]]$profilesPerformances[k,criterion] <- runif(1,minVal,maxVal)
          }
        }
      }
      
      if(veto)
      {
        for(k in 1:(numCat - 1))
        {
          for(criterion in colnames(performanceTable))
          {
            if(runif(1,0,1) < mutationProb)
            {
              # mutate one veto evaluation
              
              maxVal <- maxEvaluations[criterion]
              
              if(criteriaMinMax[criterion] == 'min')
                maxVal <- maxEvaluations[criterion] + 1
              
              minVal <- minEvaluations[criterion]
              
              if(criteriaMinMax[criterion] == 'max')
                minVal <- minEvaluations[criterion] - 1
              
              if(k < (numCat - 1))
              {
                if(criteriaMinMax[criterion] == 'max')
                  minVal <- children[[i]]$vetoPerformances[k+1,criterion]
                else
                  maxVal <- children[[i]]$vetoPerformances[k+1,criterion]
              }
              
              if(k > 1)
              {
                if(criteriaMinMax[criterion] == 'max')
                  maxVal <- children[[i]]$vetoPerformances[k-1,criterion]
                else
                  minVal <- children[[i]]$vetoPerformances[k-1,criterion]
              }
              
              if(criteriaMinMax[criterion] == 'max')
              {
                if(children[[i]]$profilesPerformances[k,criterion] %<=% maxVal)
                  maxVal <- children[[i]]$profilesPerformances[k,criterion] - 0.0000000001
              }
              else
              {
                if(children[[i]]$profilesPerformances[k,criterion] %>=% minVal)
                  minVal <- children[[i]]$profilesPerformances[k,criterion] + 0.0000000001
              }
              
              children[[i]]$vetoPerformances[k,criterion] <- runif(1,minVal,maxVal)
            }
          }
        }
      }
    }
    return(children)
  }
  
  # -------------------------------------------------------
  
  startTime <- Sys.time()
  
  # Initialize population
  
  population <- InitializePopulation()
  
  bestIndividual <- list(fitness = 0)
  
  # Main loop
  
  ct <- 0
  
  while(as.double(difftime(Sys.time(), startTime, units = 'secs')) < timeLimit)
  {
    # Evaluate population
    
    evaluations <- unlist(lapply(population, Fitness))
    
    # Store best individual if better than the overall best
    
    maxFitness <- max(evaluations)
    
    if(maxFitness >= bestIndividual$fitness)
    {
      bestIndividual <- population[[match(maxFitness,evaluations)]]
      
      bestIndividual$fitness <- maxFitness
    }
    
    # report
    
    if(as.double(difftime(Sys.time(), startTime, units = 'secs')) / 5 > ct)
    {
      ct <- ct + 1
      
      # print(sprintf("Best fitness so far: %6.2f%%", bestIndividual$fitness * 100))
    }
    
    # check if we are done
    
    if(bestIndividual$fitness == 1)
      break
    
    # Selection - not the first iteration
    
    if(length(population) > populationSize)
    {
      evaluations <- evaluations^2
      
      newPopulation <- list()
      
      newPopulation[[length(newPopulation)+1]] <- bestIndividual
      
      i <- 1
      
      while(length(newPopulation) < populationSize)
      {
        if(runif(1,0,1) <= evaluations[i])
        {
          evaluations[i] <- -1
          
          newPopulation[[length(newPopulation)+1]] <- population[[i]]
        }
        
        i <- i + 1
        
        if(i > length(population))
          i <- 1
        
      }
      
      population <- newPopulation
    }
    
    # Reproduction
    
    population <- Reproduce(population)
  }
  
  # print(sprintf("Final model fitness: %6.2f%%", bestIndividual$fitness * 100))
  
  # add dummy profiles
  
  bestIndividual$profilesPerformances <- rbind(bestIndividual$profilesPerformances,rep(NA,numCrit))
  
  bestIndividual$vetoPerformances <- rbind(bestIndividual$vetoPerformances,rep(NA,numCrit))
  
  rownames(bestIndividual$profilesPerformances) <- names(sort(categoriesRanks))
  
  rownames(bestIndividual$vetoPerformances) <- rownames(bestIndividual$profilesPerformances)
  
  # determine which vetoes are actually used and remove those that are simply an artefact of the metaheuristic
  
  used_vetoes <- MRSortIdentifyUsedVetoProfiles(performanceTable, assignments, sort(categoriesRanks), criteriaMinMax, bestIndividual$majorityThreshold, bestIndividual$criteriaWeights, bestIndividual$profilesPerformances, bestIndividual$vetoPerformances, alternativesIDs, criteriaIDs)
  
  for (k in (numCat-1):1)
  {
    for (j in 1:numCrit)
    {
      if (!used_vetoes[k,j])
        bestIndividual$vetoPerformances[k,j] <- NA
    }
  }
  
  return(bestIndividual)
}
