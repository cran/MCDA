# TOPSIS TO PYTHON
import pandas as pd
import math


def TOPSIS(
    performanceTable,
    criteriaWeights,
    criteriaMinMax,
    positiveIdealSolutions=None,
    negativeIdealSolutions=None,
    alternativesIDs=None,
    criteriaIDs=None,
):

    ## check the input data

    if not ((alternativesIDs is None) or isinstance(alternativesIDs, list)):
        raise ValueError("alternatives IDs should be in a vector")
    if not ((criteriaIDs is None) or isinstance(criteriaIDs, list)):
        raise ValueError("criteria IDs should be in a vector")
    if not ((performanceTable is None) or isinstance(performanceTable, pd.DataFrame)):
        raise ValueError("performanceTable must be a matrix or a data frame")
    if not (criteriaWeights.shape[1] == performanceTable.shape[1]):
        raise ValueError(
            "the number of criteriaWeights must equal the number of columns in the performanceTable"
        )
    if criteriaMinMax is None:
        raise ValueError("the input criteriaMinMax is required.")

    ## filter the performance table and the criteria according to the given alternatives and criteria

    if not (alternativesIDs is None):
        performanceTable = performanceTable.loc[alternativesIDs, :]

    if not (criteriaIDs is None):
        performanceTable = performanceTable[criteriaIDs]
        criteriaWeights = criteriaWeights[criteriaIDs]

        if not (positiveIdealSolutions is None):
            positiveIdealSolutions = positiveIdealSolutions[criteriaIDs]
        if not (negativeIdealSolutions is None):
            negativeIdealSolutions = negativeIdealSolutions[criteriaIDs]

        criteriaMinMax = criteriaMinMax[criteriaIDs]

    critno = criteriaWeights.shape[1]
    altno = performanceTable.shape[0]

    ## Calculate the weighted normalised matrix

    divby = list(range(critno))
    for i in range(critno):
        tmp = sum(performanceTable.iloc[:, i] ** 2)
        divby[i] = math.sqrt(tmp)

    divby = pd.DataFrame(divby)
    normalisedm = pd.DataFrame(
        performanceTable.transpose().values / divby.values,
        index=performanceTable.columns,
        columns=performanceTable.index,
    )
    normalisedm = normalisedm.transpose()
    wnm = pd.DataFrame(
        normalisedm.transpose().values * criteriaWeights.transpose().values,
        index=performanceTable.columns,
        columns=performanceTable.index,
    )
    wnm = wnm.transpose()

    ## Identify positive and negative ideal solutions

    pis = list(range(critno))
    nis = list(range(critno))

    if (positiveIdealSolutions is None) or (negativeIdealSolutions is None):

        for i in range(critno):

            if criteriaMinMax.iloc[0, i] == "max":

                pis[i] = max(wnm.iloc[:, i])
                nis[i] = min(wnm.iloc[:, i])

            else:

                pis[i] = min(wnm.iloc[:, i])
                nis[i] = max(wnm.iloc[:, i])

    else:

        ## check the input data is correct

        if not (
            len(positiveIdealSolutions) == len(negativeIdealSolutions)
            or len(positiveIdealSolutions) == critno
        ):
            raise ValueError(
                "the number of postive and negaitve ideal solutions need to equal the number of alternaitves."
            )

        pis = positiveIdealSolutions.values
        nis = negativeIdealSolutions.values

    ## Identify separation from positive and negative ideal solutions
    spis = wnm.subtract(pis) ** 2
    snis = wnm.subtract(nis) ** 2

    spisv = list(range(altno))
    snisv = list(range(altno))

    for i in range(altno):

        spisv[i] = math.sqrt(spis.iloc[i, :].sum())
        snisv[i] = math.sqrt(snis.iloc[i, :].sum())

    ## Calculate results
    results = list(range(altno))
    for i in range(altno):
        results[i] = snisv[i] / (snisv[i] + spisv[i])

    results = pd.DataFrame(results)
    results.index = performanceTable.index.values
    results.columns = ["Solution"]
    results = results.transpose()
    return results

