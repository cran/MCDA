# TOPSIS TO PYTHON
import pandas as pd


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

    if not ((alternativesIDs is None) or isinstance(alternativesIDs, pd.DataFrame)):
        raise ValueError("alternatives IDs should be in a vector")
    if not ((criteriaIDs is None) or isinstance(criteriaIDs, pd.DataFrame)):
        raise ValueError("criteria IDs should be in a vector")
    if not ((performanceTable is None) or isinstance(performanceTable, pd.DataFrame)):
        raise ValueError("performanceTable must be a matrix or a data frame")
    if not (len(criteriaWeights) == performanceTable.shape[1]):
        raise ValueError(
            "the number of criteriaWeights must equal the number of columns in the performanceTable"
        )
    if criteriaMinMax is None:
        raise ValueError("the input criteriaMinMax is required.")

    ## filter the performance table and the criteria according to the given alternatives and criteria

    if not (alternativesIDs is None):
        performanceTable = performanceTable[
            alternativesIDs,
        ]

    if not (criteriaIDs is None):
        performanceTable = performanceTable[criteriaIDs]
        criteriaWeights = criteriaWeights[criteriaIDs]

        if not (positiveIdealSolutions is None):
            positiveIdealSolutions = positiveIdealSolutions[criteriaIDs]
        if not (negativeIdealSolutions is None):
            negativeIdealSolutions = negativeIdealSolutions[criteriaIDs]

        criteriaMinMax = criteriaMinMax[criteriaIDs]

    critno = len(criteriaWeights)
    altno = len(performanceTable)

    ## Calculate the weighted normalised matrix

    divby = list(range(1, critno))
    for i in range(1, critno):

        divby[i] < -sqrt(sum(performanceTable[i] ^ 2))

    normalisedm = t(t(performanceTable) / divby)
    wnm = t(t(normalisedm) * criteriaWeights)

    ## Identify positive and negative ideal solutions

    pis = list(range(1, critno))
    nis = list(range(1, critno))

    if (positiveIdealSolutions is None) or (negativeIdealSolutions is None):

        for i in range(1, critno):

            if criteriaMinMax[i] == "max":

                pis[i] = max(wnm[i])
                nis[i] = min(wnm[i])

            else:

                pis[i] = min(wnm[i])
                nis[i] = max(wnm[i])
    else:

        ## check the input data is correct

        if not (
            len(positiveIdealSolutions) == len(negativeIdealSolutions)
            or len(positiveIdealSolutions) == critno
        ):
            raise ValueError(
                "the number of postive and negaitve ideal solutions need to equal the number of alternaitves."
            )

        pis = positiveIdealSolutions
        nis = negativeIdealSolutions

    ## Identify separation from positive and negative ideal solutions
    spis = sweep(wnm, 2, pis) ^ 2
    snis = sweep(wnm, 2, nis) ^ 2
    spisv = list(range(1, altno))
    snisv = list(range(1, altno))

    for i in range(1, altno):

        spisv[i] = sqrt(sum(spis[i,]))
        snisv[i] = sqrt(sum(snis[i,]))

    ## Calculate results
    results = list(range(1, altno))
    for i in range(1, altno):

        results[i] = snisv[i] / (snisv[i] + spisv[i])
    results.index = performanceTable.columns.vallues
    return results

