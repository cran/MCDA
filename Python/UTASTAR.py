# UTASTAR PYTHON
import pandas as pd
import numpy as np
from shutil import which
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


base = importr("base")
utils = importr("utils")
# utils.install_packages('Rglpk')


def UTASTAR(
    performanceTable,
    criteriaMinMax,
    criteriaNumberOfBreakPoints,
    epsilon,
    alternativesRanks=None,
    alternativesPreferences=None,
    alternativesIndifferences=None,
    criteriaLBs=None,
    criteriaUBs=None,
    alternativesIDs=None,
    criteriaIDs=None,
    kPostOptimality=None,
):

    # check the input data
    if not isinstance(performanceTable, pd.DataFrame):
        raise ValueError("wrong performanceTable, should be a pandas data frame")
    if not ((alternativesRanks is None) or isinstance(alternativesRanks, pd.DataFrame)):
        raise ValueError("alternativesRanks should be a pandas data frame")
    if not (
        (alternativesPreferences is None)
        or isinstance(alternativesPreferences, pd.DataFrame)
    ):
        raise ValueError("alternativesPreferences should be a matrix")
    if not (
        (alternativesIndifferences is None)
        or isinstance(alternativesIndifferences, pd.DataFrame)
    ):
        raise ValueError("alternativesIndifferences should be a matrix")
    if (
        (alternativesRanks is None)
        and (alternativesPreferences is None)
        and (alternativesIndifferences is None)
    ):
        raise ValueError(
            "at least one of alternativesRanks, alternativesPreferences or"
            + " alternativesIndifferences should not be NULL"
        )
    if not (alternativesRanks is None) and (
        not (alternativesPreferences is None) or not (alternativesIndifferences is None)
    ):
        raise ValueError(
            "alternativesRanks and one of alternativesPreferences or "
            + " alternativesIndifferences cannot be simultaneously not NULL"
        )
    if not (isinstance(criteriaMinMax, pd.DataFrame)):
        raise ValueError("criteriaMinMax should be a vector")
    if not (isinstance(criteriaNumberOfBreakPoints, pd.DataFrame)):
        raise ValueError("criteriaNumberOfBreakPoints should be a vector")
    if not ((alternativesIDs is None) or isinstance(alternativesIDs, list)):
        raise ValueError("alternativesIDs should be in a vector")
    if not ((criteriaIDs is None) or isinstance(criteriaIDs, list)):
        raise ValueError("criteriaIDs should be in a vector")
    if not ((criteriaLBs is None) or isinstance(criteriaLBs, list)):
        raise ValueError("criteriaLBs should be in a vector")
    if not ((criteriaUBs is None) or isinstance(criteriaUBs, list)):
        raise ValueError("criteriaUBs should be in a vector")

    # filter the data according to the given alternatives and criteria
    # in alternativesIDs and criteriaIDs
    if not (alternativesIDs is None):
        performanceTable = performanceTable.loc[
            alternativesIDs,
        ]
        if not (alternativesRanks is None):
            alternativesRanks = alternativesRanks.loc[:, alternativesIDs]
        if not (alternativesPreferences is None):
            tmpIds = alternativesPreferences[
                alternativesPreferences.isin(alternativesIDs)
            ]
            tmpIds = tmpIds.values.tolist()
            tmpIds = [[y for y in x if str(y) != "nan"] for x in tmpIds]
            flat_list = []
            for sublist in tmpIds:
                for item in sublist:
                    flat_list.append(item)
            tmpIds = flat_list
            tmpMatrix = pd.DataFrame()

            for i in range(1, len(alternativesPreferences)):  # for num of rows
                # append probably wrong TODO CHECK
                if (
                    alternativesPreferences.loc[i,].all in tmpIds
                ):  # if all values in list are true(all row exists in tmpides)
                    tmpMatrix = tmpMatrix.append(
                        alternativesPreferences[i,], ignore_index=True
                    )  # append row
            alternativesPreferences = tmpMatrix
        if not (alternativesIndifferences is None):
            tmpIds = alternativesIndifferences[
                alternativesIndifferences.isin(alternativesIDs)
            ]
            tmpIds = tmpIds.values.tolist()
            tmpIds = [[y for y in x if str(y) != "nan"] for x in tmpIds]
            flat_list = []
            for sublist in tmpIds:
                for item in sublist:
                    flat_list.append(item)
            tmpIds = flat_list
            tmpMatrix = pd.DataFrame()
            for i in range(1, alternativesIndifferences.shape[0]):
                # append probably wrong TODO CHECK
                if alternativesIndifferences.loc[i,].all in tmpIds:
                    tmpMatrix = tmpMatrix.append(
                        alternativesIndifferences.loc[i,], ignore_index=True
                    )
            alternativesIndifferences = tmpMatrix

    if not (criteriaIDs is None):
        criteriaMinMax = criteriaMinMax[criteriaIDs]
        performanceTable = performanceTable.loc[
            :, performanceTable.columns.isin(criteriaIDs)
        ]
        criteriaNumberOfBreakPoints = criteriaNumberOfBreakPoints[criteriaIDs]
    if not (criteriaIDs is None) and not (criteriaUBs is None):
        criteriaUBs = criteriaUBs[criteriaIDs]
    if not (criteriaIDs is None) and not (criteriaLBs is None):
        criteriaLBs = criteriaLBs[criteriaIDs]

    # only the alternatives which are in the ranking should be considered for
    #  the calculation
    # we therefore take the intersection between the alternatives present in
    #  the performance
    # table and those of the ranking

    if not (alternativesRanks is None):
        # intersection of rownames and names
        reallyActiveAlternatives = [
            x
            for x in performanceTable.index.values
            if x in alternativesRanks.columns.values
        ]

        if len(reallyActiveAlternatives) != 0:
            performanceTable = performanceTable.loc[
                reallyActiveAlternatives,
            ]
            alternativesRanks = alternativesRanks[reallyActiveAlternatives]

        else:
            raise ValueError(
                "alternatives of alternativesRanks are not compatible with those of performanceTable"
            )

    # TODO replace for loops with merge or isin
    if not (alternativesPreferences is None) or not (alternativesIndifferences is None):

        tmp = alternativesPreferences.append(
            alternativesIndifferences, ignore_index=True
        )
        # reallyActiveAlternatives = tmp[tmp.isin(performanceTable.index.values)]

        reallyActiveAlternatives = [
            x for x in performanceTable.index.values for y in tmp.values if any(y == x)
        ]
        reallyActiveAlternatives = list(dict.fromkeys(reallyActiveAlternatives))

        if len(reallyActiveAlternatives) != 0:
            performanceTable = performanceTable.loc[
                reallyActiveAlternatives,
            ]

            if not (alternativesPreferences is None):
                tmpIds = [
                    x
                    for x in reallyActiveAlternatives
                    for y in alternativesPreferences.values
                    if any(x == y)
                ]
                tmpIds = list(dict.fromkeys(tmpIds))
                # tmpIds = alternativesPreferences[alternativesPreferences.isin(reallyActiveAlternatives)]

                tmpMatrix = pd.DataFrame()
                for i in range(alternativesPreferences.shape[0]):
                    if all(alternativesPreferences.iloc[i, :].isin(tmpIds)):
                        tmpMatrix = tmpMatrix.append(
                            alternativesPreferences.iloc[i, :], ignore_index=True
                        )

                alternativesPreferences = tmpMatrix
            if not (alternativesIndifferences is None):
                tmpIds = [
                    x
                    for x in reallyActiveAlternatives
                    for y in alternativesIndifferences.values
                    if any(x == y)
                ]
                tmpIds = list(dict.fromkeys(tmpIds))
                # tmpIds = alternativesIndifferences.intersection(reallyActiveAlternatives)
                # tmpIds = alternativesIndifferences[
                #     alternativesIndifferences.isin(reallyActiveAlternatives)]

                tmpMatrix = pd.DataFrame()

                for i in range(alternativesIndifferences.shape[0]):
                    if all(alternativesIndifferences.iloc[i, :].isin(tmpIds)):
                        tmpMatrix = tmpMatrix.append(
                            alternativesIndifferences.iloc[i, :], ignore_index=True
                        )

                alternativesIndifferences = tmpMatrix
            else:
                raise ValueError(
                    "alternatives of alternativesPreferences or alternativesIndifferences are not compatible with those of performanceTable"
                )

    # data is filtered, check for some data consistency

    # are the upper and lower bounds given in the function compatible with the
    #  data in the performance table ?

    if not (criteriaUBs is None):
        if not all(
            performanceTable.apply(max, axis=1) <= criteriaUBs
        ):  # maybe is 1 beacuse of column
            raise ValueError("performanceTable contains higher values than criteriaUBs")

    if not (criteriaLBs is None):
        if not all(performanceTable.apply(min, axis=1) >= criteriaLBs):
            raise ValueError("performanceTable contains lower values than criteriaLBs")

    if not all(criteriaNumberOfBreakPoints >= 2):
        raise ValueError(
            "in criteriaNumberOfBreakPoints there should at least be 2 breakpoints for each criterion"
        )

    # if there are less than 2 criteria or 2 alternatives, there is no MCDA
    #  problem
    if performanceTable.shape[0] < 2 and performanceTable.shape[1 < 2]:
        raise ValueError("less than 2 criteria or 2 alternatives")

    # if there are no alternatives left in the ranking or the pairwise
    #  preferences
    # we stop here

    if (
        (alternativesRanks is None)
        and (alternativesPreferences is None)
        and (alternativesIndifferences is None)
    ):
        raise ValueError(
            "after filtering none of alternativesRanks, alternativesPreferences or alternativesIndifferences is not NULL"
        )

    # -------------------------------------------------------
    # number of Criteria
    numCrit = performanceTable.shape[1]
    # number of Alternatives
    numAlt = performanceTable.shape[0]
    # -------------------------------------------------------

    criteriaBreakPoints = pd.DataFrame()

    for i in range(numCrit):
        tmp = list()
        if not (criteriaLBs is None):
            mini = criteriaLBs[i]
        else:
            mini = performanceTable.iloc[:, i].min(axis=0)

        if not (criteriaLBs is None):
            maxi = criteriaUBs[i]
        else:
            maxi = performanceTable.iloc[:, i].max(axis=0)

        if mini == maxi:
            # then there is only one value for that criterion, and the algorithm
            #  to build the linear interpolation
            # will not work correctly
            raise ValueError(
                "there is only one possible value left for criterion ",
                performanceTable.iloc[:, i].values,
            )

        alphai = criteriaNumberOfBreakPoints.iloc[0, i]

        for j in range(1, alphai + 1):
            tmpnum = int(mini + (j - 1) / (alphai - 1) * (maxi - mini))
            tmp.append(tmpnum)


        # due to this formula, the minimum and maximum values might not be exactly
        #  the same than the real minimum and maximum values in the performance table
        # to be sure there is no rounding problem, we recopy these values in tmp
        #  (important for the later comparisons to these values)

        tmp[0] = mini
        tmp[alphai - 1] = maxi

        # if the criterion has to be maximized, the worst value is in the first
        #  position
        # else, we sort the vector the other way around to have the worst value
        #  in the first position

        if criteriaMinMax.iloc[0, i] == "min":
            tmp.sort(reverse=True)

        #criteriaBreakPoints.append(tmp)
        ltmp = pd.DataFrame(tmp)
        criteriaBreakPoints = pd.concat([criteriaBreakPoints,ltmp],axis = 1 )

    criteriaBreakPoints.columns = performanceTable.columns.values 
    
    ##try to use this for columns 
    numofBPSum = [x-1 for x in criteriaNumberOfBreakPoints.values  ]
    numofBPSum = pd.DataFrame(numofBPSum)
    numofBPSum = numofBPSum.values.sum()
    # -------------------------------------------------------
    # a is a matrix decomposing the alternatives in the break point space and
    #  adding the sigmaPlus and sigmaMinus columns

    a = pd.DataFrame(
        0,
        index=range(numAlt),
        columns=range(criteriaNumberOfBreakPoints.values.sum() + 2 * numAlt),
    )

    # TODO write better code replace for loops for index finding
    #   replace try catch with if/else
    for n in range(numAlt):
        for m in range(numCrit):
            j = [int(x) for x in range(len(criteriaBreakPoints.iloc[:,m].dropna(axis=0).values)) if performanceTable.iloc[n,m] ==  criteriaBreakPoints.iloc[x,m]]
            if (len(j)!=0):
                # then we have a performance value which is on a breakpoint
                j = int(j[0])
                
                if m == 0:
                    pos = j 
                else:
                    pos = criteriaNumberOfBreakPoints.iloc[0, :m].sum() + j  
                
                a.iloc[n, pos] = 1
            else:

                # then we have value which needs to be approximated by a
                #  linear interpolation
                # let us first search the lower and upper bounds of the
                # interval of breakpoints around the value

                if criteriaMinMax.iloc[0, m] == "min":
                    j = [
                        x
                        for x in range(len(criteriaBreakPoints.iloc[:, m].dropna(axis=0).values))
                        if performanceTable.iloc[n, m] > criteriaBreakPoints.iloc[x, m]
                    ]
                    j = int(j[0])
                else:
                    j = [
                        x
                        for x in range(len(criteriaBreakPoints.iloc[:, m].dropna(axis=0).values))
                        if performanceTable.iloc[n, m] < criteriaBreakPoints.iloc[x, m]
                    ]
                    j = int(j[0])


                if m == 0:
                    pos = j
                else:
                    pos = criteriaNumberOfBreakPoints.iloc[0, :m].sum() + j 
                
                j = j - 1 
                pos = pos - 1

                a.iloc[n, pos] = 1 - (
                    performanceTable.iloc[n, m] - criteriaBreakPoints.iloc[j, m]
                ) / (
                    criteriaBreakPoints.iloc[j+1, m]
                    - criteriaBreakPoints.iloc[j, m]
                )
                a.iloc[n, pos + 1] = (
                    performanceTable.iloc[n, m] - criteriaBreakPoints.iloc[j, m]
                ) / (
                    criteriaBreakPoints.iloc[j+1, m]
                    - criteriaBreakPoints.iloc[j, m]
                )

            # and now for sigmaPlus
            sigmapos = a.shape[1] - 2 * numAlt + n
            a.iloc[n, sigmapos] = -1
            # and sigmaMinus
            sigmapos = a.shape[1] - numAlt + n
            a.iloc[n, sigmapos] = +1
    
    # -------------------------------------------------------

    # the objective function : the first elements correspond to the ui's
    # , the last one to the sigmas

    obj1 = pd.DataFrame(
        0, index=range(1), columns=range(criteriaNumberOfBreakPoints.values.sum())
    )
    obj2 = pd.DataFrame(1, index=range(1), columns=range(2 * numAlt))

    obj = pd.concat([obj1, obj2], axis=1,)
    obj = obj.reset_index(drop=True,)
    # -------------------------------------------------------

    # we now build the part of the constraints matrix concerning the order
    #  / preferences / indifferences given by the decision maker

    preferenceConstraints = pd.DataFrame(
        index=range(1),
        columns=range(criteriaNumberOfBreakPoints.values.sum() + 2 * numAlt),
    )
    indifferenceConstraints = pd.DataFrame(
        index=range(1),
        columns=range(criteriaNumberOfBreakPoints.values.sum() + 2 * numAlt),
    )

    if not (alternativesRanks is None):
        # determine now in which order the alternatives should be treated for the constraints
        indexOrder = list()
        orderedAlternativesRanks = alternativesRanks.sort_values(by=0, axis=1)

        while orderedAlternativesRanks.empty != True:
            # search for the alternatives of lowest rank
            tmpIndex = [
                x
                for x in range(alternativesRanks.shape[1])
                if alternativesRanks.iloc[0, x] == orderedAlternativesRanks.iloc[0, 0]
            ]

            for j in range(0, len(tmpIndex)):
                indexOrder.append(tmpIndex[j])
            # remove the rank which has been dealt with now
            tmpordindexpos = [
                x
                for x in range(orderedAlternativesRanks.shape[1])
                if orderedAlternativesRanks.iloc[0, x]
                == orderedAlternativesRanks.iloc[0, 0]
            ]
            orderedAlternativesRanks = orderedAlternativesRanks.drop(
                orderedAlternativesRanks.columns[tmpordindexpos], axis="columns"
            )

        for i in range(alternativesRanks.shape[1] - 1):
            if (
                alternativesRanks.iloc[0, indexOrder[i]]
                == alternativesRanks.iloc[0, indexOrder[i + 1]]
            ):
                # then the alternatives are indifferent and their overall values are equal
                indifferenceConstraints = indifferenceConstraints.append(
                    a.iloc[indexOrder[i],] - a.iloc[indexOrder[i + 1],],
                    ignore_index=True,
                )
            else:
                # then the first alternative i is ranked better than the second one i+1 and i has an overall value higher than i+1
                preferenceConstraints = preferenceConstraints.append(
                    a.iloc[indexOrder[i],] - a.iloc[indexOrder[i + 1],],
                    ignore_index=True,
                )

    indifferenceConstraints = indifferenceConstraints.dropna()
    preferenceConstraints = preferenceConstraints.dropna()
    if not (alternativesPreferences is None):
        altranktmp = pd.DataFrame(performanceTable.index.values)
        for i in range(alternativesPreferences.shape[0]):
            ltmp1 = a.iloc[
                [
                    x
                    for x in range(len(performanceTable.index.values))
                    if altranktmp.iloc[x, 0] == alternativesPreferences.iloc[i, 0]
                ],
            ]
            ltmp2 = a.iloc[
                [
                    x
                    for x in range(len(performanceTable.index.values))
                    if altranktmp.iloc[x, 0] == alternativesPreferences.iloc[i, 1]
                ],
            ]
            tmp = ltmp1.values - ltmp2.values
            tmp = pd.DataFrame(tmp)
            preferenceConstraints = preferenceConstraints.append(
                tmp, ignore_index=True,
            )
    if not (alternativesIndifferences is None):
        altranktmp = pd.DataFrame(performanceTable.index.values)
        for i in range(alternativesIndifferences.shape[0]):
            ltmp1 = a.iloc[
                [
                    x
                    for x in range(len(performanceTable.index.values))
                    if altranktmp.iloc[x, 0] == alternativesIndifferences.iloc[i, 0]
                ],
            ]
            ltmp2 = a.iloc[
                [
                    x
                    for x in range(len(performanceTable.index.values))
                    if altranktmp.iloc[x, 0] == alternativesIndifferences.iloc[i, 1]
                ],
            ]
            tmp = ltmp1.values - ltmp2.values
            tmp = pd.DataFrame(tmp)
            indifferenceConstraints = indifferenceConstraints.append(
                tmp, ignore_index=True,
            )

    # add this to the constraints matrix mat
    mat = preferenceConstraints.append(indifferenceConstraints, ignore_index=True)
    # right hand side of this part of mat
    pd.DataFrame

    rhs = list()

    if preferenceConstraints.shape[0] != 0:
        for i in range(preferenceConstraints.shape[0]):
            rhs.append(epsilon)

    if indifferenceConstraints.shape[0] != 0:
        for i in range(indifferenceConstraints.shape[0]):
            rhs.append(0)

    # direection of the inequality for this part of mat

    dire = list()

    if preferenceConstraints.shape[0] != 0:
        for i in range(preferenceConstraints.shape[0]):
            dire.append(">=")

    if indifferenceConstraints.shape[0] != 0:
        for i in range(indifferenceConstraints.shape[0]):
            dire.append("==")
    # -------------------------------------------------------

    # now the monotonicity constraints on the value functions

    monotonicityConstraints = pd.DataFrame(
        index=range(1),
        columns=range(criteriaNumberOfBreakPoints.values.sum() + 2 * numAlt),
    )

    for i in range(criteriaNumberOfBreakPoints.shape[1]):
        for j in range((criteriaNumberOfBreakPoints.iloc[0, i] - 1)):
            tmp = pd.DataFrame(
                0,
                index=range(1),
                columns=range(criteriaNumberOfBreakPoints.values.sum() + 2 * numAlt),
            )
            tmp = tmp.dropna()
            if i == 0:
                pos = j
            else:
                pos = criteriaNumberOfBreakPoints.iloc[0, 0:(i)].sum() + j
            tmp.iloc[0, pos] = -1
            tmp.iloc[0, pos + 1] = 1
            monotonicityConstraints = monotonicityConstraints.append(
                tmp, ignore_index=True
            )

    # add this to the constraints matrix mat

    mat = mat.append(monotonicityConstraints, ignore_index=True)
    mat = mat.dropna()
    # the direction of the inequality

    # TODO CHECK -1 VALIDITY # Update seems fine recheck later
    for i in range(monotonicityConstraints.shape[0] - 1):
        dire.append(">=")

    # the right hand side of this part of mat

    for i in range(monotonicityConstraints.shape[0] - 1):
        rhs.append(0)

    # -------------------------------------------------------

    # normalization constraint for the upper values of the value functions (sum = 1)

    tmp = pd.DataFrame(
        0,
        index=range(1),
        columns=range(criteriaNumberOfBreakPoints.values.sum() + 2 * numAlt),
    )

    for i in range(criteriaNumberOfBreakPoints.shape[1]):
        if i == 0:
            pos = criteriaNumberOfBreakPoints.iloc[0, i]
        else:
            pos = (
                criteriaNumberOfBreakPoints.iloc[0, 0:(i)].sum()
                + criteriaNumberOfBreakPoints.iloc[0, i]
            )
        tmp.iloc[0, pos - 1] = 1
        # TODO CHECK -1 VALIDITY # Update seems ok recheck later

    # add this to the constraints matrix mat

    mat = mat.append(tmp, ignore_index=True)
    # the direection of the inequality

    dire.append("==")

    # the right hand side of this part of mat

    rhs.append(1)

    # -------------------------------------------------------

    # now the normalizaiton constraints for the lower values of the value functions (= 0)

    minValueFunctionsConstraints = pd.DataFrame(
        index=range(0),
        columns=range(criteriaNumberOfBreakPoints.values.sum() + 2 * numAlt),
    )

    for i in range(criteriaNumberOfBreakPoints.shape[1]):
        tmp = pd.DataFrame(
            0,
            index=range(1),
            columns=range(criteriaNumberOfBreakPoints.values.sum() + 2 * numAlt),
        )
        if i == 0:
            pos = i
        else:
            pos = criteriaNumberOfBreakPoints.iloc[
                0, 0:(i)
            ].sum()  # TODO +1 removal validity # Update seems ok recheck later
        tmp.iloc[0, pos] = 1
        minValueFunctionsConstraints = minValueFunctionsConstraints.append(tmp)

    # add this to the constraints matrix mat

    mat = mat.append(minValueFunctionsConstraints, ignore_index=True)

    # the direection of the inequality

    for i in range(minValueFunctionsConstraints.shape[0]):
        dire.append("==")

    # the right hand side of this part of mat

    for i in range(minValueFunctionsConstraints.shape[0]):
        rhs.append(0)

    # -------------------------------------------------------

    # TODO check if needed
    # pandas2ri.activate()

    # convert to list
    robj = [x for x in obj.iloc[0, :]]

    obj = pd.DataFrame(robj)

    # Transfromation of arrays various combinations
    # Transforms objects to R type arrays

    robj = robjects.IntVector(robj)
    rdire = robjects.StrVector(dire)
    rrhs = robjects.FloatVector(rhs)

    matrows = len(mat)
    rmat = mat.transpose()
    rmat = rmat.stack().values
    rmat = robjects.r.matrix(robjects.FloatVector(rmat), nrow=matrows)

    # LP calculation
    rglpk = importr("Rglpk")
    lpSolution = rglpk.Rglpk_solve_LP(robj, rmat, rdire, rrhs)

    # TODO FIX CODE REMOVE UGLY CODE recheck r2py
    # lpSolution = base.summary(lpSolution)
    # lpSolution = pandas2ri.ri2py_listvector(lpSolution)
    # lpSolution = robjects.r.matrix(robjects.Vector(lpSolution))

    optimum = lpSolution.rx2("optimum")
    optimum = pandas2ri.ri2py_vector(optimum).tolist()
    optimum = pd.DataFrame({"Optimum": optimum}, dtype="float")
    optimum.index = ["->"]

    solution = lpSolution.rx2("solution")
    solution = pandas2ri.ri2py_vector(solution).tolist()
    solution = pd.DataFrame({"Solution": solution})

    status = lpSolution.rx2("status")
    status = pandas2ri.ri2py_vector(status).tolist()
    status = pd.DataFrame({"Status": status})

    solution_dual = lpSolution.rx2("solution_dual")
    solution_dual = pandas2ri.ri2py_vector(solution_dual).tolist()
    solution_dual = pd.DataFrame({"Solution_dual": solution_dual})

    auxiliary = lpSolution.rx2("auxiliary")
    auxiliary_primal = auxiliary.rx2("primal")
    auxiliary_primal = pandas2ri.ri2py_vector(auxiliary_primal).tolist()
    auxiliary_primal = pd.DataFrame(
        {"Auxiliary_primal": auxiliary_primal}, index=range(len(auxiliary_primal))
    )

    auxiliary_dual = auxiliary.rx2("dual")
    auxiliary_dual = pandas2ri.ri2py_vector(auxiliary_dual).tolist()
    auxiliary_dual = pd.DataFrame(
        {"Auxiliary_dual": auxiliary_dual}, index=range(len(auxiliary_dual))
    )

    # TODO implement if nesecary
    # sensitivity_report = lpSolution.rx2("sensitivity_report")
    # sensitivity_report = pandas2ri.ri2py_vector(sensitivity_report)
    # if ( sensitivity_report != 'NA'):
    #     sensitivity_report = sensitivity_report.tolist()

    # NOT USED as lpSolution
    lpSolution = pd.concat(
        [optimum, solution, status, solution_dual, auxiliary_primal, auxiliary_dual],
        axis=1,
    )
    lpSolution = pd.DataFrame(lpSolution)

    # -------------------------------------------------------

    # create a structure containing the value functions

    valueFunctions = pd.DataFrame()
    #  TODO check if needed
    # (index=['x','y'],columns=  performanceTable.columns.values)

    for i in range(criteriaNumberOfBreakPoints.shape[1]):
        tmp = pd.DataFrame()
        ltmp = list()
        if i == 0:
            pos = 0
        else:
            pos = criteriaNumberOfBreakPoints.iloc[0, :i].sum()

        for j in range(criteriaNumberOfBreakPoints.iloc[0, i]):
            ltmp.append(solution.iloc[pos + j, 0])

        tmp = pd.DataFrame(
            {
                performanceTable.columns[i] + "X": criteriaBreakPoints.iloc[:, i].dropna(axis=0).values,
                performanceTable.columns[i] + "Y": ltmp,
            }
        )
        valueFunctions = pd.concat([valueFunctions, tmp], axis=1)

    valueFunctions = valueFunctions.transpose()

    # TODO Check if need columns names must be seperate
    # valueFunctions.columns = performanceTable.columns.values

    # it might happen on certain computers that these value functions
    # do NOT respect the monotonicity constraints (especially because of too small differences and computer arithmetics)
    # therefore we check if they do, and if not, we "correct" them

    for i in range(1, numCrit + 1):
        for j in range(
            criteriaNumberOfBreakPoints.iloc[0, i - 1] - 1
        ):  # TODO check -1 Validity # Upd seems ok recheck later

            if (
                valueFunctions.iloc[i * 2 - 1, j]
                > valueFunctions.iloc[i * 2 - 1, j + 1]
            ):
                valueFunctions.iloc[i * 2 - 1, j + 1] = valueFunctions.iloc[
                    i * 2 - 1, j
                ]

    # -------------------------------------------------------

    tmp = solution.iloc[0 : criteriaNumberOfBreakPoints.values.sum()]
    tmp2 = a.iloc[:, 0 : criteriaNumberOfBreakPoints.values.sum()]
    overallValues = pd.DataFrame(tmp2.dot(tmp))

    overallValues = overallValues.transpose()
    overallValues.columns = performanceTable.index.values
    overallValues.index = ["overallValues"]

    # -------------------------------------------------------

    # the error values for each alternative (sigma)

    errorValuesPlus = pd.DataFrame(
        solution.iloc[
            (criteriaNumberOfBreakPoints.values.sum()) : (
                criteriaNumberOfBreakPoints.values.sum() + numAlt
            ),
            0,
        ]
    )
    errorValuesMinus = pd.DataFrame(
        solution.iloc[
            (criteriaNumberOfBreakPoints.values.sum() + numAlt) : (
                criteriaNumberOfBreakPoints.values.sum() + 2 * numAlt
            ),
            0,
        ]
    )

    errorValuesPlus.index = performanceTable.index.values
    errorValuesMinus.index = performanceTable.index.values

    errorValuesMinus = errorValuesMinus.transpose()
    errorValuesPlus = errorValuesPlus.transpose()
    errorValuesMinus.index = ["ErrorValuesMinus"]
    errorValuesPlus.index = ["ErrorValuesPlus"]

    # -------------------------------------------------------

    # the ranks of the alternatives

    outRanks = overallValues.rank(method="min", axis=1, ascending=False)
    outRanks.index = ["outRanks"]

    # -------------------------------------------------------
    if (numAlt >= 3) and not (alternativesRanks is None):
        tau = alternativesRanks.iloc[0, :].corr(outRanks.iloc[0, :], method="kendall")
    else:
        tau = None

    tau = pd.DataFrame({"Kendall": [tau]})
    tau.index = ["tau"]
    # prepare the output

    # Output used to be in a single frame
    # out = pd.DataFrame(
    #     {
    #         "optimum": optimum.round(5),
    #         "valueFunctions": valueFunctions,
    #         "overallValues": overallValues.round(5),
    #         "ranks": outRanks,
    #         "errorssigmaPlus": errorValuesPlus.round(5),
    #         "errorssigmaMinus": errorValuesMinus.round(5),
    #         "Kendall": tau,
    #     }
    # )

    # TODO  WORKS BUT BETTER RETURN Multilpe dataframes
    out = pd.concat(
        [
            optimum.round(5),
            valueFunctions,
            overallValues.round(5),
            outRanks,
            errorValuesPlus.round(5),
            errorValuesMinus.round(5),
            tau,
        ],
        axis=1,sort=True
    )

    # -------------------------------------------------------

    # post-optimality analysis if the optimum is found and if kPostOptimality is not NULL, i.e. the solution space is not empty

    minWeights = None
    maxWeights = None
    averageValueFunctions = None
    averageOverallValues = None

    if not (kPostOptimality is None) and (optimum.iloc[0, 0] == 0):
        # add F \leq F* + k(F*) to the constraints, where F* is the optimum and k(F*) is a positive threshold, which is a small proportion of F*
        mat = mat.append(obj.transpose(), ignore_index=True)
        dire.append("<=")
        rhs.append(kPostOptimality)

        minWeights = pd.DataFrame()
        maxWeights = pd.DataFrame()
        combinedSolutions = pd.DataFrame()

        for i in range(numCrit):
            # first maximize the best ui for each criterion, then minimize it
            # this gives the interval of variation for each weight
            # the objective function : the first elements correspond to the ui's, the last one to the sigmas

            obj1 = pd.DataFrame(
                0,
                index=range(1),
                columns=range(criteriaNumberOfBreakPoints.values.sum()),
            )
            obj2 = pd.DataFrame(0, index=range(1), columns=range(2 * numAlt))

            obj = pd.concat([obj1, obj2], axis=1,)

            if i == 0:
                pos = criteriaNumberOfBreakPoints.iloc[0, i]
            else:
                pos = (
                    criteriaNumberOfBreakPoints.iloc[0, 0:i].sum()
                    + criteriaNumberOfBreakPoints.iloc[0, i]
                )

            obj.iloc[0, pos - 1] = 1

            ##Transformation code for LP
            robj = [x for x in obj.iloc[0, :]]
            robj = robjects.IntVector(robj)
            rdire = robjects.StrVector(dire)
            rrhs = robjects.FloatVector(rhs)

            matrows = len(mat)
            rmat = mat.transpose()
            rmat = rmat.stack().values
            rmat = robjects.r.matrix(robjects.FloatVector(rmat), nrow=matrows)

            # LP
            lpSolutionMin = rglpk.Rglpk_solve_LP(robj, rmat, rdire, rrhs)
            lpSolutionMax = rglpk.Rglpk_solve_LP(robj, rmat, rdire, rrhs, max=True)

            # Back to known types pandas dataframes
            optimumMin = lpSolutionMin.rx2("optimum")
            optimumMin = pandas2ri.ri2py_vector(optimumMin).tolist()
            optimumMin = pd.DataFrame({"Optimum": optimumMin}, dtype="float")

            optimumMax = lpSolutionMax.rx2("optimum")
            optimumMax = pandas2ri.ri2py_vector(optimumMax).tolist()
            optimumMax = pd.DataFrame({"Optimum": optimumMax}, dtype="float")

            solutionMin = lpSolutionMin.rx2("solution")
            solutionMin = pandas2ri.ri2py_vector(solutionMin).tolist()
            solutionMin = pd.DataFrame({"Solution": solutionMin})
            solutionMin = solutionMin.transpose()

            solutionMax = lpSolutionMax.rx2("solution")
            solutionMax = pandas2ri.ri2py_vector(solutionMax).tolist()
            solutionMax = pd.DataFrame({"Solution": solutionMax})
            solutionMax = solutionMax.transpose()

            minWeights = minWeights.append(optimumMin, ignore_index=True)
            maxWeights = maxWeights.append(optimumMax, ignore_index=True)
            combinedSolutions = combinedSolutions.append(solutionMin, ignore_index=True)
            combinedSolutions = combinedSolutions.append(solutionMax, ignore_index=True)

        minWeights.index = performanceTable.columns.values
        maxWeights.index = performanceTable.columns.values
        minWeights = minWeights.transpose()
        maxWeights = maxWeights.transpose()
        minWeights.index = ["minWeights"]
        maxWeights.index = ["maxWeights"]

        # calculate the average value function, for which each component is the average value obtained for each of the programs above
        averageSolution = pd.DataFrame(combinedSolutions.mean())
        averageSolution = averageSolution.transpose()

        # create a structure containing the average value functions

        averageValueFunctions = pd.DataFrame()

        for i in range(criteriaNumberOfBreakPoints.shape[1]):
            ltmp = list()
            if i == 0:
                pos = 0
            else:
                pos = criteriaNumberOfBreakPoints.iloc[0, 0:i].sum()
            for j in range(criteriaNumberOfBreakPoints.iloc[0, i]):
                ltmp.append(averageSolution.iloc[0, pos + j])

            tmp = pd.DataFrame(
                {
                    performanceTable.columns[i] + "X": criteriaBreakPoints.iloc[:, i].dropna(axis=0).values,
                    performanceTable.columns[i] + "Y": ltmp,
                }
            )
            averageValueFunctions = pd.concat([averageValueFunctions, tmp], axis=1)

        averageValueFunctions = averageValueFunctions.transpose()
        # TODO re calculate overall values after post optimality

        tmp = averageSolution.iloc[:, 0 : criteriaNumberOfBreakPoints.values.sum()]
        tmp = tmp.transpose()
        tmp2 = a.iloc[:, 0 : criteriaNumberOfBreakPoints.values.sum()]
        averageOverallValues = pd.DataFrame(tmp2.dot(tmp))

        averageOverallValues = averageOverallValues.transpose()
        averageOverallValues.columns = performanceTable.index.values
        averageOverallValues.index = ["averageOverallValues"]


    #ouput no used TODO output as a single variable dataframe ?

    out = pd.concat(
        [out, minWeights, maxWeights, averageValueFunctions,averageOverallValues], axis=1, sort=True
    )

    return (
        optimum,
        valueFunctions,
        overallValues,
        outRanks,
        errorValuesPlus,
        errorValuesMinus,
        tau,
        minWeights,
        maxWeights,
        averageValueFunctions,
        averageOverallValues
    )

