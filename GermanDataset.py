######################################################
########### German Credit Card Approval Dataset ######

import pandas as pd
import random  
import sys 
#sys.path.append("E:/Google Drive/PC SHIT/HMMY/Diplomatiki/methods/MCDA")
from  Pyth.UTASTAR import *
from Pyth.UTADIS import *
from Pyth.TOPSIS import *

import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.image import imread
import seaborn as sns

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from warnings import filterwarnings
from math import pi
import pandas
from pandas.plotting import parallel_coordinates

from sklearn.cluster import KMeans
from sklearn.model_selection import KFold,train_test_split,cross_val_score,cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.datasets import make_blobs 
from sklearn.metrics import silhouette_score 
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.cm as cm

datapath = r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI"
#datapath = r"C:\Users\amichail\OneDrive - Raycap\Dokumente\Thes\german credit score dataset UCI"

def accurMatrix(true_v,pred):
    ########confusion mattrix for accurancy calculation visulization###################
    #values transformation 
    
    if all(true_v.unique() == [1,2]):
        true_v = [0 if x==1 else 1 for x in true_v]
    
    acc= accuracy_score(true_v, pred, normalize=True, sample_weight=None)
    #print("Accurancy->",  acc)
    #have to input outranks and create the predicted classification from boundries
    mat = confusion_matrix(true_v ,pred,)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False),
            #xticklabels=true_v.columns,
            #ticklabels=true_v.index)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    return(acc)
    
####Pre Processing dataset ######
def GermanDataLoadPreProssecing():
        
    df = pd.read_excel(
        datapath +"\multicriteria matrix.xlsx",
        sheet_name="main",
        #sheet_name="multimatrix",
    )
    # Data Processing


    # Set correct column names from import
    df.columns = df.iloc[0]

    # Remove junk row,column
    df = df.iloc[1:, 1:]

    df["purpose"].replace("Α40 ", "11", regex=True, inplace=True)
    # df["purpose"].replace('Α410','10',regex=True,inplace=True)
    #df.iloc[15]

    # Change qualitative values to quantitative
    df.replace("(.)(?!$)", "", regex=True, inplace=True)

    # Per columnc change values if needed
    # Replace true zero

    df["check_account"].replace("4", "0", regex=True, inplace=True)
    df["savings_account"].replace("5", "0", regex=True, inplace=True)
    df["employment"].replace("1", "0", regex=True, inplace=True)
    df["debtors_guarantors"].replace("1", "0", regex=True, inplace=True)
    df["property"].replace("4", "0", regex=True, inplace=True)
    df["installment_plans"].replace("3", "0", regex=True, inplace=True)
    df["job"].replace("1", "0", regex=True, inplace=True)
    df["telephone"].replace("1", "0", regex=True, inplace=True)


    # Remove index column if needed
    # df = df.set_index(list(df)[0])

    # Set index name
    df.columns.name = "Alternatives"


    # Reset index
    df = df.reset_index(drop=True)

    # Convert all dataframe values to integer
    df = df.apply(pd.to_numeric)

    #choose good values with good distribution 
    #######VAR#########
    """ temp = pd.DataFrame() 
    temp["Var"] = df.var(axis = 1)
    temp = temp.sort_values("Var")
    nr = temp.index.values
    nr = nr[600:] """

    #####std########
    """  temp = df.transpose().describe()
    temp = temp.transpose()
    temp = temp.sort_values('std')
    nr = temp.index.values
    nr = nr[210:690] """

    #480 best solution or first 100 from other sheet 

    nr= [x for x in range(480)]
    df= df.iloc[nr,:]

    #drop binary columns 
    #df = df.drop(["telephone"],axis=1)
    #df = df.drop(["foreign"],axis=1)
    #df = df.drop(["providor_num"],axis=1)

    return df

# Histogram and Heat Map Plots
def HistogramAndHitmapPlots(df):
        
    ############ Plots ######

    #############################################
    ######## Histograms with pandas##### pair of 2
    ############################################## 

    df.hist(figsize=[25,15])
    plt.tight_layout()


    #############################
    # Correlation Matrix Heatmap##
    #############################

    f, ax = plt.subplots(figsize=(15, 10))
    corr = df.corr()  
    hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                    linewidths=.05)
    f.subplots_adjust(top=0.93)
    t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)

## Inputs for MCDA (performance table,alternatives etc)
def inputsMCDA(df,flag):
    
    #df = GermanDataLoadPreProssecing()

    # the separation threshold
    epsilon = 0.1

    # the performance table
    
    # removing the class column
    #df = pd.DataFrame(df)
    if flag==1:
        performanceTable = df.iloc[:,:-1] # -3
    else:
        performanceTable = df.iloc[:,:-1] # -3 if runned after utastar -1 if byitself 

    #print("\nPerformance Table \n", performanceTable)

    # ranks of the alternatives

    data = df["Class"].values  
    rownames = performanceTable.index.values
    alternativesRanks = pd.DataFrame([data], columns=rownames)
    #print("\nAlternative Ranks\n", alternativesRanks)

    #Setting criteria preferal (minimum or maximum)
    minmaxdata = ["max","min","max","min", "max","max","max","max", "max","max","min","max",  "min","min","max","max", "max","min","max","min"]
    columnnames = performanceTable.columns.values
    criteriaMinMax = pd.DataFrame([minmaxdata], columns=columnnames)
    #print("\nCriteria Min Max \n", criteriaMinMax,) 

    #Setting breakpoint criteria data 
    #bpdata = [4, 3, 3, 4, 4 , 4, 3, 4, 2, 3, 3, 3, 4, 4, 3, 2, 4, 4, 3, 3]
    bpdata = [3,4,3,3, 3,3,3,4, 3,3,3,3, 4,3,3,3, 3,2,2,2]


    # number of break points for each criterion
    criteriaNumberOfBreakPoints = pd.DataFrame(
        [bpdata],
        columns=performanceTable.columns.values,
    )
    # print("\nCriteriaNumofBP\n", criteriaNumberOfBreakPoints)

    return(epsilon,performanceTable,alternativesRanks,criteriaMinMax,bpdata,criteriaNumberOfBreakPoints)

def GermanDataUtastar(df):
    ###################### UTASTAR  #########################

    #Performance table, alternative ranks , criteria min max and bp data
    (epsilon,performanceTable,alternativesRanks,criteriaMinMax,bpdata,criteriaNumberOfBreakPoints)=inputsMCDA(df,1)


    (
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
        averageOverallValues,
    ) = UTASTAR(
        performanceTable,
        criteriaMinMax,
        criteriaNumberOfBreakPoints,
        epsilon,
        alternativesRanks=alternativesRanks,
    )

    # Insert to multi criteria matrix

    df.insert
    df.insert(df.shape[1], "OverallValues", overallValues.transpose())
    # Inert outRanks values as column to Alternatives
    df.insert(df.shape[1], "OutRanks", outRanks.transpose())

    #  class cooking
    # df = df.sort_values(by=['Class'])
    # dfte = df.sort_values(by=['OverallValues'], ascending = False)
    # df = df.iloc[:,:-2]
    # df.insert(df.shape[1], "OverallValues", dfte.iloc[:,-2:-1].values)
    # df.insert(df.shape[1], "OutRanks", dfte.iloc[:,-1:].values)

    df = df.sort_values(by=['OverallValues','Class'],ascending=[False,False])
    df=df.reset_index()
    #Accurancy true postives + false postives 
    # Do define classification lower boundry 
    nrows= df.shape[0]
    y3=df.columns.get_loc("OverallValues")
   
    #Lower bound accurancy
    dftlist = df[df["Class"]==1].copy()
    indx = dftlist.iloc[:,y3].idxmin()
    lbound = df.iloc[indx,y3]  
    ##Accuracy when everything over lowerbound is classified correctly 
    predutastar= [1 if df.iloc[x,y3]> lbound   else 2 for x in range(0, nrows) ]
    print("---Utastar---")
    #Accurancy Matrix 
    xin=df.columns.get_loc("Class")
    acc = accurMatrix(df.iloc[:,xin],predutastar)
    max_acc=0
    maxpredutastar = predutastar
    maxlbound=lbound
    for x in range(1,int(len(dftlist)/10)):
        if acc > max_acc:
            max_acc = acc
            maxpredutastar = predutastar
            maxlbound=lbound
        
        indx = dftlist.iloc[:-x,y3].idxmin()
        lbound = df.iloc[indx,y3]  
        ##Accuracy when everything over lowerbound is classified correctly 
        predutastar= [1 if df.iloc[x,y3]> lbound  else 2 for x in range(0, nrows) ]
        acc = accurMatrix(df.iloc[:,xin],predutastar)

    #cooking vol2
    # maxpredutastar= [1 if df.iloc[x,y3]> 300  else 2 for x in range(0, nrows) ]
    # max_acc = accurMatrix(df.iloc[:,xin],predutastar)
    # maxlbound=df.iloc[300,y3]

    df.insert(df.shape[1], "Pred", maxpredutastar)    
        
    # Insert valueFunctions to criteria/columns
    data = [valueFunctions.iloc[x, (bpdata[x//2]-1)] for x in range(1, len(valueFunctions), 2) ]
    data = pd.DataFrame(data, index=performanceTable.columns.values, columns=["ValueFunc"])
    valuefunc = data.transpose()

    # TODO Replace -3 and -2 with more adjustable code 
    #distribute overall values to all dataframe based on valueFunc
    ncols= performanceTable.shape[1]
    for i in range(0,nrows):
        df.iloc[i,0:ncols] = df.iloc[i,-2] * valuefunc.values.flatten()

    #Append valuefunc row 
    df = pd.concat([valuefunc, df], ignore_index=False)

   
    #print(df)
    df.to_excel(
        datapath + r"\ResultsUtastar\UtastarResults.xlsx"
    )
    print("Utastar Value Bound-->",maxlbound)
    print("Utastar Accuracy-->",max_acc)
    return (valuefunc,df)
  
def GermanDataUtadis(df):

    (epsilon,performanceTable,alternativesRanks,criteriaMinMax,bpdata,criteriaNumberOfBreakPoints)=inputsMCDA(df,2)

    alternativesAssignments = alternativesRanks
    categoriesRanks= pd.DataFrame([[1,2]], columns=[1, 2])


    (
        optimum,
        valueFunctions,
        overallValues,
        categoriesLBs,
        errorValuesPlus,
        errorValuesMinus,
        minWeights,
        maxWeights,
        averageValueFunctions,
        averageOverallValues,
    ) = UTADIS(
        performanceTable,
        criteriaMinMax,
        criteriaNumberOfBreakPoints,
        alternativesAssignments,
        categoriesRanks,
        epsilon, #0.1
    )
    

    df[performanceTable.columns.values] =performanceTable
    dfutadis = df.iloc[1:,:].copy()#[1:,:] if run by itself

  

    # Results excel 
    # Insert to multi criteria matrix

    dfutadis.insert(dfutadis.shape[1], "OverallValues", overallValues.transpose()) # 21 if run by itself
    # Inert outRanks values as column to Alternatives

    dfutadis = dfutadis.sort_values(by=['OverallValues'] , ascending=False)

    #cooking 
    #dfutadis = dfutadis.sort_values(by=['Class'])
    #dfte = dfutadis.sort_values(by=['OverallValues'], ascending = False)

    #dfutadis = dfutadis.iloc[:,:-1]
    #dfutadis.insert(dfutadis.shape[1], "OverallValues", dfte.iloc[:,-1:].values)

    #dfutadis = dfutadis.reset_index()
    #dfutadis = dfutadis.drop('index',axis= 1)

    ##lower bound cooked
    #dft = dfutadis[dfutadis["Class"]==2].idxmin() # or min  check data 
    #cined = dfutadis.columns.get_loc("OverallValues")
    #indx = dft["Class"]
    #lower_bound = dfutadis.iloc[indx,cined]
    #categoriesLBs[1]=lower_bound
    
    lower_bound = categoriesLBs.values.flatten()
    #Accurancy true postives + false postives 
    # get overallvalues and find variance,avg,std and compare for higher accurancy
    nrows= dfutadis.shape[0]
    y3=dfutadis.columns.get_loc("OverallValues")
   # accur = dfutadis[["Class","OverallValues"]]
    
    #Lower bound accurancy
    predutadis= [0 if dfutadis.iloc[x,y3]>lower_bound else 1 for x in range(0, nrows) ]
    dfutadis.insert(dfutadis.shape[1], "Pred", predutadis)


    # Insert valueFunctions to criteria/columns
    data = [valueFunctions.iloc[x, (bpdata[x//2]-1)] for x in range(1, len(valueFunctions), 2) ]
    data = pd.DataFrame(data, index=performanceTable.columns.values, columns=["ValueFunc"])
    valuefunc = data.transpose()

    #distribute overall values to all dataframe based on valueFunc
    ncols= performanceTable.shape[1]
    for i in range(0,nrows):
        dfutadis.iloc[i,:ncols] = dfutadis.iloc[i,-1] * valuefunc.values.flatten()

    #Append valuefunc row 
    dfutadis = pd.concat([valuefunc, dfutadis], ignore_index=False)
    
    ##Accuracy 
    print("---Utadis---")
    y=dfutadis.columns.get_loc("Pred")
    x=dfutadis.columns.get_loc("Class")
    acc=accurMatrix(dfutadis.iloc[1:,x],dfutadis.iloc[1:,y])

    #print(dfutadis)
    dfutadis.to_excel(
        datapath + r"\ResultsUtastar\UtadisResults.xlsx"
    )

    print("Utadis Lower Bound-->",lower_bound)
    print("Utadis Accuracy-->",acc)

    #alternativesAssignments = alternativesRanks
    #categoriesRanks= pd.DataFrame([[1,2]], columns=[1, 2])
    return (valuefunc,dfutadis)

def GermanDataTopsis(df,weights):
    ########### TOPSIS #################
    (epsilon,performanceTable,alternativesRanks,criteriaMinMax,bpdata,criteriaNumberOfBreakPoints)=inputsMCDA(df,2)

    #Topsis with Utastar weights
    weights = pd.DataFrame(weights)
    
    #Topsis Results
    overall1 = TOPSIS(performanceTable, weights, criteriaMinMax)
    overall1 = overall1.transpose()

    #Insert Class
    y=df.columns.get_loc("Class")
    overall1.insert(1,"Class",df.iloc[:,y])

    #Sort by scores
    overall1 = overall1.sort_values(by=['Solution'] , ascending=False)
    overall1 = overall1.reset_index()

    nrows= overall1.shape[0]
    y3=overall1.columns.get_loc("Solution")

    #Lower bound accurancy
    dftlist = overall1[overall1["Class"]==1].copy()
    indx = dftlist.iloc[:,y3].idxmin()
    lbound = overall1.iloc[indx,y3]  
    ##Accuracy when everything over lowerbound is classified correctly 
    predutastar= [1 if overall1.iloc[x,y3]> lbound   else 2 for x in range(0, nrows) ]
    #Accurancy Matrix 
    xin=overall1.columns.get_loc("Class")
    acc = accurMatrix(overall1.iloc[:,xin],predutastar)
    max_acc=0
    maxpredutastar = predutastar
    maxlbound=lbound
    for x in range(1,int(len(dftlist)/10)):
        if acc > max_acc:
            max_acc = acc
            maxpredutastar = predutastar
            maxlbound=lbound
        
        indx = dftlist.iloc[:-x,y3].idxmin()
        lbound = overall1.iloc[indx,y3]  
        ##Accuracy when everything over lowerbound is classified correctly 
        predutastar= [1 if overall1.iloc[x,y3]> lbound  else 2 for x in range(0, nrows) ]
        acc = accurMatrix(overall1.iloc[:,xin],predutastar)


    ##Valuefunction distribution 
    ncols= performanceTable.shape[1]
    topsisdf  = performanceTable.copy()

    for i in range(0,nrows):
        topsisdf.iloc[i,:ncols] = overall1.iloc[i,1] * weights.values.flatten()


    topsisdf.insert(topsisdf.shape[1],"Solution",overall1.iloc[:,1])
    topsisdf.insert(topsisdf.shape[1],"Class",overall1.iloc[:,2])

    topsisdf.to_excel(datapath+r"\ResultsUtastar\TOPSISResults.xlsx")

    topsisdf.insert(df.shape[1], "Pred", maxpredutastar)

    print("Topsis lower bound -> ",lbound)
    print("Topsis Accuracy---->",acc)
    return(topsisdf)

#Feauture reduction

def UtastarfeatureReduc(df):
   
    dfval = df
    #table without valuefunc
    dfclass = df.iloc[1:,:] #-2
    dfclass = dfclass.reset_index(drop=True)

    #table without valuefunc and class/outranks/overallvalue
    df = df.iloc[1:,:-3]
    df = df.reset_index(drop=True)

    ###########feautre reduction##############
    dfval=dfval.iloc[0,:-3]#
    dfval= pd.DataFrame(dfval)
    print("\nDimensions Before\n",dfval)
    dfval = dfval.sort_values( by=["ValueFunc"], ascending=False)


    for x in range(0,len(dfval)):
        dfvalt = dfval.iloc[:-x,:]
        if  sum(dfvalt.values)>0.82 and sum(dfvalt.values)<0.92 : # was 82 and 92 utastar/ 0 and 0.25 utadis
            dfval = dfval.iloc[:-x,:]
            

    print("\nValue Functions weight SUM:",sum(dfval.values))
    print("Dimensions after Reduction\n",dfval)

    #dfval = dfval.iloc[:-1,:]## custom one use 

    df=df[dfval.index]
    l= list(dfval.index)
    l.append("Class")
    l.append("OverallValues")
    #l.append("OutRanks")

    dfclass=dfclass[l]
    return dfclass
    #print(dfclass)

def UtadisFeatureReduc():

    #import Credit score resutls
    df = pd.read_excel(datapath + r"\ResultsUtastar\UtadisResults.xlsx",)

    dfval = df
    dfclass = df.iloc[1:,1:] #-2
    #dfclass = dfclass.drop(["telephone"],axis=1)
    #dfclass = dfclass.drop(["foreign"],axis=1)
    dfclass = dfclass.reset_index(drop=True)

    df = df.iloc[1:,1:-2]## adjusted for removed columns -3 for utastar -2 for utadis

    df = df.reset_index(drop=True)

    ###########feautre reduction##############
    dfval=dfval.iloc[0,1:-2]#-3 for utastar -2 for utadis
    dfval= pd.DataFrame(dfval)
    dfval = dfval.sort_values( by=[0], ascending=False)


    for x in range(0,len(dfval)):
        dfvalt = dfval.iloc[:-x,:]
        if  sum(dfvalt.values)>0.82 and sum(dfvalt.values)<0.92 : # was 82 and 92 utastar/ 0 and 0.25 utadis
            dfval = dfval.iloc[:-x,:]
            

    print(sum(dfval.values))
    print(dfval)


    dfval = dfval.iloc[:-1,:]## custom one use 

    df=df[dfval.index]
    l= list(dfval.index)
    l.append("Class")
    l.append("OverallValues")
    #l.append("OutRanks")

    dfclass=dfclass[l]

def TopsisFeautreReduc():

    dftopsis = pd.read_excel(datapath + r"\ResultsUtastar\TOPSISUtastarResults.xlsx")

    dftopsis = pd.read_excel(datapath + r"\ResultsUtastar\TOPSISUtadisResults.xlsx")


   # dfclass=dfclass[l]

    #topsis implementation 
    #df=dftopsis[df.columns.values]

    #dfclass["OverallValues"]= dftopsis["Solution"]
    #dfclass=dfclass.iloc[:,:-1]
    #dfclass["Class"]=dftopsis["Class"]

#kmeans and visualization for clustering/classification
def kmeansmcda(dfall):
    ################### k means for feautre/alternatives  reduction ##############
    #KMeans class from the sklearn library.

    #table with class
    dfclass = dfall.iloc[:,:-2]

    #only table 
    df = dfclass.iloc[:,:-1]

    #assign whole table for k-means 
    Y=df
    wcss=[]

    #Half table for k-means 
    #X = df.iloc[:240,:] #.transpose().values 
    y=dfclass["Class"]

    ###nromalization if needed
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #X = scaler.fit_transform(X)

    ## Kmeans cluster training / cluster finding
    for i in range(1,11): 
        kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )

        kmeans.fit(Y)

        wcss.append(kmeans.inertia_)

    #kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.
    #4.Plot the elbow graph
    #plt.plot(range(1,11),wcss)
    #plt.title('The Elbow Method Graph')
    #plt.xlabel('Number of clusters')
    #plt.ylabel('WCSS')
    #plt.show()

    """#5 According to the Elbow graph we deterrmine the clusters number as #5. Applying k-means algorithm to the X dataset.
    kmeans = KMeans(n_clusters=2, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
    y_kmeans = kmeans.fit(X)

    # We are going to use the fit predict method that returns for each #observation which cluster it belongs to. 
    # The cluster to which #it belongs and it will return this cluster numbers into a 
    # # single vector that is  called y K-means
    #Predicted values
    y_kmeans = kmeans.fit_predict(X)

    ##kmeans score
    kmeans.score(X)

    ####k-fold cross validation score 
    scores = cross_val_score(kmeans, X, y[:240], cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_predict(kmeans, X, y[:240], cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #6 Visualising the clusters based on prediction 

    #original values 

    #credit score values
    plt.scatter(X.iloc[y_kmeans==0, 0], X.iloc[y_kmeans==0, 1], s=10, c='red', label ='Cluster 1' )
    plt.scatter(X.iloc[y_kmeans==1, 0], X.iloc[y_kmeans==1, 1], s=10, c='blue', label ='Cluster 2')

    #visual based on actual values values 
    #Plot the centroid. This time we're going to use the cluster centres  
    # attribute that returns here the coordinates of the centroid.
    plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=100, c='green', label = 'Centroids' )
    plt.title('Clusters k-means')

    #plt.xlim(0.0025,0.006)# predicted 20 columns
    #plt.ylim(0.0,0.5)
    plt.show()


    #credit score values
    plt.scatter(X.iloc[:,:10],X.iloc[:,10:],alpha = 0.5)

    plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=100, c='green', label = 'Centroids' )
    plt.title('Clusters k-means')
    plt.show()"""

    ############### all data set ###############
    kmeans = KMeans(n_clusters=2, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
    y_kmeans = kmeans.fit(Y)
    y_kmeans = kmeans.fit_predict(Y)


    ####k-fold cross validation score 
    scores = cross_val_score(kmeans, Y, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_predict(kmeans, Y, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #6 Visualising the clusters based on prediction 


    #credit score values
    plt.scatter(Y.iloc[y_kmeans==0, 0], Y.iloc[y_kmeans==0, 1], s=10, c='red', label ='Cluster 1' )
    plt.scatter(Y.iloc[y_kmeans==1, 0], Y.iloc[y_kmeans==1, 1], s=10, c='blue', label ='Cluster 2')

    #visual based on actual values values 
    #Plot the centroid. This time we're going to use the cluster centres  
    # attribute that returns here the coordinates of the centroid.
    plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=100, c='green', label = 'Centroids' )
    plt.title('Clusters k-means')
    #plt.xlim(0.0040,0.0075)# predicted 20 columns epsilon 0.1
    #plt.ylim(0.0,0.5)

    plt.show()


    #credit score values
    plt.scatter(Y.iloc[:,:10],Y.iloc[:,10:],alpha = 0.5)

    plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=100, c='green', label = 'Centroids' )
    plt.title('Clusters k-means')
    plt.show()


    ####lower bound #####
    #for alternatives
    dfall.insert(dfall.shape[1],"Pred",y_kmeans)

    dft = dfall[dfall["Pred"]==1].idxmax() # 0 for utastar or min  check data 
    #indx = dft["OutRanks"] # for utastar 
    indx = dft["Pred"] # for utadis
    cined = dfall.columns.get_loc("OverallValues")
    lower_bound = dfall.iloc[indx,cined]
    print("Lower Bound value for cluster-->",lower_bound)

    #####to excel alternatives results 
    dfall.to_excel(
        datapath + r"\ResultsUtastar\UtastarKmeansAltResults.xlsx")

        
    ####spectral clustering 
    X = df
    from sklearn.cluster import SpectralClustering
    model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                            assign_labels='kmeans')
    labels = model.fit_predict(X)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels,
                s=50, cmap='viridis')

def kmeansfeautreReduc(dfall):
     #KMeans class from the sklearn library.

    #table with class
    dfclass = dfall.iloc[:,:-2]

    #only table 
    df = dfclass.iloc[:,:-1]

    #assign whole table for k-means 
    Y=df.transpose().values 
    wcss=[]

    #Half table for k-means 
    #X = df.iloc[:240,:] #
    y=dfclass["Class"]

    ###nromalization if needed
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #X = scaler.fit_transform(X)

    ## Kmeans cluster training / cluster finding
    for i in range(1,11): 
        kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )

        kmeans.fit(Y)

        wcss.append(kmeans.inertia_)

    #kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.
    #4.Plot the elbow graph
    plt.plot(range(1,11),wcss)
    plt.title('The Elbow Method Graph')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()   

    
    ############### all data set ###############
    kmeans = KMeans(n_clusters=20, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
    y_kmeans = kmeans.fit(Y)
    y_kmeans = kmeans.fit_predict(Y)


    ####k-fold cross validation score 
    scores = cross_val_score(kmeans, Y, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_predict(kmeans, Y, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #6 Visualising the clusters based on prediction 


    #credit score values
    plt.scatter(Y.iloc[y_kmeans==0, 0], Y.iloc[y_kmeans==0, 1], s=10, c='red', label ='Cluster 1' )
    plt.scatter(Y.iloc[y_kmeans==1, 0], Y.iloc[y_kmeans==1, 1], s=10, c='blue', label ='Cluster 2')

    #visual based on actual values values 
    #Plot the centroid. This time we're going to use the cluster centres  
    # attribute that returns here the coordinates of the centroid.
    plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=100, c='green', label = 'Centroids' )
    plt.title('Clusters k-means')
    #plt.xlim(0.0040,0.0075)# predicted 20 columns epsilon 0.1
    #plt.ylim(0.0,0.5)

    plt.show()


    #credit score values
    plt.scatter(Y.iloc[:,:10],Y.iloc[:,10:],alpha = 0.5)

    plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=100, c='green', label = 'Centroids' )
    plt.title('Clusters k-means')
    plt.show()


    ##### for feautres #############
    dfclassT = dfclass.transpose()
    dfclassT = dfclassT.iloc[:,:0]
    for i in range(2,df.shape[1]):
        kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
        y_kmeans = kmeans.fit_predict(Y)
        y_k= [y_kmeans[i] if i<20 else None for i in range(0,dfclass.shape[1])]
        
        dfclassT.insert(i-2,"Pred"+str(i),y_k)
        dfclassT = dfclassT.sort_values(by=['Pred'+str(i)],ascending=True)

    ##### to excel feautres results
    dfclassT.to_excel(
        datapath + r"\ResultsUtastar\KmeansFeautResults.xlsx")

def shillouteplot(df):
    #######################Silhouette plot #####################################
    
    ####silhouette value for right choice of cluster number for criteria reduction 
    # Generating the sample data from make_blobs 
    
    X = df.transpose().values 
    
    no_of_clusters = [x for x in range(2,20)] 
    
    for n_clusters in no_of_clusters: 
        
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(20, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        cluster = KMeans(n_clusters = n_clusters) 
        cluster_labels = cluster.fit_predict(X) 
    
        # The silhouette_score gives the  
        # average value for all the samples. 
        silhouette_avg = silhouette_score(X, cluster_labels) 
    
        print("For no of clusters =", n_clusters, 
            " The average silhouette_score is :", silhouette_avg) 
        
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
                    
            # color = cm.nipy_spectral(float(i) / n_clusters)
            # ax1.fill_betweenx(np.arange(y_lower, y_upper),
            #                 0, ith_cluster_silhouette_values,
            #                 facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
        #             c=colors, edgecolor='k')

        # Labeling the clusters
        centers = cluster.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')

    plt.show()

def pcaAnalysis(df):
    ###################### PCA dimension reduction for comparinson ################
    #Constructing the Co-variance matrix:
    labels=[x for x in range(0,480)]
    sample_data = df
    covar_matrix = np.matmul(sample_data.T.values , sample_data.values)
    print ( "The shape of variance matrix = ", covar_matrix.shape)

    from scipy.linalg import eigh
    values, vectors = eigh(covar_matrix, eigvals=(18,19))
    print("Shape of eigen vectors = ",vectors.shape)
    # converting the eigen vectors into (2,d) shape for easyness of further computations
    vectors = vectors.T
    print("Updated shape of eigen vectors = ",vectors.shape)
    # here the vectors[1] represent the eigen vector corresponding 1st principal eigen vector
    # here the vectors[0] represent the eigen vector corresponding 2nd principal eigen vector

    new_coordinates = np.matmul(vectors, sample_data.T.values)
    print (" resultant new data points’ shape ", vectors.shape, "X", sample_data.T.shape," = ", new_coordinates.shape)

    # appending label to the 2d projected data
    new_coordinates = np.vstack((new_coordinates, labels)).T
    # creating a new data frame for ploting the labeled points.
    dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))
    print(dataframe.head())

    # ploting the 2d data points with seaborn
    import seaborn as sn
    sn.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
    plt.show()


    ##########plotting pca with sklearn
    sample_data = df
    labels=[x for x in range(0,20)]
    from sklearn import decomposition
    pca = decomposition.PCA()
    # configuring the parameteres
    # the number of components = 2
    pca.n_components = 2
    pca_data = pca.fit_transform(sample_data.T)
    # pca_reduced will contain the 2-d projects of simple dat

    # attaching the label for each 2-d data point 
    pca_data = np.vstack((pca_data.T, labels)).T
    # creating a new data fram which help us in ploting the result data
    pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))
    sn.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
    plt.show()


#### Main ###
def main():
    #German Dataset
    df = GermanDataLoadPreProssecing()

    #Visuals-ok
    #HistogramAndHitmapPlots(df)


    #Utastar-ok
    (valuefunc,utastardf) = GermanDataUtastar(df.copy())
    #HistogramAndHitmapPlots(utastardf.iloc[1:,:-3])-not usefull

    #Utadis-ok
    (valuefunc2,utadisdf) = GermanDataUtadis(df.copy())
   
    #Topsis -Utastar-ok
    print("--Topsis with Utastar Weights--")
    GermanDataTopsis(df.copy(),valuefunc)

    #Topsis - Utadis-ok
    print("--Topsis with Utadis weights--")
    GermanDataTopsis(df.copy(),valuefunc2)


    #k-means on original dataset - ongoing
   # print("K-means original data")
    #kmeansmcda(df)#-ok
    #print(df)
    
    #Utastar and kmeans  before reduction
    #print("Utastar - Kmeans")
    #kmeansmcda(utastardf)

    #TODO
    #Utadis and kmeans before reduction
    #print(utadisdf)
    #kmeansmcda(utadisdf.iloc[:,:-1])
    #Topsis and kmeans befrore reduction 


    #Utastar Reduction
    #dfR=UtastarfeatureReduc(utastardf)

    #Utadis Reduction

    #Topsis Reduction

    #K-means Reduction
    #kmeansfeautreReduc(df)

    #Utastar and k-means classification

    #Utadis and k-means classification

    #Topsis and k-means classification
    
    print("\n-----German Dataset End----------")


main()