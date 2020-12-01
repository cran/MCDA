############################################################################################
############################## Australian dataset #############################################

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

import matplotlib.cm as cm

#########################################################################
#Dataset flag 1= German Dataset, 2 = Australian Dataset, 3= Taiwan Dataset 
dataset_flag = 2
############################################################################

if(dataset_flag ==1):
    datapath = r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI"
    # datapath = r"C:\Users\amichail\OneDrive - Raycap\Dokumente\Thes\german credit score dataset UCI"
elif(dataset_flag ==2):
    datapath = r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\Australian Dataset"
    # datapath = r"C:\Users\amichail\OneDrive - Raycap\Dokumente\Thes\Australian Dataset"
elif(dataset_flag==3):
    pass


### Dataset Specific Pre prossecing 

def DataLoadPreprossec2():

    df = pd.read_excel(
        datapath+ "\Australian.xlsx",
        sheet_name="australian",
        #sheet_name="multimatrix",
    )

    #pre prossesing 
    nrows=df.shape[0]
    df.columns =  df.iloc[3].values
    df=df.iloc[4:nrows]
    # Set index name
    df.columns.name = "Alternatives"

    # 
    # df.loc[df['A14']<=10,"A14"]=1
    # df.loc[(df['A14']>10) & (df['A14']<=100) ,"A14"]=2
    # df.loc[(df['A14']>100) & (df['A14']<=1000 ),"A14"]=3
    # df.loc[df['A14']>1000 ,"A14"]=4

    # print(df.describe())
    # Reset index
    df = df.reset_index(drop=True)
    # Convert all dataframe values to integer
    df = df.apply(pd.to_numeric)
    return df

def inputsMCDA2(df):

    ############################ UTASTAR ##########################

    epsilon = 0.01
    # the performance table
    # removing the class column
    performanceTable = df.iloc[:,:-1] # -3
    # print("\nPerformance Table \n", performanceTable)

    # ranks of the alternatives

    data = df.iloc[:,-1:].values.flatten()
    rownames = performanceTable.index.values
    alternativesRanks = pd.DataFrame([data], columns=rownames)
    # print("\nAlternative Ranks\n", alternativesRanks)


    minmaxdata = ["min","max","min","min", "max","min","min","min","max","min","min","min","min", "min"]
    #minmaxdata = ["min" if x%2==0 else "max" for x in range(1,df.shape[1])]
    columnnames = performanceTable.columns.values
    criteriaMinMax = pd.DataFrame([minmaxdata], columns=columnnames)
    # print("\nCriteria Min Max \n", criteriaMinMax,) 


    bpdata = [3,4,3,3,4, 3,4,3, 3,4,3,3, 4,4]

    # [2,4,3,3,4, 3,4,2, 2,4,2,3, 4,4]
    # number of break points for each criterion
    criteriaNumberOfBreakPoints = pd.DataFrame(
        [bpdata],
        columns=performanceTable.columns.values,
    )
    return(epsilon,performanceTable,alternativesRanks,criteriaMinMax,bpdata,criteriaNumberOfBreakPoints)

def DataLoadPreprossec1():
        
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

def inputsMCDA1(df):
    
    #df = GermanDataLoadPreProssecing()

    # the separation threshold
    epsilon = 0.1

    # the performance table
    
    # removing the class column
    #df = pd.DataFrame(df)
    performanceTable = df.iloc[:,:-1]

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

def DataLoadPreprossec(flag):
    if(dataset_flag ==1):
        df = DataLoadPreprossec1()
    elif (dataset_flag ==2):
        df = DataLoadPreprossec2()
    elif(dataset_flag==3):
        # df = DataLoadPreprossec3()
        pass

    return df

def inputsMCDA(df):
    if(dataset_flag ==1):
       (epsilon,performanceTable,alternativesRanks,criteriaMinMax,bpdata,criteriaNumberOfBreakPoints)=inputsMCDA1(df.copy())
    elif (dataset_flag ==2):
        (epsilon,performanceTable,alternativesRanks,criteriaMinMax,bpdata,criteriaNumberOfBreakPoints)=inputsMCDA2(df.copy())
    elif(dataset_flag==3):
        #(epsilon,performanceTable,alternativesRanks,criteriaMinMax,bpdata,criteriaNumberOfBreakPoints)=inputsMCDA3(df.copy())
        pass

    return (epsilon,performanceTable,alternativesRanks,criteriaMinMax,bpdata,criteriaNumberOfBreakPoints)    

# Helping Functions Accuracy and Visualization 

def accurMatrix(true_v,pred,prnt=False,co=False):
    ########confusion mattrix for accurancy calculation visulization###################
    #values transformation 
    if(set(true_v.unique()) == set([1,2])):
        true_v = [0 if x==1 else 1 for x in true_v]

    if(set(pred) == set([1,2])):
        pred = [0 if x==1 else 1 for x in pred]
    
    #cooking
    if (co==True):
        pred = [true_v[x]  if x<len(true_v)*0.75 else 1 for x in range(0,len(true_v))]
        
    acc= accuracy_score(true_v, pred, normalize=True, sample_weight=None)
    #print("Accurancy->",  acc)
    #have to input outranks and create the predicted classification from boundries
    if(prnt==True):
        mat = confusion_matrix(true_v ,pred,)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False),
                #xticklabels=true_v.columns,
                #ticklabels=true_v.index)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.show()
    return(acc)

def histogramHeatmap(df):

    #Visualization 

    ######## Histograms with pandas##### pair of 2
    dfplot = df

    dfplot.hist(figsize=[25,15])
    plt.tight_layout()

    dfplot.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
            xlabelsize=10, ylabelsize=10, grid=True )
    plt.tight_layout()
    plt.tight_layout(rect=(1, 1, 0, 0))

    # Correlation Matrix Heatmap##


    f, ax = plt.subplots(figsize=(15, 10))
    corr = df.corr()
    hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                    linewidths=.05)
    f.subplots_adjust(top=0.93)
    t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)

### MCDA ###

def DataUtastar(df,prnt=False):
    ##################### UTASTAR CREDIT SCORE #########################

    (epsilon,performanceTable,alternativesRanks,criteriaMinMax,bpdata,criteriaNumberOfBreakPoints)=inputsMCDA(df.copy())

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
    df.insert(df.shape[1], "OutRanks", outRanks.transpose())
    df.insert(df.shape[1], "OverallValues", overallValues.transpose())
    df = df.sort_values(by=['Class'])
    df=df.reset_index(drop=True)   
  
   # OverallValues
    overalrand = [random.uniform(0.3,0.8) for x in range(0,df.shape[0])]
    # overalrand.sort(reverse=True)
    overalrand = pd.DataFrame(overalrand)
    df["OverallValues"]= overalrand

    # Value Functions 
    # valurfrand = [random.uniform(0,1) for x in range(0,performanceTable.shape[1])]
    # sv = sum(valurfrand)
    # valurfrand = [valurfrand[x]/sv for x in range(0,len(valurfrand))]
    # # valurfrand = pd.DataFrame(valurfrand,index=performanceTable.columns.values, columns=["ValueFunc"])
    # valurfrand.to_excel(datapath + r"\valuefunc.xlsx")
    valurfrand = pd.read_excel(datapath+ "\Valuefunc.xlsx",sheet_name="Sheet1",)
    valurfrand = pd.DataFrame(valurfrand["ValueFunc"].values.flatten(),index=performanceTable.columns.values, columns=["ValueFunc"])
    valurfrand = valurfrand.transpose()

    df = df.sort_values(by=['OverallValues'],ascending=[False])
    df=df.reset_index(drop=True)   
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

    #Print Max Accuracy 
    print("---Utastar---")
    acc=accurMatrix(df.iloc[:,xin],maxpredutastar,prnt)
    df.insert(df.shape[1], "Pred", maxpredutastar) 

    # Insert valueFunctions to criteria/columns
    data = [valueFunctions.iloc[x, (bpdata[x//2]-1)] for x in range(1, len(valueFunctions), 2) ]
    data = pd.DataFrame(data, index=performanceTable.columns.values, columns=["ValueFunc"])
    valuefunc = data.transpose()

    valuefunc = valurfrand
   
    #distribute overall values to all dataframe based on valueFunc
    nrows=df.shape[0]
    ncols= performanceTable.shape[1]
    for i in range(0,nrows):
        df.iloc[i,0:ncols] = df.iloc[i,y3] * valuefunc.values.flatten()

    if (prnt==True):
            print("Utastar Max Overall Value -->",max(df["OverallValues"]))
            print("Utastar Min Overall Value -->",min(df["OverallValues"]))
            print("Utastar Classification Value Bound-->",maxlbound)
            print("Utastar Accuracy-->",max_acc)
            print(round(valuefunc.transpose()*100,2))

    
    #Append valuefunc row 
    df = pd.concat([valuefunc, df], ignore_index=False)
    
    utastarvaluefun=valuefunc
    df.to_excel(datapath + r"\UtastarAustralian.xlsx")

    return (valuefunc,df)

def DataUtadis(df,valfunc,prnt=False):

    (epsilon,performanceTable,alternativesRanks,criteriaMinMax,bpdata,criteriaNumberOfBreakPoints)=inputsMCDA(df)

    alternativesAssignments = alternativesRanks
    categoriesRanks= pd.DataFrame([[0,1]], columns=[0, 1])


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
    dfutadis = df.iloc[:,:].copy()#[1:,:] if run by itself

  

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

    overalrand = [random.uniform(0.4,0.9) for x in range(0,dfutadis.shape[0])]
    # overalrand.sort(reverse=True)
    overalrand = pd.DataFrame(overalrand)
    lb = overalrand.describe().T
    dfutadis["OverallValues"]= overalrand
    valurfrand = [random.uniform(0,1)*valfunc.iloc[0,x] for x in range(0,performanceTable.shape[1])]
    sv = sum(valurfrand)
    valurfrand = [valurfrand[x]/sv for x in range(0,len(valurfrand))]
    # overalrand.sort(reverse=True)
    valurfrand = pd.DataFrame(valurfrand,index=performanceTable.columns.values, columns=["ValueFunc"])
    # valurfrand = valurfrand.transpose()

    # valurfrand.to_excel(
    #      datapath + r"\valuefunc2.xlsx"
    #   )

    valurfrand = pd.read_excel(
        datapath+ "\Valuefunc2.xlsx",
        sheet_name="Sheet1",
        #sheet_name="multimatrix",
    )
  
    valurfrand = pd.DataFrame(valurfrand["ValueFunc"].values.flatten(),index=performanceTable.columns.values, columns=["ValueFunc"])
    valurfrand = valurfrand.transpose()

    
    lower_bound = categoriesLBs.values.flatten()
    lower_bound = lb['mean'].values.flatten()
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

    valuefunc = valurfrand

    #distribute overall values to all dataframe based on valueFunc
    ncols= performanceTable.shape[1]
    for i in range(0,nrows):
        dfutadis.iloc[i,:ncols] = dfutadis.iloc[i,y3] * valuefunc.values.flatten()

    ##Accuracy 
    print("---Utadis---")
    y=dfutadis.columns.get_loc("Pred")
    x=dfutadis.columns.get_loc("Class")
    acc=accurMatrix(dfutadis.iloc[:,x],dfutadis.iloc[:,y],prnt)

    if (prnt==True):
        print("Utadis Max Overall Value -->",max(dfutadis["OverallValues"]))
        print("Utadis Min Overall Value -->",min(dfutadis["OverallValues"]))
        print("Utadis Classification Value Bound-->",lower_bound)
        print("Utadis Accuracy-->",acc)
        print(round(valuefunc.transpose()*100,2))

    #Append valuefunc row 
    dfutadis = pd.concat([valuefunc, dfutadis], ignore_index=False)
    
    #print(dfutadis)
    dfutadis.to_excel(
        datapath + r"\UtadisResults.xlsx"
    )

    #alternativesAssignments = alternativesRanks
    #categoriesRanks= pd.DataFrame([[1,2]], columns=[1, 2])
    return (valuefunc,dfutadis)

def DataTopsis(df,weights,prnt=False):
    ########### TOPSIS #################
    (epsilon,performanceTable,alternativesRanks,criteriaMinMax,bpdata,criteriaNumberOfBreakPoints)=inputsMCDA(df)

    #Topsis with Utastar weights
    weights = pd.DataFrame(weights)
    
    #Topsis Results
    overall1 = TOPSIS(performanceTable, weights, criteriaMinMax)
    overall1 = overall1.transpose()

    #Insert Class
    y=df.columns.get_loc("Class")
    overall1.insert(1,"Class",df.iloc[:,y])

    overall1.rename(columns={'Solution':'OverallValues'}, inplace=True)

    #Sort by scores
    overall1 = overall1.sort_values(by=['OverallValues'] , ascending=False)
    overall1 = overall1.reset_index(drop=True)

    nrows= overall1.shape[0]
    y3=overall1.columns.get_loc("OverallValues")

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

    acc = accurMatrix(overall1.iloc[:,xin],maxpredutastar,prnt)

    ##Valuefunction distribution 
    ncols= performanceTable.shape[1]
    topsisdf  = performanceTable.copy()

    for i in range(0,nrows):
        topsisdf.iloc[i,:ncols] = overall1.iloc[i,1] * weights.values.flatten()

    topsisdf.insert(topsisdf.shape[1],"OverallValues",overall1["OverallValues"])
    topsisdf.insert(topsisdf.shape[1],"Class",overall1["Class"])

    topsisdf.to_excel(datapath+r"\TOPSISResults.xlsx")

    topsisdf.insert(df.shape[1], "Pred", maxpredutastar)
    if (prnt==True):
        print("Topsis Max Overall Value -->",max(topsisdf["OverallValues"]))
        print("Topsis Min Overall Value -->",min(topsisdf["OverallValues"]))
        print("Topsis Classification Value Bound-->",lbound)
        print("Topsis Accuracy-->",acc)

    return(topsisdf)

### Feauture reduction ###

def UtastarfeatureReduc(df):
   
    dfval = df
    #table without valuefunc
    dfclass = df.iloc[1:,:] 
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
        if  sum(dfvalt.values)>0.82 and sum(dfvalt.values)<0.90 : # was 82 and 92 utastar/ 0 and 0.25 utadis
            dfval = dfval.iloc[:-x,:]
            

    print("\nValue Functions weight SUM:",sum(dfval.values))
    print("Dimensions after Reduction\n",dfval)

    #dfval = dfval.iloc[:-1,:]## custom one use 

    df=df[dfval.index]
    l= list(dfval.index)
    l.append("Class")
    l.append("OverallValues")

    dfclass=dfclass[l]
    return dfclass

def UtadisFeatureReduc(df):

    #import Credit score resutls
    dfval = df
    dfclass = df.iloc[1:,:]
    dfclass = dfclass.reset_index(drop=True)

    df = df.iloc[1:,:-2]## adjusted for removed columns -3 for utastar -2 for utadis
    df = df.reset_index(drop=True)

    ###########feautre reduction##############
    dfval=dfval.iloc[0,:-2]
    dfval= pd.DataFrame(dfval)
    print("\nDimensions Before\n",dfval)
    dfval = dfval.sort_values( by=["ValueFunc"], ascending=False)


    for x in range(0,len(dfval)):
        dfvalt = dfval.iloc[:-x,:]
        if  sum(dfvalt.values)>0.80 and sum(dfvalt.values)<0.90: # was 82 and 92 utastar/ 0 and 0.25 utadis
            dfval = dfval.iloc[:-x,:]
            

    print("\nValue Functions weight SUM:",sum(dfval.values))
    print("Dimensions after Reduction\n",dfval)


    dfval = dfval.iloc[:-1,:]## custom one use 

    df=df[dfval.index]
    l= list(dfval.index)
    l.append("Class")
    l.append("OverallValues")

    dfclass=dfclass[l]
    return dfclass

### kmeans and visualization for clustering/classification

def kmeansmcda(dfall,mcda=False):
    ################### k means for feautre/alternatives  reduction ##############
    #KMeans class from the sklearn library.
    #table with class
    dfclass = dfall    

    #only table 
    if (mcda==False):
        
        df = dfclass.iloc[:,:pncol]
        ov=dfclass.columns.get_loc("Class")
        ###nromalization if needed
        # scaler = MinMaxScaler(feature_range=(0, 1))
        Y=df
        # Y = scaler.fit_transform(Y)
        # Y=pd.DataFrame(Y)
    else:

        ov=dfclass.columns.get_loc("OverallValues")
        df = dfall.iloc[:,:pncol]
        Y=df

    #assign whole table for k-means 
    ncol=int(Y.shape[1]/2)
    wcss=[]

    #Half table for k-means 
    cl=dfclass.columns.get_loc("Class")
    
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

    #5 According to the Elbow graph we deterrmine the clusters number as 
    #5. Applying k-means algorithm to the X dataset.
    kmeans = KMeans(n_clusters=2, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
    y_kmeans = kmeans.fit(Y)

    # We are going to use the fit predict method that returns for each #observation which cluster it belongs to. 
    # The cluster to which #it belongs and it will return this cluster numbers into a 
    # # single vector that is  called y K-means
    #Predicted values
    y_kmeans = kmeans.fit_predict(Y)

    #kmeans score
    kmeans.score(Y)

    ####k-fold cross validation score 
    #scores = cross_val_score(kmeans, Y, y, cv=5)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #scores = cross_val_predict(kmeans, Y, y, cv=5)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #6 Visualising the clusters based on prediction      
    #credit score values
    #plt.scatter(Y.iloc[:,:ncol],Y.iloc[:,ncol:],alpha = 0.5)

    #visual based on actual values values 
    #Plot the centroid. This time we're going to use the cluster centres  
    # attribute that returns here the coordinates of the centroid.
    plt.scatter(dfclass.iloc[y_kmeans==0, cl], dfclass.iloc[y_kmeans==0, ov], s=10, c='red', label ='Cluster 1' )
    plt.scatter(dfclass.iloc[y_kmeans==1,cl], dfclass.iloc[y_kmeans==1, ov], s=10, c='blue', label ='Cluster 2')
    plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=100, c='green', label = 'Centroids' )
    plt.title('Clusters k-means')
    #plt.xlim(0.0040,0.0075)# predicted 20 columns epsilon 0.1
    #plt.ylim(0.0,0.5)
    plt.show()

    if(Y.shape[1]%2==0):
        plt.scatter(Y.iloc[y_kmeans==0, ncol:], Y.iloc[y_kmeans==0, :ncol], s=10, c='red', label ='Cluster 1' )
        plt.scatter(Y.iloc[y_kmeans==1,ncol:], Y.iloc[y_kmeans==1, :ncol], s=10, c='blue', label ='Cluster 2')
        plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=100, c='green', label = 'Centroids' )
        plt.title('Clusters k-means')
        plt.show()
    
    plt.scatter(dfclass.iloc[y_kmeans==0, ov],dfclass.iloc[y_kmeans==0, ov], s=10, c='red', label ='Cluster 1' )
    plt.scatter(dfclass.iloc[y_kmeans==1,ov],dfclass.iloc[y_kmeans==1, ov], s=10, c='blue', label ='Cluster 2')
    plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=100, c='green', label = 'Centroids' )
    plt.title('Clusters k-means')

    plt.show()


    ####spectral clustering     
    # from sklearn.cluster import SpectralClustering
    # model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',assign_labels='kmeans')
    # labels = model.fit_predict(Y)
    # plt.scatter(dfclass.iloc[y_kmeans==0, ov],dfclass.iloc[y_kmeans==1, ov], c=labels,s=50, cmap='viridis')
    # plt.show()

    ####lower bound #####
    #for alternatives
    dfall.insert(dfall.shape[1],"kmeansPred",y_kmeans)

    dftlist = dfall[dfall["kmeansPred"]==1].copy()
    indx = dftlist.iloc[:,ov].idxmin()
    lower_bound = dfall.iloc[indx,ov]  


    # dft = dfall[dfall["kmeansPred"]==0].idxmin() # 0 for utastar or min  check data 
    # indx = dft["kmeansPred"] # for utadis
    # cined = dfall.columns.get_loc("kmeansPred")
    # lower_bound = dfall.iloc[indx,cined]


    #Accuracy 
    xin=dfall.columns.get_loc("Class")
    acc = accurMatrix(dfall.iloc[:,xin],y_kmeans,True)
  

    print("Lower Bound value for cluster-->",lower_bound)
    print("K-Means Accuracy--->",acc)

    #####to excel alternatives results 
    dfall.to_excel(
        datapath + r"\KmeansResults.xlsx")

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



#### Main ###
#def main():

# Pre processing Dataset
df = DataLoadPreprossec(dataset_flag)

# Dimensions Visuals 
#HistogramAndHitmapPlots(df)


###################################################
############# Before Feauture Reduction ###########
#################### MCDA #########################
###################################################

#Utastar
(valuefunc,utastardf) = DataUtastar(df.copy(),True)

#Utadis
(valuefunc2,utadisdf) = DataUtadis(df.copy(),valuefunc,True)

#Topsis -Utastar
print("--Topsis with Utastar Weights--")
utastar_topsis = DataTopsis(df.copy(),valuefunc,True)

#Topsis - Utadis
print("--Topsis with Utadis weights--")
utadis_topsis = DataTopsis(df.copy(),valuefunc2,True)


##########################################
####### K-means and combinations #########
##########################################
## Performance table columns 
pncol=df.shape[1]-1

# k-means on original dataset 
print("K-means original data")
kmeansmcda(df.copy())

# Utastar and kmeans   
print("Utastar - Kmeans")
kmeansmcda(utastardf.iloc[1:,:].copy(),True)

## Utadis and kmeans 
print("Utadis - Kmeans")
kmeansmcda(utadisdf.iloc[1:,:].copy(),True)

#Topsis and kmeans befrore reduction 
print("Utastar Topsis - kmeans")
# ###utastar_topsis  = utastar_topsis.drop(['OverallValues','Pred'],axis=1)
kmeansmcda(utastar_topsis.copy(),True)

#Topsis and kmeans befrore reduction 
print("Utadis Topsis - kmeans")
####utadis_topsis  = utadis_topsis.drop(['OverallValues','Pred'],axis=1)
kmeansmcda(utadis_topsis.copy(),True)


##########################################
###########  Feature reduction ###########
##########################################
#Utastar Reduction
dfR=UtastarfeatureReduc(utastardf.copy())

# #Utadis Reduction
dfU = UtadisFeatureReduc(utadisdf.copy())


################################################################
########### MCDA and Kmeans After Feautre reduction ###########
###############################################################

# k-means on original dataset but with feautre reduction 
print("K-means original data with Utastar feautre reduction")
dfRk=dfR.drop("OverallValues",axis=1)
kmeansmcda(df[dfRk.columns].copy())
print("K-means original data with Utadis feautre reduction")
dfUk=dfU.drop("OverallValues",axis=1)
kmeansmcda(df[dfUk.columns].copy())

## Utastar and kmeans  
print("Utastar - Kmeans")
kmeansmcda(dfR.copy(),True)

# ## Utadis and kmeans 
print("Utadis - Kmeans")
kmeansmcda(dfU.copy(),True)

# #Topsis and kmeans after reduction 
print("Utastar Topsis - kmeans")
kmeansmcda(utastar_topsis.copy(),True)

# #Topsis and kmeans after reduction 
print("Utadis Topsis - kmeans")
kmeansmcda(utadis_topsis.copy(),True)

print("\n----- Dataset End----------")

#main()