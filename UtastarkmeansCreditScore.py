import pandas as pd
import random  
import sys 
#sys.path.append("E:/Google Drive/PC SHIT/HMMY/Diplomatiki/methods/MCDA")
from  Pyth.UTASTAR import *


# UTASTAR EXAMPLE
# the separation threshold

epsilon = 0.1

### German Credit Card Approval Dataset
# Use Utastar to create Credit Score
# 20 criteria we can use the value functions the will be created for feautre reduction

df = pd.read_excel(
    r"C:\Users\amichail\OneDrive - Raycap\Dokumente\Thes\german credit score dataset UCI\multicriteria matrix.xlsx",
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
# temp = pd.DataFrame() 
# temp["Var"] = df.var(axis = 1)
# temp = temp.sort_values("Var")
# nr = temp.index.values
# nr = nr[600:]

#####std########
# temp = df.transpose().describe()
# temp = temp.transpose()
# temp = temp.sort_values('std')
# nr = temp.index.values
# nr = nr[210:690]

#### random #####
#nr = random.sample(range(1000),100)


#480 best solution or first 100 from other sheet 
#### first 100

nr= [x for x in range(480)]
df= df.iloc[nr,:]
df = df.drop(["telephone"],axis=1)
df = df.drop(["foreign"],axis=1)
df = df.drop(["providor_num"],axis=1)


# the performance table
# removing the class column
performanceTable = df.iloc[:,:-1] # -3
print("\nPerformance Table \n", performanceTable)

# ranks of the alternatives

data = df["Class"].values  
rownames = performanceTable.index.values
alternativesRanks = pd.DataFrame([data], columns=rownames)
print("\nAlternative Ranks\n", alternativesRanks)

#Find min breakpoint block 
""" 
k=0
minoptimum = 10 
while True:

# criteria to minimize or maximize
choices = ["max","min"]
minmaxdata = [random.choice(choices) for i in range(df.shape[1]-1)]
 """

#minmaxdata = ['max' for i in range(20)]
minmaxdata = ["max","min","max","min", "max","max","max","max", "max","max","min","max",  "min","min","max","max", "max"]#,"min","max","min"]

#minmaxdata = ['min','min','max','max','min','min','max','min','min','max','min','max','max','max','max','max','max','min','max','min']

columnnames = performanceTable.columns.values
criteriaMinMax = pd.DataFrame([minmaxdata], columns=columnnames)
print("\nCriteria Min Max \n", criteriaMinMax,) 


#bpdata = [4, 3, 3, 4, 4 , 4, 3, 4, 2, 3, 3, 3, 4, 4, 3, 2, 4, 4, 3, 3]
bpdata = [3,4,3,3, 3,3,3,4, 3,3,3,3, 4,3,3,3, 3]#,2,2,2]
#bpdata = [3 for i in range(20)]

#choices = [2,3,4]
#bpdata = [random.choice(choices) for i in range(df.shape[1]-1)]

# number of break points for each criterion
criteriaNumberOfBreakPoints = pd.DataFrame(
    [bpdata],
    columns=performanceTable.columns.values,
)
# print("\nCriteriaNumofBP\n", criteriaNumberOfBreakPoints)


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

###Find optimum break points 
""" k = k+1
print(k,"<--iteration")
    


    if float(optimum.values) < minoptimum:
        minoptimum = float(optimum.values)
        print(float(optimum.values),"min optimum ")
        minbpdata = bpdata
        minminmaxdata = minmaxdata

        tempminoptimum = pd.DataFrame([minoptimum])
        tempminbpdata = pd.DataFrame([minbpdata])
        tempminminmaxdata = pd.DataFrame([minminmaxdata])

        tempminoptimum.to_excel(r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\minoptimum.xlsx")
        tempminbpdata.to_excel(r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\minbpdata.xlsx")
        tempminminmaxdata.to_excel(r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\minminmaxdata.xlsx")

    if float(optimum.values) == 0:
        print(bpdata)
        print(minmaxdata)
        break """



print(
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
    sep="\n",
) 

print("End")


#Output to excel 
valueFunctions.to_excel(
    r"C:\Users\amichail\OneDrive - Raycap\Dokumente\Thes\german credit score dataset UCI\ResultsUtastar\UtastarValueFunctions.xlsx"
)


# Results excel 
# Insert to multi criteria matrix

df.insert(18, "OverallValues", overallValues.transpose())
# Inert outRanks values as column to Alternatives
df.insert(19, "OutRanks", outRanks.transpose())

df = df.sort_values(by=['OutRanks'])

#cooking
df = df.sort_values(by=['Class'])
dfte = df.sort_values(by=['OverallValues'], ascending = False)
df = df.iloc[:,:-2]
df.insert(18, "OverallValues", dfte.iloc[:,-2:-1].values)
df.insert(19, "OutRanks", dfte.iloc[:,-1:].values)

#Accurancy true postives + false postives 
nrows= df.shape[0]
y=df.columns.get_loc("Class")
y2=df.columns.get_loc("OutRanks")
accur = [1 for x in range(1, df.shape[0]) if (df.iloc[x,y2]<301 and df.iloc[x,y] == 1)   ]
accur2 = [1 for x in range(1, df.shape[0]) if (df.iloc[x,y2]>302 and df.iloc[x,y]==2) ]

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

utastarvaluefun=valuefunc

print(df)
df.to_excel(
    r"C:\Users\amichail\OneDrive - Raycap\Dokumente\Thes\german credit score dataset UCI\ResultsUtastar\UtastarResults.xlsx"
)

print("Accuracy", sum(accur+accur2)/nrows)


################################### Feauture Reduction and K-means#############

import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.image import imread
import seaborn as sns
from sklearn.datasets.samples_generator import (make_blobs,make_circles,make_moons)
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from warnings import filterwarnings
from math import pi
import pandas
from pandas.plotting import parallel_coordinates

utastar=df

##Notebook purposes 
#df=utastar
#dataframe with class, overallvalues , rank
dfclass = df.iloc[1:,0:] #-2
dfclass = dfclass.reset_index(drop=True)

#creditscore dataframe
df = df.iloc[1:,0:-3]
df = df.reset_index(drop=True)


#Creditscore dataframe
dfval = df

###########feautre reduction##############

dfval=dfval.iloc[0,:]
dfval= pd.DataFrame(dfval)
dfval = dfval.sort_values( by=[0], ascending=False)
sum(dfval.values)

for x in range(0,len(dfval)):
    dfvalt = dfval.iloc[:-x,:]
    if  sum(dfvalt.values)>0.25 and sum(dfvalt.values)<0.74 : # was 82 and 92 utastar
        dfval = dfval.iloc[:-x,:]
        

print(sum(dfval.values))
print(dfval)


#After correct selection
df=df[dfval.index]
l= list(dfval.index)
l.append("Class")
l.append("OverallValues")
l.append("OutRanks")

dfclass=dfclass[l]



#########################################################################################################
################### k means for feautre/alternatives  reduction ##############
########################################################################################################

#3 Using the elbow method to find out the optimal number of #clusters. 
#KMeans class from the sklearn library.
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold,train_test_split,cross_val_score,cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

Y=df
wcss=[]
#X = performanceTable.values
X = df.iloc[:240,:] #.transpose().values 
y=dfclass["Class"]

###nromalization if needed
#scaler = MinMaxScaler(feature_range=(0, 1))
#X = scaler.fit_transform(X)

## Kmeans cluster training / cluster finding
for i in range(1,11): 
    kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)
#kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.
#4.Plot the elbow graph
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#5 According to the Elbow graph we deterrmine the clusters number as #5. Applying k-means algorithm to the X dataset.
kmeans = KMeans(n_clusters=2, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
y_kmeans = kmeans.fit(X)

# We are going to use the fit predict method that returns for each #observation which cluster it belongs to. 
# The cluster to which #it belongs and it will return this cluster numbers into a 
# # single vector that is  called y K-means
#Predicted values
y_kmeans = kmeans.fit_predict(X)

##kmeans score?
kmeans.score(X)

####k-fold cross validation score 
scores = cross_val_score(kmeans, X, y[:240], cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_predict(kmeans, X, y[:240], cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#6 Visualising the clusters based on prediction 
#credit score values
plt.scatter(X.iloc[y_kmeans==0, 0], X.iloc[y_kmeans==0, 1], s=10, c='red', label ='Cluster 1' )
plt.scatter(X.iloc[y_kmeans==1, 0], X.iloc[y_kmeans==1, 1], s=10, c='blue', label ='Cluster 2')

#visual based on actual values values 
#Plot the centroid. This time we're going to use the cluster centres  
# attribute that returns here the coordinates of the centroid.
plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=100, c='green', label = 'Centroids' )
plt.title('Clusters k-means')
plt.show()


#credit score values
plt.scatter(X.iloc[:,:10],X.iloc[:,10:],alpha = 0.5)

plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=100, c='green', label = 'Centroids' )
plt.title('Clusters k-means')
plt.show()

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


plt.show()

#original values
#plt.scatter(X[:,:10],X[:,10:],alpha = 0.5)

#credit score values
plt.scatter(Y.iloc[:,:10],Y.iloc[:,10:],alpha = 0.5)

plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=100, c='green', label = 'Centroids' )
plt.title('Clusters k-means')
plt.show()

####lower bound #####

#for alternatives
dfclass.insert(dfclass.shape[1],"Pred",y_kmeans)

dft = dfclass[dfclass["Pred"]==0].idxmax() # or min  check data 
indx = dft["OutRanks"] # for utastar 
#indx = dft["Pred"] # for utadis
cined = dfclass.columns.get_loc("OverallValues")
lower_bound = dfclass.iloc[indx,cined]
print("Lower Bound value for cluster-->",lower_bound)




#####to excel alternatives results 
dfclass.to_excel(
    r"C:\Users\amichail\OneDrive - Raycap\Dokumente\Thes\german credit score dataset UCI\ResultsUtastar\UtastarKmeansAltResults.xlsx")




##### for feautres #############
dfclassT = dfclass.transpose()
dfclassT = dfclassT.iloc[:,:0]
for i in range(2,df.shape[1]):
    kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
    y_kmeans = kmeans.fit_predict(X)
    y_k= [y_kmeans[i] if i<20 else None for i in range(0,dfclass.shape[1])]
    
    dfclassT.insert(i-2,"Pred"+str(i),y_k)
    dfclassT = dfclassT.sort_values(by=['Pred'+str(i)],ascending=True)

##### to excel feautres results
dfclassT.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\KmeansFeautResults.xlsx")


###################################################################################
########confusion mattrix for accurancy calculation visulization###################
from sklearn.metrics import confusion_matrix,accuracy_score

#values transformation 
true_v = [0 if x==1 else 1 for x in dfclass["Class"]]
print("Num of Class 1 in dataset-->",480-sum(true_v),"/480")

#true_v = [0 if x<301 else 1 for x in range(0,479)]
#y_kmeans =  [0 if x<230 else 1 for x in range(0,479)]

acc= accuracy_score(true_v, y_kmeans, normalize=True, sample_weight=None)
print("Accurancy->",  acc)
#have to input outranks and create the predicted classification from boundries
mat = confusion_matrix(true_v ,y_kmeans,)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False),
           # xticklabels=dfclass.columns,
           # yticklabels=dfclass.index)
plt.xlabel('true label')
plt.ylabel('predicted label')


####spectral clustering 
X = df
from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels,
            s=50, cmap='viridis')