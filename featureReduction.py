
import pandas as pd
import random  
import sys 
#sys.path.append("E:/Google Drive/PC SHIT/HMMY/Diplomatiki/methods/MCDA")
from  Pyth.UTASTAR import *

import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.image import imread
import seaborn as sns
from sklearn.datasets.samples_generator import (make_blobs,
                                                make_circles,
                                                make_moons)
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from warnings import filterwarnings
from math import pi
import pandas
from pandas.plotting import parallel_coordinates



#import original 
df = pd.read_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\multicriteria matrix.xlsx",
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

# Per column change values if needed
# Replace true zero

df["check_account"].replace("4", "0", regex=True, inplace=True)
df["savings_account"].replace("5", "0", regex=True, inplace=True)
df["employment"].replace("1", "0", regex=True, inplace=True)
df["debtors_guarantors"].replace("1", "0", regex=True, inplace=True)
df["property"].replace("4", "0", regex=True, inplace=True)
df["installment_plans"].replace("3", "0", regex=True, inplace=True)
df["job"].replace("1", "0", regex=True, inplace=True)
df["telephone"].replace("1", "0", regex=True, inplace=True)


# Set index name
df.columns.name = "Alternatives"

# Reset index
df = df.reset_index(drop=True)

# Convert all dataframe values to integer
df = df.apply(pd.to_numeric)

#480 best solution or first 100 from other sheet 
#### first 100

nr= [x for x in range(480)]
df= df.iloc[nr,:]

# the performance table
# removing the class column
performanceTable = df.iloc[:,:-1]# -3 adjusted for removed columns 
#print("\nPerformance Table \n", performanceTable)


#import Credit score resutls
df = pd.read_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\Results.xlsx",
)
dfclass = df.iloc[1:,1:] #-2
#dfclass = dfclass.drop(["telephone"],axis=1)
#dfclass = dfclass.drop(["foreign"],axis=1)
dfclass = dfclass.reset_index(drop=True)

df = df.iloc[1:,1:-3]## adjusted for removed columns
df = df.reset_index(drop=True)


##########################################################################################################
############################ VISULIZATION ################################################################
##########################################################################################################

##kmeans feautre reduction 
#data

dfplot = df
 
#%matplotlib inline
sns.set_context('notebook')
plt.style.use('fivethirtyeight')
filterwarnings('ignore')


# Plot the data
plt.figure(figsize=(6, 6))
plt.scatter(dfplot.iloc[:, 1], dfplot.iloc[:,2],)
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of  data')
 

dfplot= performanceTable

# Plot the data
plt.figure(figsize=(6, 6))
plt.scatter(dfplot.iloc[:, 1], dfplot.iloc[:,2],)
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of  data')
 
# matrix to be used for k menas

################################################
###########parralel coordinate plot ############
#################################################
dfplot = df
pd.DataFrame(dfplot).plot()
plt.show

# matrix to be used for k menas

###########parralel coordinate plot ############
dfplot = performanceTable
pd.DataFrame(dfplot).plot()
plt.show

########################################
######### RADAR GRAPH  ##################
#########################################
dfplot = df
#dfplot = performanceTable

categories=list(dfplot)
N = len(categories)

values=dfplot.loc[0].values.flatten().tolist()
values += values[:1]
values#=dfplot.loc[1].values.flatten().tolist()
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels original data
#ax.set_rlabel_position(0)
#plt.yticks([5,10,15], ["5","10","15"], color="grey", size=7)
#plt.ylim(0,20) # 20 for perfoncatable 0.05 for df

# Draw ylabels credit score data 
ax.set_rlabel_position(0)
plt.yticks([0.01,0.02,0.03], ["0.01","0.02","0.03"], color="grey", size=7)
plt.ylim(0,0.05) # 2

# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

##############################################
############# Parallel plot with pandas ######
###############################################

# Make the plot
#with original values 
data = performanceTable.iloc[:,:10].transpose() # first five atrributes
data.insert(0,"Class",data.index)

parallel_coordinates(data, "Class", colormap=plt.get_cmap("Set2") )
plt.show()
### second half 
data = performanceTable.iloc[:,10:]
data.insert(0,"Class",data.index)

parallel_coordinates(data, "Class", colormap=plt.get_cmap("Set2") )
plt.show()

######## with credit score ########
dfplot = df

data = dfplot.iloc[:,:10] #.transpose()  ## first half attributes
data.insert(0,"Class",data.index)

parallel_coordinates(data, "Class", colormap=plt.get_cmap("Set2") )
plt.show()

###
data = dfplot.iloc[:,10:]  ##second half attributes
data.insert(0,"Class",data.index)

parallel_coordinates(data, "Class" , colormap=plt.get_cmap("Set2"))
plt.show()



#############################################
######## Histograms with pandas##### pair of 2
############################################## 
dfplot = df

dfplot.hist(figsize=[25,15])
plt.tight_layout()

dfplot.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=10, ylabelsize=10, grid=True )
plt.tight_layout()
plt.tight_layout(rect=(1, 1, 0, 0))

performanceTable.hist(figsize=[20,10])
plt.tight_layout()

performanceTable.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=10, ylabelsize=10, grid=False )
plt.tight_layout()
plt.tight_layout(rect=(0, 0, 5, 5))

####################################
###### Bar Plot######### per feaure 
#####################################

for i in range(0,performanceTable.shape[1]-1):
            fig = plt.figure(figsize = (6, 4))
            title = fig.suptitle(performanceTable.iloc[:,i].name, fontsize=14)
            fig.subplots_adjust(top=0.85, wspace=0.3)

            ax = fig.add_subplot(1,1, 1)
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Values") 
            w_q = performanceTable.iloc[:,i].value_counts() ## original data
            #w_q = dfplot['check_account'].value_counts()
            w_q = (list(w_q.index), list(w_q.values))
            ax.tick_params(axis='both', which='major', labelsize=8.5)
            bar = ax.bar(w_q[1], w_q[0], color='steelblue', 
                    edgecolor='black', linewidth=1)
            


dfplot = df
fig,axs =plt.subplots(figsize = (25, 15), ncols = 4,nrows= 5)
i=0
for nc in range(0,4):
    for nr in range(0,5):
        if (i!=20):  
            #for i in range(0,dfplot.shape[1]):
            #fig = plt.figure(figsize = (6, 4))
            #title = fig.suptitle(dfplot.iloc[:,i].name, fontsize=14)
            #fig.subplots_adjust(top=0.85, wspace=0.3)
            #ax = fig.add_subplot(4,5,1)
            #axs.set_xlabel("Frequency")
            axs[nr,nc].set_title(performanceTable.iloc[:,i].name)
            #axs[nr,nc].set_ylabel(performanceTable.iloc[:,i].name) 
            w_q = dfplot.iloc[:,i].value_counts() ## original data
            w_q = (list(w_q.index), list(w_q.values))
            #ax.tick_params(axis='both', which='major', labelsize=8.5)
            bar = axs[nr,nc].bar(w_q[1], w_q[0], color='steelblue', 
                edgecolor='black', linewidth=1  ) 
            i=i+1 
            
        else: break
fig.tight_layout()


#############################
# Correlation Matrix Heatmap##
#############################

f, ax = plt.subplots(figsize=(15, 10))
corr = performanceTable.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)

#########################
# Pair-wise Scatter Plots
############################
dfplot=performanceTable

dfplot = df

cols = dfplot.iloc[:,:5].columns
pp = sns.pairplot(dfplot[cols], size=2, aspect=1,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Credit card Pairwise Plots', fontsize=14)




#########################################################################################################
################### k means for feautre/alternatives  reduction ##############
########################################################################################################


#3 Using the elbow method to find out the optimal number of #clusters. 
#KMeans class from the sklearn library.
from sklearn.cluster import KMeans

wcss=[]
#dfk = performanceTable.values
dfk = df.transpose().values 

for i in range(1,20): 
    kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )

    kmeans.fit(dfk)

    wcss.append(kmeans.inertia_)
#kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.
#4.Plot the elbow graph
plt.plot(range(1,20),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#5 According to the Elbow graph we deterrmine the clusters number as #5. Applying k-means algorithm to the X dataset.
kmeans = KMeans(n_clusters=2, init ='k-means++', max_iter=300, n_init=10,random_state=0 )

# We are going to use the fit predict method that returns for each #observation which cluster it belongs to. 
# The cluster to which #it belongs and it will return this cluster numbers into a 
# # single vector that is  called y K-means
X = dfk
y_kmeans = kmeans.fit_predict(X)

#6 Visualising the clusters based on prediction 
#original values 
#plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=10, c='red', label ='Cluster 1' )
#plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=10, c='blue', label ='Cluster 2')

#credit score values
plt.scatter(X.iloc[y_kmeans==0, 0], X.iloc[y_kmeans==0, 1], s=10, c='red', label ='Cluster 1' )
plt.scatter(X.iloc[y_kmeans==1, 0], X.iloc[y_kmeans==1, 1], s=10, c='blue', label ='Cluster 2')

#visual based on actual values values 
#Plot the centroid. This time we're going to use the cluster centres  
# attribute that returns here the coordinates of the centroid.
plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=100, c='green', label = 'Centroids' )
plt.title('Clusters k-means')
#plt.xlim(0.0025,0.0045)# predicted 18 columns
#plt.ylim(0.0,0.5) 
plt.xlim(0.0025,0.006)# predicted
plt.ylim(0.0,0.5)
plt.show()

#original values
#plt.scatter(X[:,:10],X[:,10:],alpha = 0.5)

#credit score values
plt.scatter(X.iloc[:,:10],X.iloc[:,10:],alpha = 0.5)

plt.scatter(kmeans.cluster_centers_[:, :1], kmeans.cluster_centers_[:, 2:3], s=100, c='green', label = 'Centroids' )
plt.title('Clusters k-means')
plt.show()


####lower bound #####

#for alternatives
dfclass.insert(dfclass.shape[1],"Pred",y_kmeans)

# for feautres 
y_k= [y_kmeans[i] if i<20 else None for i in range(0,dfclass.shape[1])]
dfclassT = dfclass.transpose()
dfclassT.insert(0,"Pred",y_k)

dft = dfclass[dfclass["Pred"]==1].idxmin()
indx = dft["OutRanks"]
cined = dfclass.columns.get_loc("OverallValues")
lower_bound = dfclass.iloc[indx,cined]
print("Lower Bound value for cluster-->",lower_bound)

#####to excel alternatives results 
dfclass.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\KmeansAltResults.xlsx")

##### to excel feautres results
dfclass.to_excel(
    r"E:\Google Drive\PC SHIT\HMMY\Diplomatiki\german credit score dataset UCI\ResultsUtastar\KmeansFeautResults.xlsx")


###################################################################################
########confusion mattrix for accurancy calculation visulization###################
from sklearn.metrics import confusion_matrix,accuracy_score

#values transformation 
true_v = [0 if x==1 else 1 for x in dfclass["Class"]]
print("Num of Class 1 in dataset-->",480-sum(true_v),"/480")

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
X = dfk
from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels,
            s=50, cmap='viridis')


###########################################################################
#######################Silhouette plot #####################################
#############################################################################
####silhouette value for right choice of cluster number for criteria reduction 

from sklearn.datasets import make_blobs 
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
  
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
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

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

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
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

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


