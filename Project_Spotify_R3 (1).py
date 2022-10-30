#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd  # DataFrame， Series
import numpy as np  # Scientific computing packags- Array

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split 

from matplotlib import pyplot as plt
import seaborn as sns


# # Spotify Song Attributes EDA
# - Import dataset
# - EDA to visualize data and observe structure
# - Train a classifier(Decision Tree)
# - Predict target using the trained classifier

# In[45]:


data = pd.read_csv("C:\Ankur\Frankfurt\Classes\Introduction to Data Analytics in Business\Project\spotify.data.csv")# read data


# In[46]:


data.describe() #describe data


# In[47]:


data.head() # present first five data


# In[48]:


data.info() # information about data


# #Custom Color Palette
# red_blue = ["#19B5FE","#EF4836"]
# palette = sns.color_palette(red_blue)
# sns.set_palette(palette)
# sns.set_style("white")

# In[49]:


pos_tempo=data[data["target"]==1]["tempo"] #if target=1 postive tempo
neg_tempo=data[data["target"]==0]["tempo"] #if target=0 negative tempo

pos_dance=data[data["target"]==1]["danceability"] #if target=1 postive danceability
neg_dance=data[data["target"]==0]["danceability"] #if target=0 negative danceability

pos_duration=data[data["target"]==1]["duration_ms"] #if target=1 postive duration
neg_duration=data[data["target"]==0]["duration_ms"] #if target=0 negative duration

pos_acousticness=data[data["target"]==1]["acousticness"] #if target=1 postive acousticness
neg_acousticness=data[data["target"]==0]["acousticness"] #if target=0 negative acousticness

pos_energy=data[data["target"]==1]["energy"] #if target=1 postive energy
neg_energy=data[data["target"]==0]["energy"] #if target=0 negative energy

pos_instrumentalness=data[data["target"]==1]["instrumentalness"] #if target=1 postive instrumentalness
neg_instrumentalness=data[data["target"]==0]["instrumentalness"] #if target=0 negative instrumentalness

pos_key=data[data["target"]==1]["key"] #if target=1 postive key
neg_key=data[data["target"]==0]["key"] #if target=0 negative key

pos_liveness=data[data["target"]==1]["liveness"]
neg_liveness=data[data["target"]==0]["liveness"] 

pos_loudness=data[data["target"]==1]["loudness"]
neg_loudness=data[data["target"]==0]["loudness"] 

pos_mode=data[data["target"]==1]["mode"]
neg_mode=data[data["target"]==0]["mode"] 

pos_time_signature=data[data["target"]==1]["time_signature"]
neg_time_signature=data[data["target"]==0]["time_signature"] 

pos_valence=data[data["target"]==1]["valence"]
neg_valence=data[data["target"]==0]["valence"] 

pos_speechiness=data[data["target"]==1]["speechiness"]
neg_speechiness=data[data["target"]==0]["speechiness"] 


# In[50]:


pos_tempo


# In[51]:


fig=plt.figure(figsize=(12,8)) #define the size of the figure
plt.title("Song Tempo Like/ Dislike Distribution") #define the title of the figure

pos_tempo.hist(alpha=0.5, bins=30,label= "Like") # alpha means transparency 透明度 ，bins means how many columns the data is distributed
neg_tempo.hist(alpha=0.5, bins=30, label="Dislike")

plt.legend(loc="upper right",frameon =True,edgecolor="blue",facecolor="white",title='Like VS Dislike') 
# legend command is to edit the label; Loc means the location of the label,; 
# frameon=true means the label box has a frame
# edgecolor defines the color of the frame
# facecolor defines the color of the backgound
# title ： Add a title of the label


# In[52]:


fig2= plt.figure(figsize=(22,22))

#Danceability  question ax4？
ax2=fig2.add_subplot(4,4,1)
ax2.set_xlabel('Danceability')
ax2.set_ylabel('Count')
ax2.set_title('Song Danceabiliy Like Distribution')
pos_dance.hist(alpha=0.5,bins=30)
neg_dance.hist(alpha=0.5,bins=30)

#Duration_ms 
ax3=fig2.add_subplot(4,4,2)
ax3.set_xlabel('Duration_ms')
ax3.set_ylabel('Count')
ax3.set_title('Song Duration Like Distribution')
pos_duration.hist(alpha=0.5,bins=50)
neg_duration.hist(alpha=0.5,bins=50)

#Acousticness  
ax4=fig2.add_subplot(4,4,3)
ax4.set_xlabel('Acousticness')
ax4.set_ylabel('Count')
ax4.set_title('Song Acousticness Like Distribution')
pos_acousticness.hist(alpha=0.5,bins=15)
neg_acousticness.hist(alpha=0.5,bins=15)

#Energy  
ax5=fig2.add_subplot(4,4,4)
ax5.set_xlabel('Energy')
ax5.set_ylabel('Count')
ax5.set_title('Song Energy Like Distribution')
pos_energy.hist(alpha=0.5,bins=30)
neg_energy.hist(alpha=0.5,bins=30)

#Instrumentalness  
ax6 = fig2.add_subplot(4,4,5)
ax6.set_xlabel('Instrumentalness')
ax6.set_ylabel('Count')
ax6.set_title('Song Instrumentalness Like Distribution')
pos_instrumentalness.hist(alpha=0.5,bins=20)
neg_instrumentalness.hist(alpha=0.5,bins=20)

#Key  
ax7 = fig2.add_subplot(4,4,6)
ax7.set_xlabel('Key')
ax7.set_ylabel('Count')
ax7.set_title('Song Key Like Distribution')
pos_key.hist(alpha=0.5,bins=12)
neg_key.hist(alpha=0.5,bins=12)

#Liveness  
ax8 = fig2.add_subplot(4,4,7)
ax8.set_xlabel('Liveness')
ax8.set_ylabel('Count')
ax8.set_title('Song Liveness Like Distribution')
pos_liveness.hist(alpha=0.5,bins=30)
neg_liveness.hist(alpha=0.5,bins=30)

#Loudness  
ax9 = fig2.add_subplot(4,4,8)
ax9.set_xlabel('Loudness')
ax9.set_ylabel('Count')
ax9.set_title('Song Loudness Like Distribution')
pos_loudness.hist(alpha=0.5,bins=30)
neg_loudness.hist(alpha=0.5,bins=30)

#Mode 
ax10 = fig2.add_subplot(4,4,9)
ax10.set_xlabel('Mode')
ax10.set_ylabel('Count')
ax10.set_title('Song Mode Like Distribution')
pos_mode.hist(alpha=0.5,bins=2)
neg_mode.hist(alpha=0.5,bins=2)

#Time_signature 
ax11 = fig2.add_subplot(4,4,10)
ax11.set_xlabel('Time_signature ')
ax11.set_ylabel('Count')
ax11.set_title('Song Time_signature Like Distribution')
pos_time_signature.hist(alpha=0.5,bins=8)
neg_time_signature.hist(alpha=0.5,bins=8)

#Valence 
ax12 = fig2.add_subplot(4,4,11)
ax12.set_xlabel('Valence')
ax12.set_ylabel('Count')
ax12.set_title('Song Valence Like Distribution')
pos_valence.hist(alpha=0.5,bins=30)
neg_valence.hist(alpha=0.5,bins=30)

#Speechiness 
ax13 = fig2.add_subplot(4,4,12)
ax13.set_xlabel('Speechiness ')
ax13.set_ylabel('Count')
ax13.set_title('Song Speechiness  Like Distribution')
pos_speechiness .hist(alpha=0.5,bins=30)
neg_speechiness .hist(alpha=0.5,bins=30)


# In[53]:


data_R1=data.drop(columns=["song_title","artist","Unnamed: 0","duration_ms","key","time_signature"])
data_R1.head()


# In[54]:


data_R1.fillna(0)


# # Pearson’s Correlation

# The Pearson correlation coefficient (named for Karl Pearson) can be used to summarize the strength of the linear relationship between two data samples.
# Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))

# In[55]:


plt.figure(figsize=(20,10))
sns.heatmap(data_R1.corr(),annot=True)


# # MinMax Scaler

# Data Scaling is a data preprocessing step for numerical features. In MinMax Scaler the minimum of feature is made equal to zero and the maximum of feature equal to one. MinMax Scaler shrinks the data within the given range, usually of 0 to 1. It transforms data by scaling features to a given range. It scales the values to a specific value range without changing the shape of the original distribution.

# In[56]:


from sklearn.preprocessing import MinMaxScaler


# In[57]:


scaler_data=MinMaxScaler()
data_R2=data_R1.iloc[:,:]
data_R1.head()


# In[58]:


scaler_df1=scaler_data.fit_transform(data_R2)
scaler_df1


# In[59]:


data_R3=pd.DataFrame(scaler_df1,columns=data_R2.columns)
data_R3


# In[60]:


data_R3.shape


# In[61]:


features = ["danceability","loudness","valence","tempo","liveness","mode","speechiness","instrumentalness","acousticness","energy"]


# In[62]:


train,test = train_test_split (data_R3, test_size=0.25)
x_train=train[features]
y_train=train["target"]

x_test=train[features]
y_test=train["target"]


# # Random Forest Classifier using Scikit-learn

# The Random forest classifier creates a set of decision trees from a randomly selected subset of the training set. It is basically a set of decision trees (DT) from a randomly selected subset of the training set and then It collects the votes from different decision trees to decide the final prediction.

# In[63]:


RFC=RandomForestClassifier(max_depth=None,max_samples = 1200)


# In[64]:


rfc_data=RFC.fit(x_train,y_train)
y_test


# In[65]:


RFC.score(x_train,y_train)


# In[66]:


RFC.score(x_test,y_test)


# In[67]:


y_pred=rfc_data.predict(x_test)


# In[68]:


y_pred


# In[69]:


score=accuracy_score(y_pred,y_test)
print(score)


# In[70]:


print("Accuracy using Decision Tree:",round(score,4)*100,"%")


# # Recommendation System by KMeans Clustering

# K-Means Clustering is an Unsupervised Learning algorithm, which groups the unlabeled dataset into different clusters. Here K defines the number of pre-defined clusters that need to be created in the process, as if K=2, there will be two clusters, and for K=3, there will be three clusters, and so on.

# # The working of the K-Means algorithm is explained in the below steps:

# Step-1: Select the number K to decide the number of clusters.
# 
# Step-2: Select random K points or centroids. (It can be other from the input dataset).
# 
# Step-3: Assign each data point to their closest centroid, which will form the predefined K clusters.
# 
# Step-4: Calculate the variance and place a new centroid of each cluster.
# 
# Step-5: Repeat the third steps, which means reassign each datapoint to the new closest centroid of each cluster.
# 
# Step-6: If any reassignment occurs, then go to step-4 else go to FINISH.
# 
# Step-7: The model is ready.

# In[71]:


from sklearn.cluster import KMeans


# In[72]:


data_R4 = data_R3.drop(columns = ["target","mode"])
data_R4.head()


# In[73]:


plt.figure(figsize = (18,10))
sse = []
for k in range(2,20):
    kmeans = KMeans(n_clusters=k).fit(data_R4)
    sse.append(kmeans.inertia_)
plt.plot(range(2,20),sse)
plt.title("Elbow method")
plt.xlabel("Number of cluster")
plt.show()


# In[74]:


kmeans=KMeans(n_clusters=7)
cluster=kmeans.fit_predict(data_R4)
cluster.shape


# In[75]:


data_R5 = data_R4


# In[76]:


data_R4["cluster"]=cluster
data_R4


# In[83]:


data_R4["cluster"].value_counts()


# In[77]:


original_data = data
original_data["cluster"]=cluster
original_data.head()


# In[78]:


original_data.tail()


# In[79]:


def recomendation_system(song,amount):
    for i in range(len(original_data)):
        #first we need to check the cluster of song
        if song == data.song_title[i]:
            fea = data.cluster[i]
            break
    
    count=0
    for j in range(len(original_data)-1):
        rec_fea=original_data.cluster[j]==fea  ##this line is checking other songs having same cluster   
        if rec_fea==True:
            if count<(amount):
                count+=1
                j+=1
                print(original_data.song_title[j])
recomendation_system(song="Candy",amount=10)


# In[86]:


plt.figure(figsize=(30,20))
plt.scatter(y = 'instrumentalness', x = 'acousticness',c="cluster",cmap="viridis",data=data,s=200)
plt.show()


# In[87]:


plt.figure(figsize=(20,10))
sns.heatmap(data_R4.corr(),annot=True)


# In[88]:


plt.figure(figsize=(30,20))
plt.scatter(y = 'energy', x = 'acousticness',c="cluster",cmap="viridis",data=data,s=200)
plt.show()


# In[128]:


plt.figure(figsize=(30,20))
plt.scatter(y = 'loudness', x = 'acousticness',c="cluster",cmap="viridis",data=df1,s=200)
plt.show()


# In[89]:


plt.figure(figsize=(30,20))
plt.scatter(y = 'loudness', x = 'energy',c="cluster",cmap="viridis",data=data,s=200)
plt.show()

