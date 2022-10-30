#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


data = pd.read_csv("C:\Ankur\Frankfurt\Classes\Introduction to Data Analytics in Business\Project\spotify.data.csv")# read data


# In[ ]:





# In[3]:


data.describe() #describe data


# In[4]:


data.head() # present first five data


# In[5]:


data.info()


# #Custom Color Palette
# red_blue = ["#19B5FE","#EF4836"]
# palette = sns.color_palette(red_blue)
# sns.set_palette(palette)
# sns.set_style("white")

# In[6]:


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


# In[7]:


pos_tempo


# In[8]:


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


# In[9]:


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


# In[10]:


df=data.drop(columns=["song_title","artist","Unnamed: 0","duration_ms","key","time_signature"])
df.head()


# In[11]:


df.fillna(0)


# In[12]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)


# In[13]:


from sklearn.preprocessing import MinMaxScaler


# In[14]:


scaler_data=MinMaxScaler()
df1=df.iloc[:,:]
df1.head()


# In[15]:


scaler_df1=scaler_data.fit_transform(df1)
scaler_df1


# In[16]:


df_final=pd.DataFrame(scaler_df1,columns=df1.columns)
df_final


# In[17]:


features = ["danceability","loudness","valence","tempo","liveness","mode","speechiness","instrumentalness","acousticness","energy"]


# In[18]:


train,test = train_test_split (df_final, test_size=0.25)
x_train=train[features]
y_train=train["target"]

x_test=train[features]
y_test=train["target"]


# In[19]:


RFC=RandomForestClassifier(max_depth=10)


# In[20]:


rfc_data=RFC.fit(x_train,y_train)
y_test


# In[21]:


y_pred=rfc_data.predict(x_test)


# In[22]:


y_pred


# In[23]:


score=accuracy_score(y_pred,y_test)
print(score)


# In[24]:


print("Accuracy using Decision Tree:",round(score,4)*100,"%")


# # Recommendation System

# In[25]:


from sklearn.cluster import KMeans


# In[38]:


df1 = df_final.drop(columns = "target")
df1.head()


# In[39]:


sse={}
for k in range(1,50):
    kmeans = KMeans(n_clusters=k).fit(df1)
    df1["cluster"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.title("Elbow method")
plt.xlabel("Number of cluster")
plt.show()


# In[40]:


kmeans=KMeans(n_clusters=15)
cluster=kmeans.fit_predict(df1)
df1


# In[41]:


df1["cluster"]=cluster
df1


# In[42]:


data["cluster"]=cluster
data.head()


# In[43]:


data.tail()


# In[44]:


def recomendation_system(song,amount):
    for i in range(len(data)):
        #first we need to check the cluster of song
        if song==data.song_title[i]:
            fea=data.cluster[i]
            target_fea=data.target[i]
            break
    
    count=0
    for j in range(len(data)-1):
        #print(j,"jjj")
        rec_fea=data.cluster[j]==fea  ##this line is checking other songs having same cluster   
        rec_target=data.target[j]==target_fea
        #print(rec_fea,rec_target)
        if rec_fea==True:
        #and rec_target==True:
            if count<(amount):
                #print(j,"jj")
                count+=1
                j+=1
                #print(count,"count")
                print(data.song_title[j])
                #print(data.target[j])
recomendation_system(song="Candy",amount=10)


# In[45]:


plt.figure(figsize=(30,10))
plt.scatter(y = 'instrumentalness', x = 'acousticness',c="cluster",cmap="inferno",data=df1)


# In[ ]:




