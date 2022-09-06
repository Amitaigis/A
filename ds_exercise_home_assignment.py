#!/usr/bin/env python
# coding: utf-8

# # Introduction

# - This exercise notebook demonstrates some super basic example of our domain challenges in Upstream
# - The dataset you are going to work with is a batch of rows demonstrating vehicle messages reporting to a cloud server during some period of time
# - The exercise deals with data exploration and processing, manual anomaly investigation and prediction model toy example
# - Please fill cells only within the # Edit # sections
# - Answers to questions can be written in comments like the ones you read now
# - Although not needed, you may add any import that you like in order to complete the task
# - When you finish the task please make sure it runs from start to end with no errors
# - Please send the notebook with its output cells, as well as the "exp_df.csv" file, back to elad@upstream.auto and within 48 hours from the exercise email delivery time
# - If you can't complete some of the tasks, try to continue and complete the notebook without it
# - Enjoy and good luck :)

# # Imports

# In[330]:


################ NOT FOR EDIT #################
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
################ NOT FOR EDIT #################


# # Get Data

# In[331]:


# 1. Read the vehicle_messages.csv file into a dataframe
# 2. Print the shape of the dataframe 
# 3. Show its first rows
################ EDIT #################
vehicles_messages = pd.read_csv("vehicles_messages.csv")
print(vehicles_messages.shape)
vehicles_messages.head()


################# EDIT ################


# # Explore

# In[332]:


# The given timestamp (the time when the message was sent from the vehicle to the server) is epoch time (unix time) milliseconds
# Show the timestamp column with the timestamp's represntaion modified to pandas "datetime" (timestamps should be arround July 2022)
#### EDIT #####
vehicly_time=pd.DataFrame(pd.to_datetime(vehicles_messages['timestamp'], unit='ms'))
vehicly_time

#### EDIT #####


# In[333]:


# 1. Create a new dataframe (call it "df") and replace the timestamp column with the new timestamp represantaion
# 2. Set timestamp as index *without dropping the timestamp column* 
#### EDIT #####
df=vehicles_messages.copy()
df['timestamp']=pd.to_datetime(vehicles_messages['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True,drop=False)

#### EDIT #####


# In[334]:


# Sort the dataframe index and show it again
#### EDIT #####
df=df.sort_index()
df
#### EDIT #####


# - VIN is the unique ID of every vehicle
# - Manufacturer, Year and Model are meta data info of each vehicle
# - timestamp is the time when the message was sent from the vehicle to the server
# - Latitude and longitude are reported GPS location at the time of 
# - Velocity is the vehicle motion speed in Km/h

# In[335]:


# 1. Print the number of distinct vehicles (vins)
# 2. Show the dataframe decriptives (both numeric and categoric columns)
#### EDIT #####
print("We have "+ str(df['vin'].nunique())+ " different vehicles")


display(df.describe().style.set_table_attributes("style='display:inline'").set_caption('Decriptives of numeric values'))
df[['timestamp','model','manufacturer','vin']].describe().style.set_table_attributes("style='display:inline'").set_caption('Decriptives of categoric values')

#### EDIT #####


# In[336]:


################ NOT FOR EDIT - NOT A MUST #################
get_ipython().system('pip install geopandas')
get_ipython().system('pip install Shapely')
################ NOT FOR EDIT - NOT A MUST #################


# The cell below will print a map with locations from sample of some vehcile messages

# In[337]:


################ NOT FOR EDIT - NOT A MUST #################
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point

df_ = df.sort_index()[:100]
geometry = [Point(xy) for xy in zip(df_["longitude"], df_["latitude"])]
gdf = GeoDataFrame(df_, geometry=geometry)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf.plot(ax=world.plot(figsize=(15, 15)), marker='o', color='red', markersize=15)
################ NOT FOR EDIT - NOT A MUST #################


# # Find Anomaly

# In[380]:


# 1. Show the vehicle velocities distribution
# 2. Try to explain its charestaristics and peaks
# 3. What whould be suspected as anomaly?
#### EDIT #####
#1)

ax=df["velocity"].plot.hist(bins=18, alpha=0.5,figsize=(30,10),title='The vehicle velocities distribution')
params = {'ytick.labelsize': 20,
          'axes.titlesize': 40,
          'xtick.labelsize': 20,
          'axes.labelsize': 25}
plt.rcParams.update(params)

#2) Most of the data for velocities are between 0 and 150. More observations of velocities continue to a little over 200 and there is an unusual observation of 450. There is a decrease in observations in values between 30 and 70 and between 70 and 130 there is an increase and then there are extreme decreases in observations.
#   We get a local maximum at sea between 0 and 30 and an absolute maximum between 100 and 130.

#3) Observations that the velocities get values greater than 400.

#### EDIT #####


# In[340]:


# Explore the data to find what is common between the anomalies
#### EDIT #####

# Let's look at the data that receives a value of velocities that is greater than 400.
anomaly_data=df[df["velocity"]>400]
display(anomaly_data)

# We will note that the vehicle 4T1BF1FK8FUB03575 appears in all the observations that appear before us.

# We will check whether it is possible that the vehicle 4T1BF1FK8FUB03575 has a problem and 
# therefore we receive unusual observations and therefore we will look at all the messages 
# received from the vehicle 4T1BF1FK8FUB03575 and compare the data.

vin_4T1BF1_data=df[df["vin"]=="4T1BF1FK8FUB03575"]

equal_data=anomaly_data.equals(vin_4T1BF1_data)
 
print("We got " +str(equal_data))

# We will check whether the vehicle 4T1BF1FK8FUB03575 different values of velocity 
# or whether the vehicle sent the same value all the time.

print("We got " +str(vin_4T1BF1_data["vin"].nunique()))


# We received that all the velocities above 400 belong to the vehicle 4T1BF1FK8FUB03575 
# and that the vehicle 4T1BF1FK8FUB03575 did not send any number other than 450.

#### EDIT #####


# In[381]:


# 1. Create a new dataframe and eliminate the anomoulous messages from it
# 2. Show the velocity distribution again
#### EDIT #####

vehicles_messages_data=df[df["velocity"]!=450]

ax=vehicles_messages_data["velocity"].plot.hist(bins=14, alpha=0.5,figsize=(30,10),title='The vehicle velocities distribution without anomoulous')
params = {'ytick.labelsize': 20,
          'axes.titlesize': 40,
          'xtick.labelsize': 20,
          'axes.labelsize': 25}
plt.rcParams.update(params)

#### EDIT #####


# # Predict Velocities

# - In this section you will preprocess data, build a model and evaluate it
# - You will use every 20m average from every single vehicle to predict its next 20m average

# ## Preprocess Data - Get Vehicles of Interest

# In[158]:


# Show all the vehicles total count of messages per manufacturer and model
#### EDIT #####


def count_messeges(data):
    manufacturer_freq=pd.DataFrame(data["manufacturer"].value_counts())
    grouped=data.groupby(['manufacturer','model']).count()
    grouped=pd.DataFrame(grouped["vin"])
    grouped['Amount of messages'] = grouped.groupby([grouped.index.get_level_values(0)])["vin"].transform('sum')
    grouped=grouped.set_index([grouped.index.get_level_values(0),"Amount of messages",grouped.index.get_level_values(1)])
    grouped.rename({'vin': 'Amount of messages'}, axis=1, inplace=True)
    grouped=grouped.set_index([grouped.index.get_level_values(0),grouped.index.get_level_values(1),grouped.index.get_level_values(2),"Amount of messages"])
    return(grouped)
count_messeges(vehicles_messages_data)
#### EDIT #####


# In[159]:


# 1. Create a new dataframe from Toyota and Volksvagen vehicles (manufacturer) only
# 2. Show the total counts again (as was done in the previuos cells)
#### EDIT #####
Toyota_Volksvagen_data=vehicles_messages_data.loc[(vehicles_messages_data['manufacturer'] =="Toyota" ) | (vehicles_messages_data['manufacturer'] == "Volkswagen")]
count_messeges(Toyota_Volksvagen_data)
#### EDIT #####


# ## Dataset Creation

# In[49]:


# Show the start and end time for each vechile
#### EDIT #####
start_time=Toyota_Volksvagen_data.loc[Toyota_Volksvagen_data.groupby('vin').timestamp.idxmin()][['vin','timestamp']]
start_time.rename({'timestamp': 'start timestamp'}, axis=1, inplace=True)
end_time=Toyota_Volksvagen_data.loc[Toyota_Volksvagen_data.groupby('vin').timestamp.idxmax()][['vin','timestamp']]
end_time.rename({'timestamp': 'end timestamp'}, axis=1, inplace=True)
pd.merge(start_time, end_time, on='vin', how='outer')



#### EDIT #####


# In[384]:


# 1. Show the distribution of messages count per 20 minutes bin (before the previuos aggregation)
# 2. Validate the aggregation: do count per 20 minutes makes sense to infer average from?

#### EDIT #####

per20_data = Toyota_Volksvagen_data["timestamp"].groupby(pd.Grouper(freq='20Min'))
ax=per20_data.count().plot(kind="hist",alpha=0.5,figsize=(30,10),title='The messages count per 20 minutes distribution without anomoulous')

params = {'ytick.labelsize': 20,
          'axes.titlesize': 40,
          'xtick.labelsize': 20,
          'axes.labelsize': 25}
plt.rcParams.update(params)

# 2) Yes. There is a distribution that resembles normal even though there are a few observations on the side that impair the shape. And it's a shape that's good for trying to predict things related to speed

#### EDIT #####


# In[388]:


# 1. Create a new dataframe having one data column of the mean velocity per 20 minutes per vechile
# (Use resmaple for this task. make sure that "vin" and "timestamp" are multiindex after creating this dataframe) 
# 2. Show table
#### EDIT #####
data_per_20=Toyota_Volksvagen_data[["timestamp","vin","velocity"]].groupby(['vin']).resample('20min',on='timestamp').mean()
data_per_20

#### EDIT #####


# In[317]:


# 1.Create a new dataframe that has the "vin" and "timestamp" as columns again by reseting the index and keeping them
# 2. Set timestamp as index again, and show the table
#### EDIT #####
data_vin_time=Toyota_Volksvagen_data[["timestamp","vin"]]
data_vin_time=data_vin_time.sort_index()
data_vin_time
#### EDIT #####


# In[448]:


# 1. Create a new column (the target column) having the 20 minutes velocity mean prior the current one, 
#    for every row and *for each vehilcle separately* 
# 2. Show df
#### EDIT #####
past=Toyota_Volksvagen_data[["timestamp","vin","velocity"]].groupby(['vin']).resample('20min', loffset=pd.Timedelta("+00:20:00")).mean()
past.rename({'velocity': 'prior'}, axis=1, inplace=True)
data_per_20.rename({'velocity': 'predict'}, axis=1, inplace=True)
merged_df = pd.concat([past,data_per_20], axis=1)
merged_df = merged_df.reset_index()
merged_df.set_index('timestamp', inplace=True,drop=False)
merged_df=merged_df.sort_index()
merged_df.index.names = ['index']
merged_df
#### EDIT #####


# In[457]:


# 1. Create a new dataset, remove rows with nulls 
# 2. show table
#### EDIT #####
merged_df=merged_df.dropna()
merged_df
#### EDIT #####


# In[458]:


# 1. save dataframe as "exp_df.csv"
#### EDIT #####
merged_df.to_csv("exp_df.csv")
#### EDIT #####


# ## Create Expirement

# In[463]:


#### NOT FOR EDIT #####
df_ml = pd.read_csv("exp_df.csv")
#### NOT FOR EDIT #####


# In[464]:


# 1. Define x and y from the problem (x is the prior mean velocity, y is the current mean velocity)
# 2. print thier shapes
#### EDIT #####
df_ml.set_index('index', inplace=True)
print(df_ml.shape)
#### EDIT #####


# In[485]:


# Split the dataset to train and test (80%, 20%), show shapes of x,y (train and test)
# Explaing how did you choose what data is in the train set and what goes to the test set

#### EDIT #####


vin_values=set(df_ml["vin"])
train_data=pd.DataFrame(columns = ['vin', 'timestamp', 'prior',"predict"])
train_data.index.names = ['index']
test_data=pd.DataFrame(columns = ['vin', 'timestamp', 'prior',"predict"])
test_data.index.names = ['index']


for name in vin_values:
    vin_data=df_ml[df_ml["vin"]==name]
    train_name=vin_data.head(int(len(vin_data)*(80/100)))
    test_name=vin_data.tail(int(len(vin_data)*(20/100))) 
    train_data=pd.concat([train_data,train_name])
    test_data=pd.concat([test_data,test_name])


#### EDIT #####


# In[499]:


# 1. Create a decision tree regressor object to be used for our prediction task
# 2. Reshape x_train and x_test with x_.values.reshape(-1, 1)
# 3. fit the model, than predict with test set into y_hat
#### EDIT #####
from sklearn.metrics import mean_squared_error
y_train=train_data["predict"]
y_test=test_data["predict"]
x_train=train_data["prior"]
x_test=test_data["prior"]
x_train=x_train.values.reshape(-1, 1)
x_test=x_test.values.reshape(-1, 1)
regressor = DecisionTreeRegressor(max_depth=2)
regressor.fit(x_train, y_train)
print(mean_squared_error(regressor.predict(x_train),y_train))
y_1 = regressor.predict(x_test)
y_1
print(mean_squared_error(y_test, y_1))



#### EDIT #####


# ## Evaluate

# In[ ]:


# 1. How do you know how good is the model?
# 2. What evaluation metric will be relevant here?
# 3. Evaluate the model error and print the result
#### EDIT #####

#### EDIT #####


# In[ ]:


# investiage the error:
# 1. Assuming that every velocity in this dataset was radnomaly sampled from a known same distribution, 
# *regardless the timestamp and the vehicle identity*, what would you expect from the model evaluated error?
# 2. Is the error what you expected it to be? If yes, explain why
# 3. If not, what can be changed in the notebook to get the expected error?
# 4. Demonstrate your answers and/or assumptions with graphs/histograms/charts
#### EDIT #####

#### EDIT #####

