#!/usr/bin/env python
# coding: utf-8

# ##Landing Page: A/B testing

# This notebook analyzes a synthetic web traffic dataset for the website https://onpotatotime.com. 
# 
# Elements of the Dataset:
# 1) Video: watched duration, page load time, click information, created_at (per SQL 101)
# 2) Pricing: scroll information (how long user spent looking at each part of page), click information, created_at
# 
# The goal is to make a data-driven business decision between two landing pages under consideration:
# 
# 1) Landing Page A features a video, explaining how PotatoTime works.
# 2) Landing Page B features a new pricing section of the webpage, more clearly explaining how pricing works (completely fictional pricing).
# 
# 

# Hypotheses:
# 1) CTR increases if  landing page features a video and pricing section isn't needed.
# 2) Users find pricing useful and CTR increases.

# ###Preprocessing data in Pandas

# In[ ]:


#Import data


# In[91]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_pickle('views.pkl')
df


# In[11]:


df.mean()


# In[12]:


df.min()


# ###Cleaning data

# In[ ]:


#Delete duplicates


# In[14]:


df.drop_duplicates()


# In[ ]:


#Fill missing information values 'NaN'


# In[19]:


df.fillna(df.mean())


# In[112]:


# Check data on sanity check. Duration should less or equal 60 s


# In[21]:


df['video_watched_s'].max()


# In[22]:


df['video_watched_s_trunc']=df['video_watched_s'].clip(0,60)
df


# ###Data analysis

# In[ ]:


#Number of days spent


# In[24]:


df.index.max()-df.index.min()


# In[ ]:


#Views  of pages per day


# In[37]:


def events_per_day(df):
    datetimes=df.index
    days=datetimes.floor('d')
    events_per_day=days.value_counts()
    return events_per_day.sort_index()


# In[38]:


views_per_day= events_per_day(df)
views_per_day


# In[ ]:


#Clicks per day


# In[39]:


def get_click_events(df):
    selector=df['has_clicked']
    clicks=df[selector]
    return clicks


# In[40]:


clicks=get_click_events(df)
clicks_per_day=events_per_day(clicks)
clicks_per_day


# In[ ]:


#Correlation


# In[42]:


clicks_per_day.values


# In[ ]:


# Sanity check. Clicks must be less or equal to amount of views


# In[62]:


clicks_per_day.values<=views_per_day.values


# In[63]:


#Correlation between page load time and video duration
df.corr()


# In[ ]:


#When page load time increases, clicking decreases


# In[114]:


viewsA=df[df['webpage']=='A']
viewsB=df[df['webpage']=='B']
print(viewsA['has_clicked'].mean())
print(viewsB['has_clicked'].mean())


# In[ ]:


#We can see:
#1) Video correlates with clicking
#2) Pricing is not correlated with clicking
#Correlation suggests Webpage A (with video) is better.
#But CTR suggests Webpage B(pricing) is better.


# In[ ]:


###Data Visualization with Mathplotlib


# In[72]:


df.corr()


# In[ ]:


#Video correlation with clicking=0.329487 is higher than pricing correlates with clicking=0.104316


# In[73]:


viewsA.corr()


# In[ ]:


#If we look only on Webpage A correlation between video and clicks is much higher than we thought


# In[74]:


viewsB.corr()


# In[ ]:


#If we look only on Webpage B correlation between pricing and clicks is very low=0.000418


# In[ ]:


#Argument1: Videos views correlate with clicks and pricing page views don't correlate with clicks


# In[78]:


plt.title('Videos are Strongly Correlated with Clicking')
plt.bar(['Page Load', 'Video Watching', 'Reading Pricing'], [-0.39, 0.67,0.0004])
plt.ylabel("Correlation with clicking")


# In[82]:


def get_daily_stats(df):
    grouper=pd.Grouper(freq="D")
    groups=df.groupby(grouper)
    daily=groups.mean()
    return daily


# In[83]:


daily_viewsA=get_daily_stats(viewsA)
daily_viewsB=get_daily_stats(viewsB)
daily_viewsA


# In[ ]:


#Create line plot to see changes over time


# In[84]:


plt.title('Clicks Through Rates over Time')
plt.plot(daily_viewsA ['has_clicked'], label='A')
plt.plot(daily_viewsB ['has_clicked'], label='B')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Click Through Rate')


# In[85]:


plt.title('Webpage Page Load Times')
plt.plot(daily_viewsA ['page_load_ms'], label='A')
plt.plot(daily_viewsB ['page_load_ms'], label='B')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Page Load Time (ms)')


# In[ ]:


#We can see from the plot above that slow page loads damage CTR


# In[89]:


viewsA['page_load_ds']=viewsA["page_load_ms"]//20*20
page_load=viewsA.set_index('page_load_ds')
page_load=viewsA.groupby(['page_load_ds'])
page_load=page_load.mean()
page_load=page_load.sort_index()
page_load


# In[90]:


plt.plot(page_load['has_clicked'])


# In[ ]:


#We can see from above that CTR drops as page load gets slower


# In[ ]:


#Let's estimate how quickly CTR decreases as page load timeincrease


# In[95]:


m, b=np.polyfit(page_load.index, page_load['has_clicked'],1)

m*100


# In[ ]:


#Argument 2: Webpage A's CTR is artificially low


# In[98]:


plt.title('Every 100ms of Page Load Time Costs 7% CTR')
plt.plot(page_load['has_clicked'], label="CTR")
plt.plot(page_load.index, m*page_load.index+b, label='Fitted CTR')
plt.xlabel("Page Load Time(ms)")
plt.ylabel('Click Through Rate')
plt.legend()


# In[ ]:


#Let's see what would have happened if CTR hadn't been impacted by a slower page load


# In[100]:


viewsA[viewsA['page_load_ms']<550] ['has_clicked'].mean()


# In[ ]:


#CTR=70%


# In[101]:


viewsB[viewsB['page_load_ms']<550] ['has_clicked'].mean()


# In[115]:


#Argument3: Webpage A's CTR actually outperformn B's by 30%


# In[109]:


clicksA=get_click_events(viewsA)
clicksAdaily=events_per_day(clicksA)
clicksB=get_click_events(viewsB)
clicksBdaily=events_per_day(clicksB)
clicksAdaily


# In[111]:


plt.title('Webpage A Could Have Bossted CTR by 30%')
plt.plot(clicksAdaily, label='A')
plt.plot(clicksBdaily, label='B')
plt.plot(views_per_day*0.70*0.5, label='A (projected)')
plt.xlabel('Date')
plt.xlabel('Clicks')
plt.legend()


# In[ ]:


#Conclusion: we recommend Webpage A with informational video for 3 reasons:
#1) Watching videos is strongly correlated with clicking the sign up
#2) Reading the pricing section has zero bearing on whether or not users click to sign up 
#3) Although CTR rate dropped  precipitously at the same time when a page load time increased,but overall CYR cannot be trusted
#According to projections if the page load time had not shot up two month into the experiment, the Webpage A would have sustained 30% higher CTR than webpage B

