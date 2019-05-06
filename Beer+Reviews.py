
# coding: utf-8

# ##### Candidate Name: Dr. Gaurav Budhirja
# ##### Email ID: gb.linked@gmail.com
# ##### Phone: (919)-949-8170
# 
# ### Step 1:  Import Libraries

# In[1]:

get_ipython().magic('matplotlib inline')

import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')
sns.set(style="ticks", color_codes=True)


# In[2]:

os.getcwd() #To Check current workign directory


# In[3]:

os.listdir() # List all files present in the working directory.


# ### Step 2: Load csv file as  dataframe using pandas library in Python

# In[4]:

filename = "beer_reviews.csv"

try: 
    beerdf = pd.read_csv(filename, delimiter = ',', encoding='utf-8') #Load csv file as dataframe 
except: 
    print('Data Loading Failed')
else:
    print('Data Loaded Succesfully')


# ### Step 3: Inspect Data and Clean it up (if needed)

# In[5]:

beerdf.shape  #Check the number of rows and columns


# In[6]:

beerdf.columns


# In[7]:

beerdf.rename(columns = {'beer_beerid':'beer_id'}, inplace = True) # Rename Column 'beer_beerid' as'beer_id'


# In[8]:

beerdf.head()


# In[9]:

beerdf.info()


# In[10]:

beerdf.isnull().sum()


# There is missing data (NaN/ None) for following columns: 'brewery_name', 'review_profilename' and 'beer_abv'. 
# Missing 'review_profilename' will be filled up using label 'unknown' or 'missing'
# Missing values 'beer_abv' (type: float) will be handled(or dropped) later as per requirement.

# In[11]:

beerdf[["review_profilename"]] = beerdf[["review_profilename"]].fillna('unknown')


# In[12]:

beerdf[["brewery_name"]] = beerdf[["brewery_name"]].fillna('missing')


# In[13]:

beerdf.info()


# In[14]:

beerdf_obj = beerdf.select_dtypes(['object']) # Subset columns with datatype objects for further cleaning


# In[15]:

# beerdf.brewery_name.replace({r'[^\x00-\x7F]+':''}, regex = True, inplace = True) # To get rid of non-English Characters.


# In[16]:

# beerdf.beer_name.replace({r'[^\x00-\x7F]+':''}, regex = True, inplace = True) # To get rid of Non-English Characters (if needed).


# In[17]:

beerdf_obj = beerdf.select_dtypes(['object']) # Subset datatype objects for further processing.


# In[18]:

beerdf[beerdf_obj.columns] = beerdf_obj.apply(lambda x: x.str.strip()) # To get rid of trailing spaces.


# In[19]:

beerdf = beerdf.drop_duplicates() # remove any duplicate rows if present


# In[20]:

beerdf.shape


# ### Step 4:  Lets check if breweries have been assigned unique ID or not.

# In[21]:

subset1 = beerdf[["brewery_id", "brewery_name"]].drop_duplicates()


# In[22]:

subset1.head() 


# In[23]:

subset1.shape


# In[24]:

subset1.groupby(['brewery_name']).brewery_id.nunique().sort_values(ascending=False).reset_index(name='unique brewery_id count').head(10)


# ##### As shown in the above table, some of the breweries have been assigned more than one unique ID. Therefore, for analysis purpose we may not use brewery ID. 

# ### Step 5:  Lets check if beers from each breweries have been assigned unique beer_ID or not.

# In[25]:

subset2 = beerdf[["brewery_name", "beer_id", "beer_name"]].drop_duplicates()


# In[26]:

subset2.head() 


# In[27]:

subset2.shape


# In[28]:

subset2.groupby(['brewery_name','beer_name']).beer_id.nunique().sort_values(ascending=False).reset_index(name='unique beer_id count').head(10)


# ##### Once again, as shown in the above table, some beers from the same brewery have been assigned multiple beer_IDs. Therefore, for analysis purpose we may have to drop beer_IDs as well. 

# In[29]:

beerdf1 = beerdf.drop(['brewery_id', 'beer_id'], axis = 1).drop_duplicates() # Drop columns = 'brewery_id', 'beer_id' and remove duplicates


# In[30]:

beerdf1.head()


# In[31]:

beerdf1.shape


# ### Q1: Which brewery produces the strongest beers by ABV%?

# In[32]:

Strong_ABV = beerdf1[["brewery_name", "beer_name", "beer_abv"]].nlargest(10,"beer_abv") # Extract top 3 rows w.r.t beer_ABV values


# In[33]:

Strong_ABV


# In[34]:

beerdf1[["beer_abv"]].hist()
plt.xlabel('beer_abv %')
plt.ylabel('Frequency')
plt.title('NUmbers of Beers w.r.t to beer_abv %')
plt.show()


# In[35]:

print(str(Strong_ABV.iloc[0,0]) + ' produces the strongest beer with ' +str(Strong_ABV.iloc[0,2])  +'% ABV.')


# 

# ### Step 6: Perform Exploratory Data Analysis

# In[36]:

beerdf1.describe().round(2) # To take a look at Statistical Summary


# Based on summary statistics, all reviews lie between 1 and 5. 

# In[37]:

beerdf1['beer_name'].value_counts().head() 


# In[38]:

beerdf1['beer_name'].value_counts().tail()


# ## No of review per beer varies from 1 to 3290

# In[39]:

beerdf1['beer_name'].value_counts(ascending = False)[1:20].sort_values(ascending=True).plot(kind='barh')
plt.xlabel('No of Reviews')
plt.ylabel('Beer Name')
plt.title('Most reviewed Beers (Top 20)')
plt.show()


# In[40]:

hist = beerdf1[["beer_abv", "review_aroma", "review_appearance", "review_palate", "review_taste", "review_overall"]].hist()


# In[41]:

beerdf2 = beerdf1[["review_aroma", "review_appearance", "review_palate", "review_taste", "review_overall"]]


# In[42]:

pairplot = sns.pairplot(beerdf2)


# ### Q3:  If you had to pick 3 beers to recommend using only this data, which would you pick?

# ##### Let's start by  looking at the  list of beers mean overall raiting of each bear (in the descending order). 

# In[43]:

beerdf1.groupby('beer_name')['review_overall'].mean().sort_values(ascending=False).head(3) # Top five rows only. 


# ##### Problem with the above stats is that even beer with one review (5 star rating) can make to this list. Therefore, it can be misleading. Let's now plot the total number of reviews  for each beer.

# In[44]:

beerdf1['beer_name'].value_counts().head()


# ##### There is a good probability that good beers normally receive more number of reviews as well as higher ratings. Now we know that both the average review per beer and the number of ratings per beer are important attributes. Let's create a new dataframe that contains both of these attributes.

# In[45]:

mean_overall_review_df = pd.DataFrame(beerdf1.groupby('beer_name')['review_overall'].mean())


# In[46]:

mean_overall_review_df['review_counts'] = pd.DataFrame(beerdf1.groupby('beer_name')['review_overall'].count())


# In[47]:

mean_overall_review_df.rename(columns = {'review_overall':'mean_review_overall'}, inplace = True)


# In[48]:

mean_overall_review_df.head()


# ##### Now , above table displays the beer name along with the average rating and number of ratings for each beer. Now Let's plot a histogram for the number of ratings represented by the "rreview_counts" column in the above dataframe.

# In[49]:

mean_overall_review_df['review_counts'].hist(bins=10)
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')
plt.title('Number of Reviews/Beer')
plt.show()


# ##### From the histogram shown above, you can see that majority of the beers have received less than 300 (approx.) reviews. While the number of beers having more than 650 reviews is very low. Now, let's plot a histogram for average ratings.

# In[50]:

mean_overall_review_df['mean_review_overall'].hist(bins=50)
plt.xlabel('Mean_Overall_Review')
plt.ylabel('Frequency')
plt.title('Distribution of Average Raiting')
plt.show()


# ##### In the plot shown above, the integer values have taller bars than the floating values since most of the users assign rating as integer value i.e. 1, 2, 3, 4 or 5. Furthermore, it is evident that the data has a weak normal distribution with the mean of around 3.5. There are a few outliers in the data. Now, lets plot Mean_Overall_Review against the number of reviews

# In[51]:

sns.jointplot(x='mean_review_overall', y='review_counts', data=mean_overall_review_df, alpha=0.4)


# ##### The plot shows that, beer with higher overall reviews  have more number of reviews, compared with beers that have lower mean overall_review. Now lets build a  recommendation system using matrix factorization. For that purpose,  pick a beer that is highly reviewd and mean rated value above 4.5

# In[52]:

mean_overall_review_df


# In[53]:

mean_overall_review_df[(mean_overall_review_df['review_counts']>300) & (mean_overall_review_df['mean_review_overall'] >=4.5)].sort_values('review_counts', ascending=False).head()


# ### Lets Pick Beer 'Pliny The Elder' ( Highly reviewed: 2527 reviews and mean overall rating of approx 4.6)

# In[54]:

## Take sum of all reviews and subset beerdf1 to select only revelant columns


# In[55]:

beerdf1['sum_reviews'] = beerdf1.review_overall + beerdf1.review_aroma + beerdf1.review_appearance + beerdf1.review_palate + beerdf1.review_taste


# In[56]:

beerdf2 = beerdf2 = beerdf1[["review_profilename", "beer_name", "sum_reviews"]]


# In[57]:

beerdf2.head()


# #### To create the matrix of beers and corresponding aggregated reviews, execute the following script:

# In[58]:

beerdf2_pivot = beerdf2.pivot_table(index=['review_profilename'], columns=['beer_name'], values='sum_reviews')


# In[59]:

beerdf2_pivot.head()


# In[60]:

Pliny_Elder_reviews = beerdf2_pivot['Pliny The Elder']


# In[61]:

Pliny_Elder_reviews.head()


# In[62]:

## Now let's retrieve all the beers that are similar to "Pliny The Elder"


# In[63]:

beer_like_Pliny_Elder = beerdf2_pivot.corrwith(Pliny_Elder_reviews)

corr_Pliny_Elder = pd.DataFrame(beer_like_Pliny_Elder, columns=['Correlation'])  
corr_Pliny_Elder.dropna(inplace=True)  
corr_Pliny_Elder.head()


# In[64]:

## Let's sort the beers in descending order of correlation to see highly correlated beers at the top.


# In[65]:

corr_Pliny_Elder.sort_values('Correlation', ascending=False).head(10)


# In[66]:

## Retrieve only those correlated beers that have at least more than 300 reviews. To do so, will add the review_counts column from the rating_mean_count dataframe to our mean_overall_review_df dataframe. 


# In[67]:

corr_Pliny_Elder = corr_Pliny_Elder.join(mean_overall_review_df['review_counts'])  
corr_Pliny_Elder.head() 


# In[68]:

corr_Pliny_Elder[corr_Pliny_Elder ['review_counts']>300].sort_values('Correlation', ascending=False).head(3)


#  ### Answer 2: Recommended Beers: Pliny The Elder, Pliny The Younger &  Blind Pig IPA

# In[69]:

mean_overall_review_df.loc['Pliny The Elder'] 


# In[70]:

mean_overall_review_df.loc['Pliny The Younger']


# In[71]:

mean_overall_review_df.loc['Blind Pig IPA']


# ### Q3. Which of the factors (aroma, taste, appearance, palette) are most important in determining the overall quality of a beer?

# In[72]:

beerdf3 = beerdf1[["review_overall", "review_aroma", "review_appearance", "review_palate", "review_taste"]]


# In[73]:

beerdf3.corr()


# ##### Taste is the most important factor  as it is highly correlated with overall review of the beer. ( we are assuming that Quality is directly proportion to Overall_review)

# ### Q4: Lastly, if I typically enjoy a beer due to its aroma and appearance, which beer style should I try?

# In[74]:

# Lets create subset of dataframe


# In[75]:

beer_styledf = beerdf1[['review_aroma', 'review_appearance', 'beer_style']]


# In[76]:

beer_styledf.head()


# In[77]:

##Take Average of Aroma Review and Appearance review


# In[78]:

beer_styledf['average_aroma_appearance'] = (beer_styledf.review_aroma + beer_styledf.review_appearance)/2


# In[79]:

beer_styledf = beer_styledf[['beer_style', 'review_aroma', 'review_appearance', 'average_aroma_appearance']]


# In[80]:

beer_styledf.head()


# In[81]:

#### Take mean vakues of rating for each bear and sort from high to low.


# In[82]:

beer_styledf_agg = beer_styledf.groupby('beer_style')[['review_aroma', 'review_appearance', 'average_aroma_appearance' ]].mean().sort_values(by='average_aroma_appearance', ascending=False)[:10]


# In[83]:

beer_styledf_agg


# ##### Table shown above clearly indicates American Double / Imperial Stout with highest overall rating. However, scatterplot and pair plot can be  analyzed prior to making solution. 

# In[84]:

g = sns.jointplot(x='review_aroma', y='review_appearance', data=beer_styledf_agg, color="r", alpha=0.4, marker = 's')


# In[85]:

sns.pairplot(beer_styledf_agg)


# In[86]:

beer_styledf_agg.corr()


# ### Answer 4: Reccomended beer_style based on aroma and appearance is:  American Double / Imperial Stout	
