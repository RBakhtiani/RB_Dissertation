#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from threadpoolctl import threadpool_limits
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import openai
from math import sqrt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


# # DATA LOADING

# In[3]:


order_items = pd.read_csv('raw_data/order_items.csv')
orders = pd.read_csv('raw_data/orders.csv')
products = pd.read_csv('raw_data/products.csv')
users = pd.read_csv('raw_data/users.csv')
# targeting only UK users
uk_users = users[users['country'] == 'United Kingdom']


# # EDA & DATA CLEANING

# In[4]:


# basic info
# print(order_items.info())
# print(orders.info())
# print(products.info())
# print(uk_users.info())


# In[ ]:


# summary statistics for numeric columns
# print(order_items.describe())
# print(orders.describe())
# print(products.describe())
# print(uk_users.describe())


# In[ ]:


# summary statistics for object columns
# print(order_items.describe(include=['object']))
# print(orders.describe(include=['object']))
# print(products.describe(include=['object']))
# print(uk_users.describe(include=['object']))


# In[ ]:


# checking missing values in each table
# print(order_items.isnull().sum())
# print(orders.isnull().sum())
# print(products.isnull().sum())
# print(uk_users.isnull().sum())


# In[5]:


# droping missing values in products dataset
products.dropna(subset=['brand'], inplace=True)
products.dropna(subset=['name'], inplace=True) 


# In[ ]:


# checking for duplicates
# print(order_items.duplicated().sum())
# print(orders.duplicated().sum())
# print(products.duplicated().sum())
# print(uk_users.duplicated().sum())


# # PRE-PROCESSING & BDA

# In[6]:


# orders + order_item table
merged_orders = pd.merge(order_items, orders, on='order_id', how='inner')
merged_orders = merged_orders[['order_id', 'user_id_x', 'product_id','sale_price','num_of_item']].rename(columns={'user_id_x': 'user_id'})


# In[ ]:


# merged_orders.info()
# merged_orders.isnull().sum()


# In[7]:


# previous join + product
orders_product = pd.merge(merged_orders, products, left_on='product_id', right_on='id', how='inner')
orders_product = orders_product[['order_id','user_id','product_id','category','brand','retail_price','department','sale_price','num_of_item']]


# In[ ]:


# orders_product.info()
# orders_product.isnull().sum()


# In[8]:


# previous join + users
joined_data = pd.merge(orders_product, uk_users, left_on='user_id', right_on='id', how='inner')
joined_data = joined_data[['order_id','user_id','num_of_item','product_id','category','brand','retail_price','sale_price','department','first_name','last_name','email','age','gender','state','street_address','postal_code','city','country']]


# In[ ]:


# joined_data.info()
# joined_data.isna().sum()
# joined_data.head(5)


# In[9]:


# correlation matrix
numeric_columns = joined_data.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(15, 12))
sns.heatmap(joined_data[numeric_columns].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[10]:


order_item_users = pd.merge(order_items, uk_users, left_on='user_id', right_on='id', how='inner')

avg_order_value = order_item_users.groupby('user_id')['sale_price'].mean().reset_index()


# In[11]:


# average order value by user
avg_order_value['price_range'] = pd.cut(avg_order_value.sale_price, bins=[0, 9, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200], right=True, labels=['under 10', '10-20', '20-30', '30-40' , '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', 'over 100'])

plt.figure(figsize=(20, 12))
sns.countplot(x='price_range', data=avg_order_value)
plt.title('Spent per user')
plt.show()


# In[12]:


# distribution of age_group
joined_data['age_group'] = pd.cut(joined_data.age, bins=[0, 19, 30, 40, 50, 60, 200], right=True, labels=['under 20', '20-30', '30-40' , '40-50', '50-60', 'over 60'])

plt.figure(figsize=(20, 12))
sns.countplot(x='age_group', data=joined_data)
plt.title('Age Distribution of UK Users')
plt.show()


# In[13]:


# count of products per category
plt.figure(figsize=(20, 12))
sns.countplot(x='category', data=joined_data, order=joined_data['category'].value_counts().index[:10])
plt.title('Top 10 category by Product Count')
plt.show()


# In[14]:


# count of products per brand
plt.figure(figsize=(15, 9))
sns.countplot(x='brand', data=joined_data, order=joined_data['brand'].value_counts().index[:10])
plt.title('Top 10 Brands by Product Count')
plt.show()


# In[15]:


# distribution of price_range
joined_data['price_range'] = pd.cut(joined_data.age, bins=[0, 19, 30, 40, 50, 60, 200], right=True, labels=['under 20', '20-30', '30-40' , '40-50', '50-60', 'over 60'])

plt.figure(figsize=(20, 12))
sns.countplot(x='price_range', data=joined_data)
plt.title('Price Distribution of Products')
plt.show()


# In[16]:


orders_users = pd.merge(orders, uk_users, left_on='user_id', right_on='id', how='inner')
distinct_orders_per_user = joined_data.groupby('user_id')['order_id'].nunique().reset_index()


# In[17]:


# orders per user
plt.figure(figsize=(15, 9))
sns.countplot(x='order_id', data=distinct_orders_per_user, order=distinct_orders_per_user['order_id'].value_counts().index[:10])
plt.xlabel("Number of orders")
plt.ylabel("Count")
plt.title('orders per user')
plt.show()


# In[18]:


# num_of_item per order
plt.figure(figsize=(15, 9))
sns.countplot(x='num_of_item', data=orders_users, order=orders_users['num_of_item'].value_counts().index[:10])
plt.xlabel("Number of items")
plt.ylabel("Count")
plt.title('num_of_item per order')
plt.show()


# # CUSTOMER SEGMENTATION

# In[19]:


total_order_value_per_order = joined_data.groupby('order_id')['sale_price'].sum().reset_index()
total_order_value_per_order.rename(columns={'sale_price': 'total_order_value'}, inplace=True)

total_spent_per_customer = joined_data.groupby('user_id')['sale_price'].sum().reset_index()
total_spent_per_customer.rename(columns={'sale_price': 'total_user_spent'}, inplace=True)


# In[20]:


joined_data = joined_data.merge(total_order_value_per_order, on='order_id', how='left')
joined_data = joined_data.merge(total_spent_per_customer, on='user_id', how='left')


# In[21]:


# features for segmentation
segmentation_features = joined_data.groupby('user_id').agg({
    'total_user_spent': 'mean',
    'age': 'mean',
    'order_id': 'nunique'  # frequency of orders
}).reset_index()

segmentation_features_order = joined_data.groupby(['user_id','order_id']).agg({
    'num_of_item': 'mean',
    'total_order_value': 'mean'
}).reset_index()

segmentation_features_user = segmentation_features_order.groupby('user_id').agg({
    'num_of_item': 'sum'
}).reset_index()

# renameing the columns for clarity
segmentation_features.rename(columns={'order_id': 'order_frequency'}, inplace=True)

segmentation_features = segmentation_features.merge(segmentation_features_user, on='user_id', how='left')


# In[ ]:


# segmentation_features[segmentation_features['user_id'] == 49796]
# segmentation_features.shape


# In[22]:


segmentation_features_km = segmentation_features.copy()
segmentation_features_sc = segmentation_features.copy()
segmentation_features_hc = segmentation_features.copy()


# In[23]:


scaler = StandardScaler()
scaled_features_km = scaler.fit_transform(segmentation_features_km[['total_user_spent', 'age', 'order_frequency', 'num_of_item']])
scaled_features_sc = scaler.fit_transform(segmentation_features_sc[['total_user_spent', 'age', 'order_frequency', 'num_of_item']])
scaled_features_hc = scaler.fit_transform(segmentation_features_hc[['total_user_spent', 'age', 'order_frequency', 'num_of_item']])


# ## KMeans Clustering

# In[24]:


# determining the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features_km)
    wcss.append(kmeans.inertia_)


# In[25]:


# plotting the Elbow Method
import matplotlib.pyplot as plt
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[26]:


for k in range(2, 6):  # testing k values from 2 to 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_features_km)
    silhouette = silhouette_score(scaled_features_km, labels)
    print(f"Silhouette Score for k={k}: {silhouette}")


# In[27]:


kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_features_km)
segmentation_features_km['Cluster'] = kmeans_labels


# In[28]:


cluster_analysis = segmentation_features_km.groupby('Cluster').mean()
print("Cluster Analysis")
print(cluster_analysis)


# In[29]:


cluster_sizes = segmentation_features_km['Cluster'].value_counts()
print("Cluster Sizes:")
print(cluster_sizes)


# ## Spectral Clustering

# In[30]:


for k in range(2, 6):  # testing k values from 2 to 5
    spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
    labels = spectral.fit_predict(scaled_features_sc)
    silhouette = silhouette_score(scaled_features_sc, labels)
    print(f"Silhouette Score for k={k}: {silhouette}")


# In[31]:


spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
spectral_labels = spectral.fit_predict(scaled_features_sc)
segmentation_features_sc['Cluster_Spectral'] = spectral_labels


# In[32]:


cluster_analysis = segmentation_features_sc.groupby('Cluster_Spectral').mean()
print("Cluster Analysis:\n")
print(cluster_analysis)


# In[33]:


cluster_sizes = segmentation_features_sc['Cluster_Spectral'].value_counts()
print("Cluster Sizes:")
print(cluster_sizes)


# ## Hierarchical Clustering

# In[34]:


for k in range(2, 6):  # testing k values from 2 to 5
    linkage_matrix = linkage(scaled_features_hc, method='ward')
    hierarchical_labels = fcluster(linkage_matrix, t=k, criterion='maxclust')
    silhouette = silhouette_score(scaled_features_hc, hierarchical_labels)
    print(f"Silhouette Score for k={k}: {silhouette}")


# In[35]:


linkage_matrix = linkage(scaled_features_hc, method='ward')
hierarchical_labels = fcluster(linkage_matrix, t=2, criterion='maxclust')
segmentation_features_hc['Cluster_Hierarchical'] = hierarchical_labels


# In[36]:


cluster_analysis = segmentation_features_hc.groupby('Cluster_Hierarchical').mean()
print("Cluster Analysis:")
print(cluster_analysis)


# In[37]:


cluster_sizes = segmentation_features_hc['Cluster_Hierarchical'].value_counts()
print("Cluster Sizes:")
print(cluster_sizes)


# In[38]:


clustered_data = pd.merge(joined_data, segmentation_features_hc, on='user_id', how='inner')
clustered_data = clustered_data[['order_id','user_id','num_of_item_x','product_id','category','brand','retail_price','sale_price','department','first_name','last_name','email','age_x','gender','state','street_address','postal_code','city','country','total_order_value','total_user_spent_x','order_frequency','Cluster_Hierarchical']].rename(columns={'num_of_item_x': 'num_of_item','age_x':'age','total_user_spent_x':'total_user_spent','Cluster_Hierarchical':'cluster'})


# In[39]:


# visualizing Product Category Distribution per Cluster
category_distribution = clustered_data.groupby(['cluster', 'category']).size().reset_index(name='count')

plt.figure(figsize=(20, 10))
sns.barplot(data=category_distribution, x='category', y='count', hue='cluster', dodge=True)
plt.title('Product Category Distribution by Cluster')
plt.xticks(rotation=90)
plt.show()


# In[40]:


# analyzing Brands per Cluster
brand_distribution = clustered_data.groupby(['cluster', 'brand']).size().reset_index(name='count')
top_brands = brand_distribution.groupby('brand')['count'].sum().nlargest(10).index  # Top 10 brands

brand_distribution = brand_distribution[brand_distribution['brand'].isin(top_brands)]

plt.figure(figsize=(12, 6))
sns.barplot(data=brand_distribution, x='brand', y='count', hue='cluster', dodge=True)
plt.title('Top Brand Distribution by Cluster')
plt.xticks(rotation=45)
plt.show()


# # RECOMMENDATION ENGINE

# In[41]:


def create_train_test_data(clustered_data, test_size=0.2):
    train_data, test_data = train_test_split(clustered_data, test_size=test_size, random_state=42)

    test_data = test_data[['user_id', 'product_id']].drop_duplicates()

    train_data = train_data.reset_index(drop=True)

    return train_data, test_data


# In[42]:


clustered_data['gender_encoded'] = LabelEncoder().fit_transform(clustered_data['gender'])

# extracting user data
user_data = clustered_data.groupby('user_id').agg({
        'gender_encoded':'first',
        'total_user_spent':'mean',
        'order_frequency':'mean',
        'product_id':lambda x: list(x), # List of purchased products
        'cluster':'mean'
    }).reset_index()
user_data.rename(columns={'product_id':'purchased_products'}, inplace=True)

user_data['price_range'] = user_data['total_user_spent'] / user_data['order_frequency']
    
# extracting product data
product_data = clustered_data[['product_id','gender_encoded','sale_price']].drop_duplicates()
product_data.rename(columns={'sale_price':'price_range'}, inplace=True)


# In[43]:


def get_recommendations(user_data, product_data, n=3):
    # standardising numerical features
    scaler = StandardScaler()
    user_features = scaler.fit_transform(user_data[['gender_encoded', 'price_range']])
    product_features = scaler.fit_transform(product_data[['gender_encoded', 'price_range']])

    recommendations_list = []
    user_data = user_data.reset_index(drop=True)

    for user_id in user_data['user_id'].unique():
        user_index = user_data[user_data['user_id'] == user_id].index[0]
        user_vector = user_features[user_index].reshape(1, -1)

        similarity_scores = cosine_similarity(user_vector, product_features).flatten()
        product_data['similarity_score'] = similarity_scores

        # including all products (no exclusion)
        product_data_sorted = product_data.sort_values(by='similarity_score', ascending=False)
        top_recommendations = product_data_sorted.head(n)

        for _, row in top_recommendations.iterrows():
            recommendations_list.append({
                'user_id': user_id,
                'product_id': row['product_id'],
                'similarity_score': row['similarity_score']
            })

    return pd.DataFrame(recommendations_list)


# In[48]:


# top 5 recommendations
recommendations_df = get_recommendations(user_data,product_data,5)


# In[45]:


def evaluate_recommendations(recommendations_df, test_data, k=10):
    recommendations_df = recommendations_df.groupby('user_id').apply(lambda x: x.nlargest(k, 'similarity_score'))

    actual_purchases = test_data.groupby('user_id')['product_id'].apply(set).to_dict()

    precisions, recalls, accuracies = [], [], []

    for user_id in recommendations_df['user_id'].unique():
        recommended_products = set(recommendations_df[recommendations_df['user_id'] == user_id]['product_id'].tolist())

        actual_products = actual_purchases.get(user_id, set())

        true_positives = len(recommended_products & actual_products)
        precision = true_positives / k if k > 0 else 0
        recall = true_positives / len(actual_products) if len(actual_products) > 0 else 0
        accuracy = true_positives / len(recommended_products) if len(recommended_products) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(accuracy)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    accuracies = np.array(accuracies)

    avg_precision = precisions.mean()
    avg_recall = recalls.mean()
    avg_accuracy = accuracies.mean()
    avg_f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    return avg_precision, avg_recall, avg_f1, avg_accuracy


# In[46]:


# spliting the data into training and testing sets
train_data, test_data = create_train_test_data(clustered_data)


# In[47]:


for k in [3, 5, 7, 10]:
    precision_k, recall_k, f1_k, accuracy_k = evaluate_recommendations(recommendations_df, test_data, k=k)
    print(f"K={k}: Precision={precision_k:.4f}, Recall={recall_k:.4f}, F1={f1_k:.4f}, Accuracy={accuracy_k:.4f}")


# In[49]:


num_users = recommendations_df['user_id'].nunique()
num_products_recommended = recommendations_df['product_id'].nunique()
print(f"Users with recommendations: {num_users}")
print(f"Unique products recommended: {num_products_recommended}")


# In[50]:


recommended_in_test = recommendations_df['product_id'].isin(test_data['product_id']).sum()
total_recommended = recommendations_df['product_id'].nunique()
print(f"Overlap with test data: {recommended_in_test}/{total_recommended} products")


# In[51]:


joined_data[joined_data['user_id'] == 49796]


# In[52]:


recommendations_df[recommendations_df['user_id'] == 49796]


# # GEN AI

# In[ ]:


# merging product and user details with recommendations


# In[61]:


product_recommendations = pd.merge(recommendations_df, products, left_on='product_id', right_on='id', how='inner')


# In[62]:


product_recommendations = product_recommendations[['user_id','product_id','similarity_score','category','name','brand','retail_price','department']]


# In[63]:


user_recommendations = pd.merge(product_recommendations, uk_users, left_on='user_id', right_on='id', how='inner')


# In[65]:


user_recommendations = user_recommendations[['user_id','product_id','similarity_score','category','name','brand','retail_price','department','first_name']]
user_recommendations[user_recommendations['user_id'] == 49796]


# In[70]:


# grouping recommendations by user_id
grouped_recommendations = user_recommendations.groupby('user_id').apply(
    lambda x: x[['name', 'category']].to_dict(orient='records')
).to_dict()


# In[67]:


grouped_recommendations


# In[71]:


# Set up OpenAI API Key
openai.api_key = my_key

def generate_campaign_message(user_id, recommendations):
    
    product_list = ", ".join([f"{rec['name']} ({rec['category']})" for rec in recommendations])
    prompt = f"""
    Create a personalized marketing email for User {user_id}. 
    The email should recommend the following products: {product_list}.
    Make it engaging, friendly, and persuasive, focusing on why the user should buy these products.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a marketing assistant who specializes in creating campaigns."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7,
    )

    return response['choices'][0]['message']['content'].strip()


# In[72]:


# generating campaigns for each user
campaign_messages = {}
for user_id, recommendations in grouped_recommendations.items():
    campaign_messages[user_id] = generate_campaign_message(user_id, recommendations)


# In[ ]:


# printing campaign messages
for user_id, message in campaign_messages.items():
    print(f"Campaign for User {user_id}:\n{message}\n")

