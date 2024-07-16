#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Importing the libraries
import pandas as pd
import numpy as np
# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)  # None means unlimited
pd.set_option('display.max_columns', None)


# In[11]:


# read the data into a dataframe
df = pd.read_csv('random_sample_data.csv')
# creating a copy of df to avoid repeat reading from the csv
df1 = df.copy()
df1.head()


# In[12]:


# Convert 'dob' column to datetime format
df1['dob'] = pd.to_datetime(df1['dob'])

# Display the data type of the 'dob' column
print(df1['dob'].dtype)

# Create new features
df1['age'] = pd.Timestamp.now().year - df1['dob'].dt.year
df1['year_of_birth'] = df1['dob'].dt.year
df1['month_of_birth'] = df1['dob'].dt.month
df1['day_of_birth'] = df1['dob'].dt.day

df1.drop(columns=['dob'],axis = 1, inplace = True)


# In[13]:


# Function to convert features to relevant data types
def convert_data_types(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            # Convert object type columns to category if the number of unique values is low
            if len(df[column].unique()) / len(df[column]) < 0.05:
                df[column] = df[column].astype('category')
        elif df[column].dtype == 'int64':
            # Convert int64 type columns to int16
            if df[column].min() >= -32768 and df[column].max() <= 32767:
                df[column] = df[column].astype('int16')
        elif df[column].dtype == 'float64':
            # Convert float64 type columns to smaller float types if applicable
            if df[column].isnull().sum() == 0:
                if df[column].min() >= 0:
                    if df[column].max() < 3.4e38:
                        df[column] = df[column].astype('float32')
            else:
                if df[column].max() < 1.7e308 and df[column].min() > -1.7e308:
                    df[column] = df[column].astype('float64')
        elif df[column].dtype == 'object' and column.lower().endswith('date'):
            # Convert date type columns to datetime datatype
            df[column] = pd.to_datetime(df[column], errors='coerce')

# Apply conversion function
convert_data_types(df1)

# Display the data types of the DataFrame after conversion
df1.info()


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[ ]:





# In[15]:


# Select relevant features for content-based filtering
features = ['acc_balance', 'asset_allocation', 'avg_monthly_balance', 'avg_monthly_spending', 'budget_practices', 'business_ownership', 'charitable_contributions', 'charitable_giving_history', 'country', 'cc_usage', 'credit_score', 'customer_satisfaction', 'debt_payoff_goal', 'debt_to_income', 'dependents', 'digital_literacy', 'disability', 'education', 'emergency_fund_goal', 'industry', 'employment_status', 'bank_accounts', 'existing_loans', 'retirement_plan', 'expected_retirement_age', 'family_income', 'family_size', 'family_structure', 'financial_concerns', 'financial_goals', 'financial_literacy', 'financial_planning', 'financial_responsibilities', 'location', 'health_goals', 'health_insurance', 'hobbies', 'home_ownership', 'housing_expenses', 'income', 'industry_sector', 'insurance_coverage', 'interest_cc', 'interest_investment', 'investment_account', 'investment_knowledge', 'investment_portfolio', 'investment_risk', 'life_insurance', 'loan_history', 'long_term_savings', 'major_life_events', 'major_purchase', 'marital_status', 'mid_term_savings', 'military_service', 'monthly_expenses', 'mortgage_status', 'num_dependents', 'occupation', 'online_shopping', 'other_assets', 'pet_ownership', 'banking_channel', 'communication_channels', 'investment_duration', 'investment_types', 'prev_banking_exp', 'prev_product_use', 'primary_language', 'product_inquiries', 'online_banking_propensity', 'social_media_info', 'real_estate', 'recreational_activities', 'retirement_age_goal', 'retirement_planning_status', 'retirement_plans', 'risk_appetite', 'risk_tolerance', 'risk_propensity', 'savings_amount', 'savings_rate', 'short_term_savings', 'social_media_presence', 'state_province', 'stock_market_exp', 'subscription_services', 'tax_filing_status', 'tech_adoption', 'transaction_history', 'travel_frequency', 'urban_rural', 'vehicle_ownership', 'years_employed', 'years_experience', 'product', 'age', 'year_of_birth', 'month_of_birth', 'day_of_birth']

# Combine the selected features into a single string for each item
df1['combined_features'] = df1.apply(lambda row: ' '.join(row[features].astype(str)), axis=1)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the TF-IDF vectorizer on the combined features
tfidf_matrix = tfidf_vectorizer.fit_transform(df1['combined_features'])

# Compute similarity scores using cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#print("the cosine similarity score is")
#print(cosine_sim)

# Defining get_recommendations functions with product_id as inout and cosine similarity as similrity measure
def get_recommendations(product_id, cosine_sim=cosine_sim, data=df1):
    # Convert the product_id to lowercase
    product_id = product_id.lower()

    # Get the lowercase version of the product names
    lowercase_product = data['product'].str.lower()

    # Get the indices of the products that contain the product_id
    indices = lowercase_product[lowercase_product.str.contains(product_id)].index.tolist()

    # If no match is found, return an empty list
    if not indices:
        return "The product is not in the cart, could you please retry"

    # Get the pairwise similarity scores of all products with those products
    sim_scores = [(i, cosine_sim[idx][i]) for idx in indices for i in range(len(cosine_sim[idx]))]

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Initialize an empty list to store the recommended products
    recommend_products = []

    # Iterate over the product indices
    for idx in product_indices:
        # Get the product name
        product = data['product'].iloc[idx]

        # If the product is not already in the recommended products list, add it
        if product not in recommend_products:
            recommend_products.append(product)

        # If we have found 10 unique products, stop looking
        if len(recommend_products) == 10:
            break

    # Return the recommended products
    return recommend_products


# In[16]:


df1['product'].unique().tolist()


# In[26]:


# Example usage: Get recommendations for a product with ID 'xyz'
# recommended_products = get_recommendations('ETFs')
# if type(recommended_products)== list:
#     print(*recommended_products,sep='\n')
# else:
#     print(recommended_products)


import streamlit as st
st.title("Welcome to Standard Chartered Bank")
#st.image(logo_url, width=100)
with st.sidebar:    
    #Final_product = st.selectbox('Select a product:', product_list)
    Final_product=st.text_input('Enter your query:')
recommended_products = list(set(get_recommendations(Final_product)))

#recommended_products.index += 1 
#recommended_products.rename('Customer recommended products', inplace=True)
st.write("Your recommended product for your search:", Final_product)
st.table(recommended_products) 





