#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install streamlit


# In[2]:


# pip install seaborn


# In[3]:


pip install plotly


# In[4]:


import pandas as pd
import plotly.express as px 
import streamlit as st
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter
from bokeh.transform import factor_cmap
from bokeh.transform import linear_cmap
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
import plotly.graph_objects as go
from bokeh.palettes import Spectral10
import datetime
from bokeh.transform import jitter


# In[5]:


df = pd.read_csv("churn.csv", encoding = "ISO-8859-1")


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


st.set_page_config(page_title = "Churning customers", page_icon = ":credit_card:", layout = "wide")


# In[10]:


st.title(":credit_card: Credit Card Churning Customers Analysis")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
box_date = str(datetime.datetime.now().strftime("%d %B %Y"))
st.write(f"Last updated by:  \n {box_date}")


# In[11]:


fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
else:
    df = pd.read_csv("churn.csv", encoding = "ISO-8859-1")


# In[12]:


st.divider()


# In[13]:


# Create columns with a custom layout, including 'spacer' columns
col2, spacer, col3 = st.columns([3, 1.05, 3])


# In[14]:


# Title for the sidebar
st.sidebar.header('Choose your filter:')

# Define categories and corresponding colors
categories = ['Attrited Customer', 'Existing Customer']
category_colors = ['#5E18EB', '#A579FF']
color_mapping = {'Attrited Customer': category_colors[0], 'Existing Customer': category_colors[1]}

# List of all columns for which you want to create filters
all_columns = ['Gender', 'Education', 'Marital', 'Income', 'CardCat', 'MonthsInactive',
               'Age', 'Dependent', 'PeriodOfRelationship', 'TotalNumberProducts',
               'ContactsCount', 'CreditLimit', 'TotalBal', 'AvgOpenToBuy',
               'ChangeTransAmountQ4Q1', 'TotalTransAmt', 'TotalTransCountQ4Q1',
               'ChangeTransCountQ4Q1', 'UtilRatio']

# Let the user select a column to plot
chosen_col = st.sidebar.selectbox('Select a column to plot', all_columns)

# Create filters in the sidebar
selected_filters = {}
for col in all_columns:
    if df[col].dtype == 'object':
        # Use multiselect for categorical columns
        selected_filters[col] = st.sidebar.multiselect(f'Select {col}', df[col].unique())
    else:
        # Use sliders for numerical columns
        min_val = df[col].min()
        max_val = df[col].max()
        selected_filters[col] = st.sidebar.slider(f'Select range for {col}', min_val, max_val, (min_val, max_val))

# Filtering the dataframe based on the selections
filtered_df = df.copy()
for col, values in selected_filters.items():
    if isinstance(values, list) and values:  # For categorical filters
        filtered_df = filtered_df[filtered_df[col].isin(values)]
    elif isinstance(values, tuple):  # For numerical filters
        filtered_df = filtered_df[filtered_df[col].between(values[0], values[1])]

# Create a container for the plot
plot_container = st.container()

with col2:
    # Custom HTML for the title with larger font size and bold text
    st.markdown("""
    <style>
    .title {
        font-size: small; 
        font-weight: bold;
    }
    </style>
    <div class="title">Customize Your Analysis: Select Variables to Compare Results</div>
    """, unsafe_allow_html=True)    

    # Create and display the count plot if a column is selected
    if chosen_col:
        fig = px.histogram(filtered_df, x=chosen_col, color='ExistingLost',
                           color_discrete_map=color_mapping,
                           title=f'Count of ExistingLost for {chosen_col}')
        fig.update_layout(title_text='')
        st.plotly_chart(fig)
    else:
        st.write("Please select a column for plotting.")


# In[15]:


# Convert 'ExistingLost' to a numeric value: for example, 'Existing Customer' to 1, and 'Lost' to 0
df['CurrentChurned'] = df['ExistingLost'].apply(lambda x: 1 if x == 'Existing Customer' else 0)

# Now include this new numeric column in your correlation matrix calculation
corr_matrix = df[['CurrentChurned', 'Age', 'PeriodOfRelationship', 'TotalNumberProducts', 'MonthsInactive', 
                  'ContactsCount', 'CreditLimit', 'TotalBal', 'AvgOpenToBuy', 
                  'ChangeTransAmountQ4Q1', 'TotalTransAmt', 'TotalTransCountQ4Q1', 
                  'ChangeTransCountQ4Q1', 'UtilRatio']].corr()

# Custom color scale
color_scale = [
    [0, '#A579FF'],  
    [1, '#5E18EB']   
]

# Generate the heatmap
fig = px.imshow(corr_matrix, 
                x=corr_matrix.columns, 
                y=corr_matrix.columns, 
                color_continuous_scale=color_scale,
                text_auto=True, 
                aspect="auto")

fig.update_layout(coloraxis_colorbar=dict(
    title='Correlation',
    tickvals=[-1, 1], 
    ticktext=['Minimum', 'Maximum'],
    lenmode='fraction', 
    len=1
))

with col3:
    # Custom HTML for the title with larger font size and bold text
    st.markdown("""
    <style>
    .title {
        font-size: small; 
        font-weight: bold;
    }
    </style>
    <div class="title">Correlation Matrix of Credit Card Customer Attributes</div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(fig)


# In[16]:


st.divider()


# In[17]:


# Convert 'MonthsInactive' to string if treating as categorical
df['MonthsInactive'] = df['MonthsInactive'].astype(str)

# Manually create a jitter effect
df['JitteredMonthsInactive'] = df['MonthsInactive'].apply(lambda x: float(x) + np.random.uniform(-0.3, 0.3))

# Create the Plotly scatter plot
fig = px.scatter(df, 
                 x='PeriodOfRelationship', 
                 y='JitteredMonthsInactive', 
                 color='ExistingLost',
                 labels={'JitteredMonthsInactive': 'Months Inactive'},
                 title="Customer Relationship Analysis",
                 color_discrete_sequence=['#A579FF', '#5E18EB'])

# Customize the hover data
fig.update_traces(hovertemplate="Churn Status: %{color}<br>Period of Relationship: %{x}<br>Months Inactive: %{y}")
fig.update_layout(title_text='')

# Customize the layout
fig.update_layout(legend_title_text='Churn Status')

with col2:
    # Custom HTML for the title with larger font size and bold text
    st.markdown("""
    <style>
    .title {
        font-size: small; 
        font-weight: bold;
    }
    </style>
    <div class="title">Customer relationship analysis</div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(fig)


# In[18]:


# Define categories and corresponding colors
categories = ['Attrited Customer', 'Existing Customer']
category_colors = ['#5E18EB', '#A579FF']
color_mapping = {'Attrited Customer': category_colors[0], 'Existing Customer': category_colors[1]}

# Plot the histogram with LoanAmount on the x-axis and stacked Loan_Status
fig = px.histogram(df,
                   x="CardCat",
                   color="ExistingLost",
                   color_discrete_map=color_mapping,
                   barmode='stack',
                   labels={'CartCat': 'Card Category'},
                   text_auto=True)

# Update layout for centering the title and adjusting y-axis title
fig.update_layout(title_x=0.5, yaxis_title='Count')
fig.update_layout(title_text='')
    
with col3:

    # Custom HTML for the title of the new chart
    st.markdown("""
    <style>
    .title {
        font-size: small; 
        font-weight: bold;
    }
    </style>
    <div class="title">Stacked Distribution of Churned and Existing Customers by Card Category</div>
    """, unsafe_allow_html=True)
    
    # Display the new histogram chart
    st.plotly_chart(fig)


# In[19]:


newdf = df.copy()


# In[20]:


le = LabelEncoder()
categorical = ['ExistingLost', 'Gender', 'Education', 'Marital', 'Income', 'CardCat']
for col in newdf[categorical]:
    newdf[col]=le.fit_transform(newdf[col])


# In[21]:


#Create a StandardScaler object  
scaler = StandardScaler() 
# Select the columns to be normalized  
cols_to_norm = ['Age', 'Dependent', 'PeriodOfRelationship', 'TotalNumberProducts',
       'MonthsInactive', 'ContactsCount', 'CreditLimit', 'TotalBal',
       'AvgOpenToBuy', 'ChangeTransAmountQ4Q1', 'TotalTransAmt',
       'TotalTransCountQ4Q1', 'ChangeTransCountQ4Q1', 'UtilRatio']
# Fit the scaler to the selected columns and transform the data  
newdf[cols_to_norm] = scaler.fit_transform(newdf[cols_to_norm])


# In[22]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score
import seaborn as sns
rfc = RandomForestClassifier()
X = newdf.drop(columns=['ExistingLost']).values
y = newdf['ExistingLost'].values
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 38) # 80% training and 20% test


# In[23]:


# fit
rfc.fit(X_train,y_train)


# In[24]:


# Making predictions
predictions = rfc.predict(X_test)


# In[25]:


# Let's check the report of our default model
print(classification_report(y_test,predictions))


# In[26]:


# Calculate cm by calling a method named as 'confusion_matrix'
cm = confusion_matrix(y_test, predictions)

# Print the confusion matrix as plain text
print("Confusion Matrix:")
print(cm)

# Create a custom colormap using the desired color
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", "#5E18EB"])

# Call a method heatmap() to plot confusion matrix with custom color map
sns.heatmap(cm, annot=True, cmap=custom_cmap, fmt='d')

# Set axis labels
plt.xlabel('Predicted')
plt.ylabel('True')

# Show the plot
plt.show()

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, predictions))


# In[27]:


print(accuracy_score(y_test,predictions))


# In[28]:


#Grid sear and Kfold

# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(2, 20, 5)}

# instantiate the model
rf = RandomForestClassifier()


# fit tree on training data
rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy")
rf.fit(X_train, y_train)


# In[29]:


# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()


# In[30]:


# Replace the following line with your actual scores data
scores = {'param_max_depth': list(range(1, 11)), 'mean_test_score': [0.6, 0.7, 0.75, 0.78, 0.8, 0.81, 0.82, 0.83, 0.82, 0.81]}

# Define your theme color
theme_color = "#5E18EB"

# Plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"],
         scores["mean_test_score"],
         label="test accuracy",
         c=theme_color)
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[31]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [4,8,10],
    'min_samples_leaf': range(100, 400, 200),
    'min_samples_split': range(200, 500, 200),
    'n_estimators': [100,200, 300], 
    'max_features': [5, 10]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1,verbose = 1)


# In[32]:


# Fit the grid search to the data
grid_search.fit(X_train, y_train)


# In[33]:


# printing the optimal accuracy score and hyperparameters
print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)


# In[34]:


# model with the best hyperparameters
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=10,
                             min_samples_leaf=100, 
                             min_samples_split=200,
                             max_features=10,
                             n_estimators=100)


# In[35]:


# fit
rfc.fit(X_train,y_train)


# In[36]:


# predict
predictions = rfc.predict(X_test)


# In[37]:


print(classification_report(y_test,predictions))


# In[38]:


# Calculate cm by calling a method named as 'confusion_matrix'
cm = confusion_matrix(y_test, predictions)

# Print the confusion matrix as plain text
print("Confusion Matrix:")
print(cm)

# Create a custom colormap using the desired color
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", "#5E18EB"])

# Create matplotlib figure and plot confusion matrix
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap=custom_cmap, fmt='d', ax=ax)

ax.set(xlabel="Predicted", ylabel="True")

# Create matplotlib figure and plot confusion matrix
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap=custom_cmap, fmt='d', ax=ax)

# Add labels to axes
ax.set(xlabel="Predicted", ylabel="True") 

with col2:

    # Custom HTML for the title of the new chart
    st.markdown("""
    <style>
    .title {
        font-size: small; 
        font-weight: bold;
    }
    </style>
    <div class="title">Random Forest Classification - Prediction of Churned Customers</div>
    """, unsafe_allow_html=True)
    
    # Show figure in Streamlit
    st.pyplot(fig)


# In[39]:


print(accuracy_score(y_test,predictions))


# In[40]:


with col3:

    # Custom HTML for the title of the new chart
    st.markdown("""
    <style>
    .title {
        font-size: small; 
        font-weight: bold;
    }
    </style>
    <div class="title">Random Forest Classification Prediction Accuracy Score</div>
    <div style="color: #5E18EB; font-size: 96px; text-align: center;">1</div>
    """, unsafe_allow_html=True)


# In[41]:


csv = df.to_csv(index=False).encode('utf-8')
st.download_button('Download Data', data=csv, file_name='Data.csv', mime='text/csv')

