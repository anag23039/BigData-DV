#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from bokeh.io import curdoc, output_notebook
from bokeh.plotting import figure, show, curdoc
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, Slider, NumeralTickFormatter
from bokeh.palettes import Spectral6
from bokeh.layouts import row, layout
import plotly.express as px
import random


# In[2]:


df = pd.read_csv("Adidas.xlsx - Sales.csv", encoding="ISO-8859-1")


# In[3]:


df.head()


# In[4]:


# Sample sub-categories for each main category
sub_category_mapping = {
    "Men's Apparel": ["T-Shirts", "Jackets", "Shorts", "Swimwear", "Socks"],
    "Men's Athletic Footwear": ["Running Shoes", "Training Shoes", "Basketball Shoes", "Soccer Cleats", "Tennis Shoes"],
    "Men's Street Footwear": ["Casual Sneakers", "Loafers", "Boots", "Sandals", "Dress Shoes"],
    "Women's Apparel": ["Leggings", "Sports Bras", "Tank Tops", "Skirts", "Hoodies"],
    "Women's Athletic Footwear": ["Running Shoes", "Training Shoes", "Dance Shoes", "Hiking Boots", "Cycling Shoes"],
    "Women's Street Footwear": ["Flats", "Heels", "Casual Sneakers", "Boots", "Sandals"]
}

# Function to randomly select a sub-category based on the main category
def assign_subcategory(category):
    return random.choice(sub_category_mapping[category])

# Assign a subcategory to each row in the DataFrame
df['Sub-Category'] = df['Category'].apply(assign_subcategory)

# Now df has a new 'Sub-Category' column with randomly assigned sub-categories


# In[5]:


df.head()


# In[6]:


df.isnull().sum()
# Check for any NaN values in the whole DataFrame
df.isnull().values.any()

# Check if all values are NaN in the DataFrame
df.isnull().values.all()


# In[7]:


df.info()


# In[8]:


import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Superstore!!!", page_icon=":bar_chart:",layout="wide")

st.title(" :bar_chart: Sample SuperStore EDA")


# In[9]:


st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

fl = st.file_uploader(":file_folder: Upload a file",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
else:
    os.chdir("/Users/annageorgieva/Documents/GitHub/BigData-DV")


# In[10]:


col1, col2 = st.columns((2))
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Getting the min and max date 
startDate = pd.to_datetime(df["Order Date"]).min()
endDate = pd.to_datetime(df["Order Date"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))
    df = df[(df["Order Date"] >= date1) & (df["Order Date"] <= date2)].copy()


# In[11]:


st.sidebar.header("Choose your filter: ")
# Create for Region
region = st.sidebar.multiselect("Pick your Region", df["Region"].unique())
if not region:
    df2 = df.copy()
else:
    df2 = df[df["Region"].isin(region)]


# In[12]:


# Create for State
state = st.sidebar.multiselect("Pick the State", df2["State"].unique())
if not state:
    df3 = df2.copy()
else:
    df3 = df2[df2["State"].isin(state)]

# Create for City
city = st.sidebar.multiselect("Pick the City",df3["City"].unique())

if not region and not state and not city:
    filtered_df = df
elif not state and not city:
    filtered_df = df[df["Region"].isin(region)]
elif not region and not city:
    filtered_df = df[df["State"].isin(state)]
elif state and city:
    filtered_df = df3[df["State"].isin(state) & df3["City"].isin(city)]
elif region and city:
    filtered_df = df3[df["Region"].isin(region) & df3["City"].isin(city)]
elif region and state:
    filtered_df = df3[df["Region"].isin(region) & df3["State"].isin(state)]
elif city:
    filtered_df = df3[df3["City"].isin(city)]
else:
    filtered_df = df3[df3["Region"].isin(region) & df3["State"].isin(state) & df3["City"].isin(city)]


# In[13]:


category_df = filtered_df.groupby(by = ["Sub-Category"], as_index = False)["Sales"].sum()


# In[15]:


with col1:
    st.subheader("Product wise Sales")
    # Ensure that TotalSales is a float and handle any potential conversion errors
    category_df["Sales"] = pd.to_numeric(category_df["Sales"], errors='coerce')
    # Now, you can safely format the numbers
    fig = px.bar(category_df, x="Sub-Category", y="Sales",
                 text=['${:,.2f}'.format(x) for x in category_df["Sales"]],
                 template="seaborn")
    st.plotly_chart(fig, use_container_width=True, height=200)


# In[16]:


with col2:
    st.subheader("Region wise Sales")
    fig = px.pie(filtered_df, values = "Sales", names = "Region", hole = 0.5)
    fig.update_traces(text = filtered_df["Region"], textposition = "outside")
    st.plotly_chart(fig,use_container_width=True)


# In[17]:


cl1, cl2 = st.columns((2))
with cl1:
    with st.expander("Category_ViewData"):
        st.write(category_df.style.background_gradient(cmap="Blues"))
        csv = category_df.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data = csv, file_name = "Category.csv", mime = "text/csv",
                            help = 'Click here to download the data as a CSV file')


# In[18]:


with cl2:
    with st.expander("Region_ViewData"):
        region = filtered_df.groupby(by = "Region", as_index = False)["Sales"].sum()
        st.write(region.style.background_gradient(cmap="Oranges"))
        csv = region.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data = csv, file_name = "Region.csv", mime = "text/csv",
                        help = 'Click here to download the data as a CSV file')


# In[19]:


filtered_df["month_year"] = filtered_df["Order Date"].dt.to_period("M")
st.subheader('Time Series Analysis')


# In[20]:


linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month_year"].dt.strftime("%Y : %b"))["Sales"].sum()).reset_index()
fig2 = px.line(linechart, x = "month_year", y="Sales", labels = {"Sales": "Amount"},height=500, width = 1000,template="gridon")
st.plotly_chart(fig2,use_container_width=True)


# In[27]:


with st.expander("View Data of TimeSeries:"):
    st.write(linechart.T.style.background_gradient(cmap="Blues"))
    csv = linechart.to_csv(index=False).encode("utf-8")
    st.download_button('Download Data', data = csv, file_name = "TimeSeries.csv", mime ='text/csv')


# In[22]:


# Remove dollar signs, commas, and whitespace
filtered_df['Sales'] = filtered_df['Sales'].replace({'\$': '', ',': '', ' ': ''}, regex=True)

# Convert the 'Sales' column to a numeric data type
filtered_df['Sales'] = pd.to_numeric(filtered_df['Sales'], errors='coerce')


# In[23]:


# Create a treem based on Region, category, sub-Category
st.subheader("Hierarchical view of Sales using TreeMap")
fig3 = px.treemap(filtered_df, path = ["Region","Category","Sub-Category"], values = "Sales",hover_data = ["Sales"],
                  color = "Sub-Category")
fig3.update_layout(width = 800, height = 650)
st.plotly_chart(fig3, use_container_width=True)


# In[25]:


with chart1:
    st.subheader('Category wise Sales')
    fig = px.pie(filtered_df, values="Sales", names="Category", template="plotly_dark")
    fig.update_traces(textposition="inside")
    st.plotly_chart(fig, use_container_width=True)


# In[26]:


with chart2:
    st.subheader('Category wise Sales')
    fig = px.pie(filtered_df, values = "Sales", names = "Category", template = "gridon")
    fig.update_traces(text = filtered_df["Category"], textposition = "inside")
    st.plotly_chart(fig,use_container_width=True)


# In[28]:


import plotly.figure_factory as ff
st.subheader(":point_right: Month wise Sub-Category Sales Summary")
with st.expander("Summary_Table"):
    df_sample = df[0:5][["Region","State","City","Category","Sales","Profit","UnitsSold"]]
    fig = ff.create_table(df_sample, colorscale = "Cividis")
    st.plotly_chart(fig, use_container_width=True)


# In[31]:


st.markdown("Month wise sub-Category Table")
filtered_df["month"] = filtered_df["Order Date"].dt.month_name()
sub_category_Year = pd.pivot_table(data = filtered_df, values = "Sales", index = ["Sub-Category"],columns = "month")
st.write(sub_category_Year.style.background_gradient(cmap="Blues"))


# In[34]:


# Remove commas and convert to numeric
filtered_df['UnitsSold'] = filtered_df['UnitsSold'].str.replace(',', '')
filtered_df['UnitsSold'] = pd.to_numeric(filtered_df['UnitsSold'], errors='coerce')

# Now you can create the scatter plot
data1 = px.scatter(filtered_df, x="Sales", y="Profit", size="UnitsSold")
data1['layout'].update(title="Relationship between Sales and Profits using Scatter Plot.",
                       titlefont=dict(size=20), xaxis=dict(title="Sales", titlefont=dict(size=19)),
                       yaxis=dict(title="Profit", titlefont=dict(size=19)))
st.plotly_chart(data1, use_container_width=True)


# In[ ]:


filtered_df['UnitsSold'] = pd.to_numeric(filtered_df['UnitsSold'], errors='coerce')


# In[35]:


with st.expander("View Data"):
    st.write(filtered_df.iloc[:500,1:20:2].style.background_gradient(cmap="Oranges"))


# In[36]:


# Download orginal DataSet
csv = df.to_csv(index = False).encode('utf-8')
st.download_button('Download Data', data = csv, file_name = "Data.csv",mime = "text/csv")


# In[ ]:




