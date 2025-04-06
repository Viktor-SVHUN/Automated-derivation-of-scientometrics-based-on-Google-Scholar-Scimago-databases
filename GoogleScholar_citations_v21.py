from scholarly import scholarly
import pandas as pd
from datetime import datetime
import os
import plotly.graph_objects as go
from collections import Counter
import requests
from io import StringIO
import networkx as nx
import numpy as np
import time
from statsmodels.tsa.arima.model import ARIMA
import random
from rapidfuzz import fuzz, process
import re

# Count the length of the run
start_time = time.time()

# Search for the profile by name or specific user ID
search_query = scholarly.search_author('Viktor Sebestyen')  # or use 'g0_46QEAAAAJ' for user ID
author = next(search_query)

# Fill in the complete author information
author = scholarly.fill(author)

# Extract the number of citations
Total_number_of_citations = author['citedby']
print(f"Total citations: {Total_number_of_citations}")

#%% List the counts
# Get today's date in the format 'YYYY-MM-DD'
today = datetime.today().strftime('%Y-%m-%d')

# Search for the author profile by user ID
author = scholarly.search_author_id('g0_46QEAAAAJ')

# Fill in the author details, including their publications
author = scholarly.fill(author, sections=['publications'])

# Create lists to store the data
data = []
titles = []
years = []
journals = []
authors = []

# Loop through each publication and store the title, citation count, and publication year
for pub in author['publications']:
    pub = scholarly.fill(pub)
    title = pub['bib']['title']
    pub_year = pub['bib'].get('pub_year', 'Unknown')  # Handle missing year data
    journal = pub['bib'].get('journal', 'Unknown')  # Handle missing journal data
    citations = pub.get('num_citations', 0)  # Handle missing citation data
    author_list = pub['bib'].get('author', '')  # Extract the authors as a single string
    
    
    titles.append(title)
    years.append(pub_year)
    journals.append(journal)
    data.append(citations)
    authors.append(author_list)  # Store the authors' string


# Initialize the coauthor graph
coauthor_graph = nx.Graph()

# Loop through each publication's authors list and add edges between coauthors
for author_str in authors:
    # Split the authors for each publication (assuming they are separated by "and")
    authors_list = [author.strip() for author in author_str.split(" and")]
    
    # Add edges between all coauthors in this publication
    for i in range(len(authors_list)):
        for j in range(i + 1, len(authors_list)):
            author1 = authors_list[i]
            author2 = authors_list[j]
            
            if coauthor_graph.has_edge(author1, author2):
                # If the edge already exists, increment the weight
                coauthor_graph[author1][author2]['weight'] += 1
            else:
                # Otherwise, create a new edge with weight 1
                coauthor_graph.add_edge(author1, author2, weight=1)

# Create a DataFrame for the new data, including publication years
new_data = pd.DataFrame({'Title': titles, today: data})

# File path for the Excel file
file_path = 'google_scholar_citations.xlsx'

if os.path.exists(file_path):
    # Load the existing Excel file
    existing_data = pd.read_excel(file_path)

    # Merge the new data with the existing data
    merged_data = pd.merge(existing_data, new_data, on='Title', how='outer')

else:
    # If the file does not exist, start with the new data
    merged_data = new_data

# Save the updated DataFrame to the Excel file
merged_data.to_excel(file_path, index=False)

print(f"Results saved to {file_path}")

#%% Track institutional rank. THIS TAKES A LOT OF TIME...
#Search for authors by an institution: you should use the observed institute name in the parenthesis.
# Inst_search_query = scholarly.search_author('University of Pannonia')

# # Loop through the search results and fetch details
# for author in Inst_search_query:
#     Inst_author_filled = scholarly.fill(author)  # Fill out the author's profile
#     print(f"Author: {Inst_author_filled['name']}, Citations: {Inst_author_filled['citedby']}")

# A query jo csak sajnos kiakad mert tul sok az adat.... 

#%% Lets dash it
# Step 1: Create a Plotly line plot
def create_line_plot():
    # Melt the DataFrame to have 'Date' and 'Citations' as separate columns
    df_melted = merged_data.melt(id_vars=['Title'], var_name='Date', value_name='Citations')

    # Create an empty figure
    fig = go.Figure()

    # Loop through each publication and plot its citation trend
    for title in df_melted['Title'].unique():
        subset = df_melted[df_melted['Title'] == title]
        
        # Add line traces for each publication
        fig.add_trace(go.Scatter(
            x=subset['Date'], 
            y=subset['Citations'], 
            mode='lines+markers', 
            name=title
        ))

    # Customize the layout (you can adjust this later)
    fig.update_layout(
        title="Citation Trends Over Time for Each Publication",
        xaxis_title="Date",
        yaxis_title="Citations",
        xaxis=dict(tickangle=-45),
        legend_title="Publication"
    )
    
    return fig

# Convert the 'years' list into a DataFrame with a column name 'Year'
years_df = pd.DataFrame(years, columns=['Year'])

publications_per_year = years_df.groupby('Year').size().reset_index(name='Publication_Count')

# Calculate the academic age of the author
earliest_year = publications_per_year['Year'].min()

# Get the current year
current_year = datetime.now().year

# Calculate the elapsed years
academic_age = current_year - earliest_year

# Access the second and last columns (ignoring the 'Title' column)
second_column = merged_data.iloc[:, 1]  # Second column (first citation count column)
last_column = merged_data.iloc[:, -1]   # Last column (most recent citation count column)

# Calculate the citation difference without modifying merged_data
citation_diff = merged_data.iloc[:, -1] - merged_data.iloc[:, 2]

# Find the index of the maximum difference in citation count
max_diff_index = citation_diff.idxmax()

# Retrieve the row corresponding to the maximum difference
most_improved_paper_row = merged_data.loc[max_diff_index]

# Extract the paper title from the second column (index 1)
most_improved_paper_title = most_improved_paper_row[0]

# Extract the maximum citation difference
max_citation_diff = int(citation_diff[max_diff_index])


# Calculate the sum of each column
sum_second_column = second_column.sum()
sum_last_column = last_column.sum()

# Calculate the difference between the sums
difference = int(sum_last_column - sum_second_column)

num_papers = len(merged_data)

avg_cit = str(round(Total_number_of_citations/num_papers, 2))

avg_ann_cit = str(round(Total_number_of_citations/academic_age, 2))

avg_ann_pap = str(round(num_papers/academic_age, 2))

zero_citt_ratio = str(round((data.count(0)/num_papers)*100, 1))

author_name = author["name"]
single_aut = authors.count(author_name)

first_aut_count = sum(1 for a in authors if a.startswith(author_name)) - authors.count(author_name)

supervised_aut_count = sum(1 for a in authors if a.endswith(author_name)) - authors.count(author_name)

disting_pos = round(((first_aut_count+supervised_aut_count)/num_papers)*100, 1)

# Total number of coauthors
Num_coauthors = len(coauthor_graph.nodes())-1 # -1 because of the analyzed author name

# Predictive modeling of the expected number of citations
# Sum the values of each date column (excluding the 'Title' column)
total_cit = merged_data.iloc[:, 1:].sum(axis=0)

# Convert the result into a DataFrame with column labels and sums
total_cit_df = pd.DataFrame({
    'Date': total_cit.index,
    'Total_Citations': total_cit.values
})

# Ensure the 'Date' column is a datetime object
total_cit_df['Date'] = pd.to_datetime(total_cit_df['Date'])

# Calculate the elapsed days
rep_elapsed_days = (total_cit_df['Date'].max() - total_cit_df['Date'].min()).days

# Set the 'Date' column as the index
total_cit_df.set_index('Date', inplace=True)

# Resample to daily data if necessary (this assumes you have daily or lower frequency data)
daily_data = total_cit_df.resample('D').sum()

# Replace zeros or missing data with the previous row's value
daily_data = daily_data.replace(0, pd.NA)  # Replace 0s with NaN (missing data)
daily_data.fillna(method='ffill', inplace=True)  # Forward fill the NaN values

# Train an ARIMA model (you may need to adjust the p, d, q parameters for best results)
model = ARIMA(daily_data['Total_Citations'], order=(0, 2, 2))
model_fit = model.fit()

# Forecast for 365 days (1 year)
forecast = model_fit.forecast(steps=365-rep_elapsed_days) #Here you can modify the lenght of predicted time interval (!!!!!dont forget to adjust the periods 3 lines below!!!!!)

# Create a DataFrame for the forecasted results
forecast_dates = pd.date_range(start=daily_data.index[-1], periods=365-rep_elapsed_days, freq='D')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_Citations': forecast})

# Calculate the difference between consecutive rows in 'Total_Citations'
total_cit_df['Change'] = total_cit_df['Total_Citations'].diff()

# Find the maximum change
max_change = int(total_cit_df['Change'].max())

avg_day_cit = str(round(difference/rep_elapsed_days, 2))

# Step 2: Create a line chart for Citations Forecast
def create_citations_forecast_chart(daily_data, forecast_df):
    # Create a line chart using Plotly
    fig = go.Figure()

    # Historical Citations line
    fig.add_trace(go.Scatter(
        x=daily_data.index, 
        y=daily_data['Total_Citations'], 
        mode='lines', 
        name='Historical Citations (observations)'
    ))

    # Forecasted Citations line
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'], 
        y=forecast_df['Predicted_Citations'], 
        mode='lines', 
        name='Forecasted Citations',
        line=dict(color='orange')
    ))

    # Customize the layout
    fig.update_layout(
        title="ARIMA citations forecast",
        xaxis_title="Date",
        yaxis_title="Total number of citations",
        template="plotly_white"
    )

    return fig

# Step 3: Extract the last date and calculate the elapsed time
def get_report_info():
    # Extract column names excluding the first two (index and title)
    date_columns = merged_data.columns[1:]  # Skip the 'Title' column

    # Get the first and last date
    first_date = pd.to_datetime(date_columns[1])
    last_date = pd.to_datetime(date_columns[-1])
    
    # Calculate the elapsed time between first and last dates
    elapsed_time = last_date - first_date
    
    # Report information
    report_info = {
        "date_of_report": last_date.strftime("%Y-%m-%d"),  # Format last date as string
        "observed_period": elapsed_time
    }
    
    return report_info


# Step 5: Create H-index Plot
def create_h_index_plot(citations_data):
    # Sort citations in descending order
    sorted_citations = sorted(citations_data, reverse=True)

    # Calculate the h-index
    h_index = 0
    for i, citation in enumerate(sorted_citations):
        if citation >= i + 1:
            h_index = i + 1
        else:
            break

    # Create the line chart for the citations
    fig = go.Figure()

    # Plot the citations (orange line plot)
    fig.add_trace(go.Scatter(
        x=list(range(1, len(sorted_citations) + 1)),
        y=sorted_citations,
        mode='lines+markers',
        name='Citations per paper',
        line=dict(color='orange'),
        fill='tozeroy',  # Fill area below the line
        fillcolor='rgba(255,165,0,0.3)'  # Orange with 30% opacity
    ))

    # Add the reference diagonal line for H-index threshold (blue dashed line with fill)
    diagonal_line = list(range(1, len(sorted_citations) + 1))  # Line where y = x

    fig.add_trace(go.Scatter(
        x=diagonal_line,
        y=diagonal_line,
        mode='lines',
        line=dict(color='blue', dash='dash'),
        name='H-index threshold',
        fill='tozeroy',  # Fill area below the line
        fillcolor='rgba(0,0,255,0.3)'  # Blue with 30% opacity
    ))

    # Highlight the intersection point (h-index) with a gold star
    fig.add_trace(go.Scatter(
        x=[h_index],
        y=[h_index],
        mode='markers',
        marker=dict(color='gold', size=12, symbol='star'),
        name=f'H-index = {h_index}'
    ))

    # Customize the layout
    fig.update_layout(
        title=f'H-index Visualization (H-index = {h_index})',
        xaxis_title='Paper Rank (sorted by citations)',
        yaxis_title='Number of Citations',
        xaxis=dict(tickformat='d'),  # Ensure integer ticks on x-axis
        yaxis=dict(tickformat='d'),  # Ensure integer ticks on y-axis
        showlegend=True,
        template='plotly_white'
    )

    return fig


# Count the occurrences of each journal in the 'journals' list
journal_counts = Counter(journal for journal in journals if journal != "Unknown")

# Create a DataFrame from the journal count data
journal_df = pd.DataFrame({
    'Journal': list(journal_counts.keys()),
    'Paper Count': list(journal_counts.values())
})

# Step 2: Create a Pie Chart for journal distribution
# Function to generate random hex color codes
def generate_random_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        colors.append(color)
    return colors

# Step 2: Create a Pie Chart for journal distribution
def create_journal_pie_chart(journal_data):
    # Generate random colors for each slice
    random_colors = generate_random_colors(len(journal_data['Journal']))

    # Create a pie chart using Plotly
    fig = go.Figure(data=[go.Pie(
        labels=journal_data['Journal'],
        values=journal_data['Paper Count'],
        hoverinfo='label+percent',
        textinfo='value',
        pull=[0.05]*len(journal_data['Journal']),  # Slightly "explode" each slice
        marker=dict(colors=random_colors)  # Assign random colors
    )])

    # Customize the layout
    fig.update_layout(
        title="Distribution of Papers Across Different Journals",
        template="plotly_white"
    )

    return fig

# Create the journal pie chart
journal_pie_fig = create_journal_pie_chart(journal_df)


# Function to download SJR journal ranking data for a specific year (CSV file)
def journal_url(year):
    return f"https://www.scimagojr.com/journalrank.php?year={year}&out=xls"

# Function to download and load the CSV data
def download_sjr_data(year):
    url = journal_url(year)
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    # Read the CSV file from the content
    return pd.read_csv(StringIO(response.content.decode('utf-8')), sep=';')

# Define the range of years you want to collect data for
years = range(earliest_year, current_year+1)  # From 1999 to 2017

# Initialize an empty DataFrame to store all the data
all_journals_df = pd.DataFrame()

# Loop over each year and download data
for year in years:
    print(f"Downloading data for year {year}...")
    df_year = download_sjr_data(year)
    df_year['year'] = year  # Add a column for the year
    all_journals_df = pd.concat([all_journals_df, df_year], ignore_index=True)

# Clean the column names
all_journals_df.columns = all_journals_df.columns.str.lower().str.replace(' ', '_')

# D1 CATEGORY DETECTION
# Step 1: Split the 'areas' column into individual subject areas
# Assuming 'areas' contains subject areas separated by semicolons
all_journals_df['areas_split'] = all_journals_df['areas'].str.split('; ')

# Step 2: Explode the 'area_split' column to have each subject area in its own row
exploded_df = all_journals_df.explode('areas_split')

# Step 1: Sort the DataFrame by 'year', 'area_split', and 'SJR'
exploded_df = exploded_df.sort_values(by=['year', 'areas_split', 'sjr'], ascending=[True, True, False])

# Step 2: Use groupby and cumcount to calculate the rank of each journal occurrence within the given 'year' and 'area_split'
exploded_df['rank_own'] = exploded_df.groupby(['year', 'areas_split']).cumcount() + 1  # Start the count from 1

# Use transform to get the count back into the original DataFrame
exploded_df['subject_area_journal_count'] = exploded_df.groupby(['year', 'areas_split'])['title'].transform('count')

# Step 1: Calculate 'Percentile_own' as the ratio of 'rank' to 'subject_area_count' multiplied by 100
exploded_df['Percentile_own'] = (exploded_df['rank_own'] / exploded_df['subject_area_journal_count']) * 100

# Step 1: Group by 'title' and get the minimum 'Percentile_own'
min_percentile_own = exploded_df.groupby('title')['Percentile_own'].min().reset_index()

# Step 2: Rename the column for clarity
min_percentile_own.rename(columns={'Percentile_own': 'Min_Percentile_own'}, inplace=True)

# Step 3: Merge this information into all_journals_df
all_journals_df = pd.merge(all_journals_df, min_percentile_own, how='left', on='title')

# Step 1: Define the bins and labels for quartiles
bins = [-float('inf'), 10, 25, 50, 75, float('inf')]  # Define the bin edges
quartiles = ['D1', 'Q1', 'Q2', 'Q3', 'Q4']  # Define corresponding labels

# Step 2: Create the 'sjr_best_quartile_own' column using pd.cut
all_journals_df['sjr_best_quartile_own'] = pd.cut(
    all_journals_df['Min_Percentile_own'], 
    bins=bins, 
    labels=quartiles, 
    right=False  # Ensure intervals are [a, b)
)

# Filter the combined DataFrame to only include the journals from the list
filtered_df = all_journals_df[all_journals_df['title'].isin(journals)]

# Optionally, save the filtered data to a CSV file
filtered_df.to_csv('selected_journal_metrics.csv', index=False)

# Ensure years_df is in the correct format
years = years_df['Year'].tolist()  # Convert years_df to a list of years

# Ensure journals and years match in size
if len(journals) != len(years):
    raise ValueError("The size of 'journals' list and 'years_df' must be the same.")

# Create a DataFrame with 'journals' and 'years' from the lists
combined_df = pd.DataFrame({
    'title': journals,
    'year': years
})

# Merge combined_df with filtered_df to get 'sir_best_quartile' for each journal and year
result_df = pd.merge(combined_df, filtered_df, how='left', on=['title', 'year'])

# Select only the relevant columns: 'title', 'year', and 'sir_best_quartile' -----> "_own" ended sjr_best_quartile means that this value is calculated by the author.
result_df = result_df[['title', 'year', 'sjr_best_quartile', 'sjr_best_quartile_own']]

# ROUGH ESTIMATION OF IMPACT FACTORS 
# Merge combined_df with filtered_df to get 'cites_/_doc._(2years)' for each journal and year
rough_if_df = pd.merge(combined_df, filtered_df, how='left', on=['title', 'year'])

# Select only the relevant columns: 'title', 'year', and 'cites_/_doc._(2years)'
rough_if_df = rough_if_df[['title', 'year', 'cites_/_doc._(2years)']]
rough_if_df = rough_if_df.dropna(subset=['cites_/_doc._(2years)'])

# Replace commas with periods in 'cites_/_doc._(2years)' column
rough_if_df['cites_/_doc._(2years)'] = rough_if_df['cites_/_doc._(2years)'].str.replace(',', '.')

# Convert the column to numeric (float) type
rough_if_df['cites_/_doc._(2years)'] = pd.to_numeric(rough_if_df['cites_/_doc._(2years)'], errors='coerce')

cumulated_if = round(rough_if_df['cites_/_doc._(2years)'].sum(), 2)

# Group by year and calculate the sum of 'cites_/_doc.(2years)' per year
annual_IF_sum = rough_if_df.groupby('year')['cites_/_doc._(2years)'].sum().reset_index()

# Find the index of the row with the maximum 'cites_/doc._(2years)' value
max_index = rough_if_df['cites_/_doc._(2years)'].idxmax()

# Extract the title and the citation value based on the max index
max_IF = rough_if_df.loc[max_index, ['title', 'cites_/_doc._(2years)']]

# Extract the paper title from the Series (index 0)
max_IF_paper_title = max_IF['title']

# Extract the maximum citation difference from the Series (index 1)
max_IF_paper_IF = max_IF['cites_/_doc._(2years)']

# Clear the errors of the Results df: This is especially important if a journal is listed more than once by the code, since in this case there are several hits with the exact same name that it assigns.
result_df = result_df.drop(index=5) # drop the Sustainability journal bad entry.

# Step 1: Identify categorical columns
categorical_columns = result_df.select_dtypes(include=['category']).columns

# Step 2: For each categorical column, add 'other' to the categories
for col in categorical_columns:
    result_df[col] = result_df[col].cat.add_categories('other')

# Step 3: Replace NaN values with 'other' in all categorical columns
result_df = result_df.fillna('other')

result_df.loc[
    (result_df['sjr_best_quartile'] == 'Q1') & (result_df['sjr_best_quartile_own'] == 'D1'),
    'sjr_best_quartile'
] = 'D1'

# Optionally, save the result to a CSV
result_df.to_csv('journal_quartiles.csv', index=False)

# We should calculate the annual achievements based on the Scimago database classification: Group by 'year' and 'sir_best_quartile' and count occurrences
grouped_data = result_df.groupby(['year', 'sjr_best_quartile']).size().unstack(fill_value=0)

avg_IF_per_paper = str(round(cumulated_if/num_papers, 2))

#Track the number of couathorships
# Calculate the number of authors for each paper
num_authors = [len(author.split(" and ")) for author in authors]

# Calculate the average number of coauthors
average_coauthors = str(round(sum(num_authors) / len(authors), 2))

# DETECTION OF THE PUBLISHERS 
# Merge combined_df with filtered_df to get 'publisher' for each journal and year
publisher_df = pd.merge(combined_df, filtered_df, how='left', on=['title', 'year'])

# Select only the relevant columns: 'title', 'year', and 'publisher'
publisher_df = publisher_df[['title', 'year', 'publisher']]

# Drop rows where 'publisher' is NaN
publisher_df = publisher_df.dropna(subset=['publisher'])

# Step 1: Group by 'publisher' and count the number of occurrences
publisher_count = publisher_df['publisher'].value_counts().reset_index()
publisher_count.columns = ['Publisher', 'Publication_Count']

# Clean the publisher_count df in order to avoid the unnecessary repetitions
def clean_publisher_name(name):
    # Define suffixes to remove
    suffixes = r"(B\.V\.|Ltd|Inc\.|LLC|Co\.|Corp\.)$"
    # Remove these suffixes
    return re.sub(suffixes, "", name).strip()

# Apply cleaning to the 'Publisher' column
publisher_count['Publisher_Cleaned'] = publisher_count['Publisher'].apply(clean_publisher_name)

# Create an empty dictionary to store grouped publishers
grouped_publishers = {}

# Similarity threshold (adjust as needed)
threshold = 90  

# Iterate through cleaned publisher names
for i, publisher in enumerate(publisher_count['Publisher_Cleaned']):
    # Check if publisher is already grouped
    if publisher in grouped_publishers:
        continue
    
    # Compare with other cleaned publishers
    similar = process.extract(publisher, publisher_count['Publisher_Cleaned'], scorer=fuzz.token_sort_ratio)
    
    # Filter similar names above the threshold
    similar_names = [x[0] for x in similar if x[1] >= threshold]
    
    # Assign the group name as the first match
    group_name = similar_names[0]
    
    # Group all similar names
    for name in similar_names:
        grouped_publishers[name] = group_name

# Replace similar names with the group name in the DataFrame
publisher_count['Publisher_Grouped'] = publisher_count['Publisher_Cleaned'].map(grouped_publishers)

# Aggregate Publication_Count by grouped names
merged_publisher_count = publisher_count.groupby('Publisher_Grouped', as_index=False).agg({
    'Publication_Count': 'sum'
})

#print(merged_publisher_count)

# Step 2: Create a Plotly bar plot with unique colors for each publisher
# Function to generate a random color in hex format
def random_color():
    return f'#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}'

# Step 2: Create a Plotly bar plot with random colors for each publisher
def create_merged_publisher_bar_plot(merged_publisher_count):
    # Generate a list of random colors for each publisher
    colors = [random_color() for _ in range(len(merged_publisher_count))]

    # Create a bar plot with each bar having a random color
    fig = go.Figure([go.Bar(
        x=merged_publisher_count['Publisher_Grouped'], 
        y=merged_publisher_count['Publication_Count'], 
        text=merged_publisher_count['Publication_Count'],
        textposition='auto',
        marker=dict(
            color=colors  # Apply random colors to each bar
        )
    )])

    # Customize the layout
    fig.update_layout(
        title="Cumulative Number of Publications by Publisher",
        xaxis_title="Publisher",
        yaxis_title="Number of Publications",
        xaxis=dict(type='category'),  # Ensure that publishers are treated as categories
        yaxis=dict(tickformat='d'),  # Force y-axis to show integer values
        template="plotly_white"
    )

    return fig

# Step 3: Call the function to create the plot
fig = create_merged_publisher_bar_plot(merged_publisher_count)
fig.show()

# Step 2: Create a stacked bar plot using Plotly
def create_quartile_bar_plot(grouped_data):
    
    # Define colors for each quartile
    quartile_colors = {
        'D1': '#00B050',  # Dark green
        'Q1': '#ADD473',  # Light green
        'Q2': '#EAD96A',  # Yellow
        'Q3': '#FBAC65',  # Orange
        'Q4': '#E06A5F',  # Red
        'other': '#DDEBF7'  # Light blue
    }
    
    # Create an empty figure
    fig = go.Figure()

     # Add a bar for each quartile with its corresponding color and numbers
    for quartile in grouped_data.columns:
        fig.add_trace(go.Bar(
            x=grouped_data.index,  # Years
            y=grouped_data[quartile],  # Count for each quartile
            name=quartile,  # Label for each quartile
            marker_color=quartile_colors.get(quartile, '#DDEBF7'),  # Use defined color or default light blue
            text=grouped_data[quartile],  # Show numbers on the bars
            textposition='auto'  # Automatically position the text
        ))

    # Calculate cumulative sums for each year
    cumulative_totals = grouped_data.cumsum(axis=1).iloc[:, -1]

    # Add the cumulative total text on top of the stacked bars with increased font size and bold style
    fig.add_trace(go.Scatter(
        x=grouped_data.index,
        y=cumulative_totals,  # Cumulative sum for each year
        mode='text',
        text=cumulative_totals,  # Display the cumulative total as text
        textposition='top center',  # Place the text above the bars
        textfont=dict(size=18, color='black', family='Arial Black'),  # Increase font size and use a bold font
        showlegend=False  # Hide this from the legend
    ))

    # Customize the layout for stacked bars
    fig.update_layout(
        title="Number of Journals per Quartile by Year",
        xaxis_title="Year",
        yaxis_title="Number of Journals",
        barmode='stack',  # Stack the bars
        xaxis=dict(type='category'),  # Ensure that years are treated as categories
        yaxis=dict(tickformat='d'),  # Force y-axis to show integer values
        template="plotly_white"
    )

    return fig


num_Q1_papers = grouped_data["Q1"].sum() + grouped_data["D1"].sum()
num_D1_papers = grouped_data["D1"].sum()

# Create the plot and display it
fig = create_quartile_bar_plot(grouped_data)
fig.show()

# Create a Plotly bar plot for IF tracking
def create_IF_per_year_bar_plot(annual_IF_sum):
    # Create a bar plot with viridis colorscale
    fig = go.Figure([go.Bar(
        x=annual_IF_sum['year'], 
        y=annual_IF_sum['cites_/_doc._(2years)'], 
        text=annual_IF_sum['cites_/_doc._(2years)'].round(2),
        texttemplate='%{text:.2f}',  # Format the labels to 2 decimal places
        textposition='auto',
        marker=dict(
            color=annual_IF_sum['cites_/_doc._(2years)'],  # Set color based on values
            colorscale='Viridis',  # Use the viridis colorscale
            colorbar=dict(title="IF")  # Show color bar with title
        )
    )])

    # Customize the layout
    fig.update_layout(
        title="Annual rough estimated cumulated impact factors",
        xaxis_title="Year",
        yaxis_title="Cumulated impact factor",
        xaxis=dict(type='category'),  # Ensure that years are treated as categories
        yaxis=dict(tickformat='d'),  # Force y-axis to show integer values
        template="plotly_white"
    )

    return fig

# Call the function to create the plot
fig = create_IF_per_year_bar_plot(annual_IF_sum)

# Show the plot
fig.show()

def create_monthly_citations_bar_plot(total_cit_df):
    # Ensure that the index is a datetime index
    total_cit_df.index = pd.to_datetime(total_cit_df.index)

    # Resample the data to get the total change in citations per month
    monthly_citations = total_cit_df.resample('M').sum()

    # Create a bar plot using Plotly
    fig = go.Figure([go.Bar(
        x=monthly_citations.index.strftime('%Y-%m'),  # Format dates as Year-Month
        y=monthly_citations['Change'],  # The total citation changes per month
        text=monthly_citations['Change'].round(2),  # Show the rounded change values on the bars
        texttemplate='%{text:.2f}',  # Format the labels to 2 decimal places
        textposition='auto',
        marker=dict(
            color=monthly_citations['Change'],  # Set bar color based on values
            colorscale='Magma',  # Use your plotly colorscale
            colorbar=dict(title="Monthly citations")  # Show color bar with title
        )
    )])

    # Customize the layout
    fig.update_layout(
        title="Monthly number of citations",
        xaxis_title="Month",
        yaxis_title="Total number of itations",
        xaxis=dict(type='category'),  # Ensure months are treated as categories
        yaxis=dict(tickformat='d'),  # Force y-axis to show integer values
        template="plotly_white"
    )

    return fig

# Step 5: Visualize Coauthor Network with Adjusted Node Size Scaling
def create_coauthor_network_plot(coauthor_graph):
    # Create the positions for the nodes in a spring layout
    pos = nx.spring_layout(coauthor_graph)
    
    # Calculate node sizes based on the sum of edge weights (co-occurrences)
    node_edge_weights = {}
    for node in coauthor_graph.nodes():
        total_weight = sum(data['weight'] for _, _, data in coauthor_graph.edges(node, data=True))
        node_edge_weights[node] = total_weight  # Sum of weights for each node

    # Apply a square root transformation to reduce the size gap
    node_sizes = [np.sqrt(node_edge_weights[node]) * 5 for node in coauthor_graph.nodes()]  # Scale factor for visibility

    # Create the Plotly scatter plot data for the nodes
    node_x = []
    node_y = []
    node_labels = []
    for node in coauthor_graph.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(node)
    
    # Create the scatter plot for the edges
    edge_x = []
    edge_y = []
    
    # Iterate through edges and add edge weights (co-occurrences)
    for edge in coauthor_graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Use the weight (co-author occurrence count)
        weight = edge[2].get('weight', 1)  # The calculated co-author occurrences

    # Plotly traces for edges
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Plotly traces for nodes
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',  # Add '+text' to show labels
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_sizes,  # Set size based on the scaled sum of edge weights
            color=list(node_edge_weights.values()),  # Color nodes by their sum of edge weights
            colorbar=dict(
                thickness=15,
                title='Total Edge Weights',
                xanchor='left',
                titleside='right'
            )
        ),
        text=node_labels,  # Set the node labels
        textposition="top center"  # Position of the labels
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],  # Remove edge_text trace
                    layout=go.Layout(
                        title="Square root transformed co-author network",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        annotations=[dict(
                            text=" ",
                            showarrow=False,
                            xref="paper", yref="paper"
                        )],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    
    return fig

# DETECTION OF THE ACTIVE PAPERS
# Assuming citation_diff is a Series, years_df and new_data are DataFrames
activ_df = pd.concat([years_df, new_data, citation_diff], axis=1)

# Optionally, you can rename the columns if needed
activ_df.columns = ['Year', 'Title', 'Total_num_cit', 'Citation_Diff']

# Replace NaN values with 0 in the 'Citation_Diff' column
activ_df['Citation_Diff'] = activ_df['Citation_Diff'].fillna(0)

# Calculate the age and add it as a new column
activ_df['Paper_age'] = current_year - activ_df['Year']

# Calculate the average annual citation and add it as a new column
activ_df['Avg_annual_cit'] = activ_df['Total_num_cit'] / activ_df['Paper_age']

activ_df['Avg_annual_cit'] = activ_df['Avg_annual_cit'].fillna(0)

# Calculate 'Time_prop_citations' and add it as a new column
activ_df['Time_prop_cit'] = activ_df['Citation_Diff'] * 365 / rep_elapsed_days

# Define the conditions for the 'State' column
conditions = [
    (activ_df['Paper_age'] < 1.5) & (activ_df['Total_num_cit'] == 0),  # Condition for "Retention"
    (activ_df['Time_prop_cit'] > activ_df['Avg_annual_cit']),         # Condition for "Active"
    (activ_df['Time_prop_cit'] <= activ_df['Avg_annual_cit'])         # Condition for "Post"
]

# Define the corresponding values for each condition
values = ['I. Retention period', 'II. Active contribution period', 'III. Post contribution period']

# Apply the conditions and create the new 'State' column
activ_df['State'] = np.select(conditions, values)

# Step 1: Create the Plotly table with conditional fill in 'State' column
def create_table(activ_df):
    # Define color conditions for the 'State' column
    fill_colors = np.where(
        activ_df['State'] == 'I. Retention period', '#E7E6E6',
        np.where(
            activ_df['State'] == 'II. Active contribution period', '#E2EFDA',
            '#FCE4D6'  # For 'III. Post contribution period'
        )
    )
    
    # Create table
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=['Title', 'Year', 'State'],
                    fill_color='white',  # No fill for header
                    align='left'),
        cells=dict(
            values=[activ_df['Title'], activ_df['Year'], activ_df['State']],  # Table content
            fill_color=['white', 'white', fill_colors],              # Conditional fill only for 'State' column
            align='left'
        )
    )])

# Convert the table figure to HTML
    table_html = table_fig.to_html(full_html=False, include_plotlyjs=False)
    return table_html


retention_pap = activ_df['State'].value_counts().get('I. Retention period', 0)
active_pap = activ_df['State'].value_counts().get('II. Active contribution period', 0)
post_pap = activ_df['State'].value_counts().get('III. Post contribution period', 0)

active_pap_rat = round(((active_pap)/num_papers)*100, 1)
  
def create_html_dashboard(line_plot_fig, monthly_citations_bar_plot, quartile_bar_plot_fig, IF_per_year_bar_fig, h_index_fig, journal_pie_fig, coauthor_network_fig, cit_arima_fig, header_content, report_info, total_citations, academic_age, merged_publisher_bar_plot_fig, activ_df):
    # Save Plotly plot as an HTML string
    line_plot_html = line_plot_fig.to_html(full_html=False, include_plotlyjs='cdn')
    #bar_plot_html = bar_plot_fig.to_html(full_html=False, include_plotlyjs=False)
    monthly_citations_bar_plot_html = monthly_citations_bar_plot.to_html(full_html=False, include_plotlyjs=False)
    quartile_bar_plot_html = quartile_bar_plot_fig.to_html(full_html=False, include_plotlyjs=False)
    IF_per_year_bar_fig_html = IF_per_year_bar_fig.to_html(full_html=False, include_plotlyjs=False)
    h_index_html = h_index_fig.to_html(full_html=False, include_plotlyjs=False)
    pie_chart_html = journal_pie_fig.to_html(full_html=False, include_plotlyjs=False)
    coauthor_network_html = coauthor_network_fig.to_html(full_html=False, include_plotlyjs=False)
    cit_arima_fig_html = cit_arima_fig.to_html(full_html=False, include_plotlyjs=False)
    merged_publisher_bar_plot_html = merged_publisher_bar_plot_fig.to_html(full_html=False, include_plotlyjs=False)
    table_html = create_table(activ_df)

 # Extract author information
    author_name = author.get("name", "Unknown Author")
    author_affiliation = author.get("affiliation", "Unknown Affiliation")
    image_path = author.get("url_picture", "https://via.placeholder.com/150")  # Placeholder if image URL is missing


    # Basic HTML structure with header, report info, total citations, and plot
    html_content = f"""
    <html>
    <head>
        <title>Scientific achievements of {author_name}</title>
    </head>
    <body>
       <header style="display: flex; align-items: center;">
            <div style="margin-right: 20px;">
                <img src="{image_path}" alt="Author's Photo" style="max-width: 150px; height: auto;" />
            </div>
            <div>
                <h1>Scientific Achievements of {author_name}</h1>
                <p><strong>Affiliation:</strong> {author_affiliation}</p>
                <p>{header_content}</p>
            </div>
        </header>

        <section>
            <h2>Report Information</h2>
            <p><strong>The date of the report:</strong> {report_info['date_of_report']}</p>
            <p><strong>The length of the observed period:</strong> {report_info['observed_period'].days} days</p>
        </section>

        <section>
            <h2>Static scientometrics of the author (indicators to describe the entire academic career)</h2>
            <p><strong>Academic age:</strong> {academic_age} years</p> <!-- Added academic age here -->        
            <p><strong>Total number of publications:</strong> {num_papers}</p>
            <p><strong>Total number of citations:</strong> {total_citations}</p>
            <p><strong>Average number of citations per publication:</strong> {avg_cit}</p>
            <p><strong>Annual average number of citations:</strong> {avg_ann_cit}</p>
            <p><strong>Annual average number of publications:</strong> {avg_ann_pap}</p>
            <p><strong>Ratio of the uncited publications:</strong> {zero_citt_ratio} %</p>
            <p><strong>Number of single-authored publications:</strong> {single_aut}</p>
            <p><strong>Number of first-authored publications (without single-authored publications):</strong> {first_aut_count}</p>
            <p><strong>Number of supervisor-authored publications (last author position without single-authored publications):</strong> {supervised_aut_count}</p>
            <p><strong>Proportion of publications with a distinguished position (first or last authored publications without single-authored publications):</strong> {disting_pos} %</p>
            <p><strong>Number of Q1 papers (D1 papers are counted here as well):</strong> {num_Q1_papers}</p>   
            <p><strong>Number of D1 papers:</strong> {num_D1_papers}</p> 
            <p><strong>Rough estimated cumulated impact factor (based on Scimago database):</strong> {cumulated_if}</p>
            <p><strong>Average impact factor of publications:</strong> {avg_IF_per_paper}</p>
            <p><strong>Highest impact factor paper:</strong> {max_IF_paper_title}: {max_IF_paper_IF}</p>
            <p><strong>Average number of authors in publications:</strong> {average_coauthors}</p>
            <p><strong>Number of co-authors:</strong> {Num_coauthors}</p>
        </section>    

        <section>
            <h2>Dynamic scientometrics of the author (indicators to describe the observation pediod)</h2>
            <p><strong>Citations:</strong> {difference}</p>
            <p><strong>Average daily number of citations:</strong> {avg_day_cit}</p>
            <p><strong>Most cited paper:</strong> {most_improved_paper_title}:  <strong>{max_citation_diff}</strong> citations</p>
            <p><strong>The highest number of citations in one day in the observed period (based on queried dates):</strong> {max_change}</p>
            <p><strong>Ratio of the active publications:</strong> {active_pap_rat} %</p>
        </section>  
         
        <section>
            <h2>Let's take a look at the evolution of citations to scientific publications over time</h2>
            {line_plot_html}
        </section>
        
        <section>
            <h2>Monthly distribution of citations</h2>
            {monthly_citations_bar_plot_html}
        </section>
        
        <section>
            <h2>Number of Publications per quartile by year based on Scimago database</h2> <!-- Add the Quartile Bar Plot -->
            {quartile_bar_plot_html}
        </section>
        
        <section>
            <h2>Rough estimated annual cumulated impact factors based on Scimago database</h2> 
            {IF_per_year_bar_fig_html}
        </section>
        
        <section>
            <h2>h-index Graph</h2> <!-- Added H-index section -->
            {h_index_html}
        </section>
        
        <section>
            <h2>Distribution of Papers in Different Journals</h2>
            {pie_chart_html}
        </section>
        
        <section>
            <h2>Co-authorship Network</h2>
            {coauthor_network_html}
        </section>
        
        <section>
            <h2>Autoregressive Integrated Moving Average (ARIMA) prediction of expected citations</h2>
            {cit_arima_fig_html}
        </section>
        
        <section>
            <h2>Distribution of published papers by Publishers</h2>
            {merged_publisher_bar_plot_html}
        </section>
        
        <section>
            <h2>Characterization of the life stages of publications</h2>
            <p><strong>Number of active contribution phase papers:</strong> {active_pap}</p>
            <p><strong>Number of retention phase papers:</strong> {retention_pap}</p>
            <p><strong>Number of post contribution phase papers:</strong> {post_pap}</p>
            {table_html}
        </section>
        
    </body>
    </html>
    """

    # Save the HTML content to a file
    with open("dashboard.html", "w") as file:
        file.write(html_content)
    print("Dashboard HTML file created!")

# Example usage:
if __name__ == "__main__":
    # Step 1: Create the Plotly line plot
    line_plot_fig = create_line_plot()
    
    # Step 2: Extract the report information
    #bar_plot_fig = create_publications_bar_plot(publications_per_year)
    
    # Step 3: Create monthly citation bar plot
    monthly_citations_bar_plot = create_monthly_citations_bar_plot(total_cit_df)
    
    # Step 3: Create the quartile bar plot
    quartile_bar_plot_fig = create_quartile_bar_plot(grouped_data)
    
    IF_per_year_bar_fig = create_IF_per_year_bar_plot(annual_IF_sum)
    
    # Step 4: Extract the report information
    report_info = get_report_info()
    
    # Step 5: Create H-index plot
    h_index_fig = create_h_index_plot(last_column)
    
    # Step 6: Create the journal distribution pie chart
    journal_pie_fig = create_journal_pie_chart(journal_df)
    
    # Step 7: Get the total number of citations
    total_citations = author['citedby']
    
    # Step 8: Create coauthor network plot
    coauthor_network_fig = create_coauthor_network_plot(coauthor_graph)
    
    # Step 8: Create predictive plot
    cit_arima_fig = create_citations_forecast_chart(daily_data, forecast_df)
    
    # Step 8: Create predictive plot
    merged_publisher_bar_plot_fig = create_merged_publisher_bar_plot(publisher_count)
    
    # Step 9: Extracted information to place in the header
    extracted_info = "This analytical tool is used to monitor the scientific performance of researchers based on Google Scholar & Scimago databases."
    
    # Step 10: Create the HTML dashboard with the plot, report info, and total citations
    # If you want to add the simple barplot use this
    #create_html_dashboard(line_plot_fig, bar_plot_fig, quartile_bar_plot_fig, h_index_fig, journal_pie_fig, extracted_info, report_info, total_citations, academic_age)
    create_html_dashboard(line_plot_fig, monthly_citations_bar_plot, quartile_bar_plot_fig, IF_per_year_bar_fig, h_index_fig, journal_pie_fig, coauthor_network_fig, cit_arima_fig, extracted_info, report_info, total_citations, academic_age, merged_publisher_bar_plot_fig, activ_df)

#%%
end_time = time.time()
run_time = end_time - start_time  # Elapsed time in seconds
print(f"Run time: {run_time} second")

#%% Tovabbi fejlesztesek amiket meg akarok csinalni

# ACTIV_DF EL VAN RONTVA MERT AZ OSSZEFUZENDO TETELEK MAS SORRENDBEN VANNAK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Impack faktorok meghatarozasa (durvan megvan a Scimgao adatbazisabol)
# Cikkenkenti hivatkozasok exportalasa egy excel fajlba
# Uj hivatkozasok automatikus detektalasa
# Monthly citations bar chart with cumulated line citation line (A kumulalt vonaldiagramot eltekintve megvan)

# Alex javaslatai okt 25 témabeszámoló alapján
# Levenshtein distance alapon nézni a kiadók hasonlóságát és merge-olni őket.

# Sajat exceles mutatoim implementalasa:
# Academic age based on the first authored paper
# Hány cikkére kell hivatkozást kapjon, hogy a h-indexe 1-el nőjön
# A jelentés tárgyévében kapott hivatkozások száma
# Két független hivatkozás között eltelt napok átlagos száma a teljes időszakban
# Két független hivatkozás között eltelt napok átlagos száma a jelentés tárgyévében
# A cikkek száma és a független hivatkozások száma közötti összefüggés erőssége
# Olyan cikkek száma, amiben két cikkére is hivatkoztak
# Olyan cikkek száma, amiben három cikkére is hivatkoztak
# A Pannon Egyetem kutatói között hányadik helyen szerepel a Google Scholar rangsorában
# A Pannon Egyetem kutatói között elfoglalt relatív hely alapján besorolás a Google Scholar rangsorában
# D1 cikkek átlagos helyezése az összes cikk rangsorában
# Q1 cikkek átlagos helyezése az összes cikk rangsorában
# Q2 cikkek átlagos helyezése az összes cikk rangsorában
# Q3 cikkek átlagos helyezése az összes cikk rangsorában
# Q4 cikkek átlagos helyezése az összes cikk rangsorában
# Könyvfejezetek cikkek átlagos helyezése az összes cikk rangsorában
# A legrosszabb cikk (hivatkozások éves száma alapján)
# A legjobb cikk (hivatkozások éves száma alapján)
# A legjobb újság impakt faktort növelő cikk (az újság impakt faktorához képest kapott időarányos hivatkozások száma alapján)
