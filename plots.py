# Creates the plots for the dashboard

import plotly.express as px
import pandas as pd

# Plots the emotions for a given year
def create_yearly_plot(df, year, selected_emotions, magnitude_threshold, y_max):
    # Placeholder for empty df
    if df.empty:
        fig = px.bar(
            title=f'No data available for {year}',
            x=[],  
            y=[],  
            text=[]
        )
        fig.update_layout(
            xaxis_title='',
            yaxis_title='',
            plot_bgcolor='rgba(30, 215, 96, 0.1)',
            paper_bgcolor='rgba(30, 215, 96, 0.1)',
            font=dict(
                family="Arial, sans-serif",
                size=14,
                color="white"
            )
        )
        return fig

    emotion_columns = [col for col in df.columns if col.startswith('GPT_')]
    df[emotion_columns] = df[emotion_columns].apply(pd.to_numeric, errors='coerce')
    df[emotion_columns] = df[emotion_columns].fillna(0)

    # Calculate total emotion intensity per month
    total_emotion = df.groupby('month_added')[emotion_columns].sum()
    overall_sum = total_emotion.sum(axis=1)

    # Sort by calendar month order
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]

    if selected_emotions:
        percentage_df = pd.DataFrame()
        count_df = pd.DataFrame()

        for emotion in selected_emotions:
            # Calculate emotion intensity above the threshold
            filtered_emotion = df[df[emotion] >= magnitude_threshold]
            emotion_sum = filtered_emotion.groupby('month_added')[emotion].sum()
            song_count = filtered_emotion.groupby('month_added')[emotion].count()

            # Calculate percentage
            percentage = (emotion_sum / overall_sum) * 100
            percentage = percentage.reset_index()
            percentage.columns = ['month_added', emotion]

            # Calculate song counts
            counts = song_count.reset_index()
            counts.columns = ['month_added', f'{emotion}_count']

            # Merge percentage and counts
            if percentage_df.empty:
                percentage_df = percentage
                count_df = counts
            else:
                percentage_df = pd.merge(percentage_df, percentage, on='month_added', how='outer')
                count_df = pd.merge(count_df, counts, on='month_added', how='outer')

        # Replace NaN with 0
        percentage_df = percentage_df.fillna(0)
        count_df = count_df.fillna(0)

        # Sort by calendar month order
        percentage_df['month_added'] = pd.Categorical(
            percentage_df['month_added'],
            categories=month_order,
            ordered=True
        )
        percentage_df = percentage_df.sort_values('month_added')

        count_df['month_added'] = pd.Categorical(
            count_df['month_added'],
            categories=month_order,
            ordered=True
        )
        count_df = count_df.sort_values('month_added')

        # Add all months to the DataFrames
        all_months = pd.DataFrame({'month_added': month_order})
        percentage_df = pd.merge(all_months, percentage_df, on='month_added', how='left').fillna(0)
        count_df = pd.merge(all_months, count_df, on='month_added', how='left').fillna(0)

        # Melt the DataFrames for Plotly express
        percentage_melted = percentage_df.melt(id_vars='month_added', var_name='Emotion', value_name='Percentage')
        count_melted = count_df.melt(id_vars='month_added', var_name='Emotion_Count', value_name='Count')

        # Merge percentage and count melted DataFrames
        merged_df = pd.merge(
            percentage_melted,
            count_melted,
            on='month_added', 
            how='left'
        )

        merged_df['Emotion'] = merged_df['Emotion_Count'].str.replace('_count', '')
        merged_df['text'] = merged_df.apply(lambda row: f"{row['Percentage']:.1f}%\n({int(row['Count'])})", axis=1)

        # Create grouped bar chart
        fig = px.bar(
            merged_df,
            x='month_added',
            y='Percentage',
            color='Emotion',
            barmode='group',
            text='text',
            title=f'Percent of Selected Emotions with Magnitude > {magnitude_threshold} by Month ({year})',
            labels={'Count': 'Count (songs)', 'month_added': 'Month'},
            color_discrete_sequence=px.colors.qualitative.Vivid
        )

        fig.update_layout(
            yaxis_title='Count (songs)',
            xaxis_title='Month',
            yaxis=dict(range=[0, 10]), 
            plot_bgcolor='rgba(30, 215, 96, 0.1)',
            paper_bgcolor='rgba(30, 215, 96, 0.1)',
            font=dict(
                family="Arial, sans-serif",
                size=14,
                color="white"
            )
        )
        fig.update_traces(textposition='inside', textfont=dict(color='black'))

    else:
        # When no emotions are selected, display overall emotion distribution
        monthly_counts = total_emotion.reset_index()
        monthly_counts['month_added'] = pd.Categorical(
            monthly_counts['month_added'],
            categories=[
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ],
            ordered=True
        )
        monthly_counts = monthly_counts.sort_values('month_added')

        # Ensure all months are present in the DataFrame
        all_months = pd.DataFrame({'month_added': month_order})
        monthly_counts = pd.merge(all_months, monthly_counts, on='month_added', how='left').fillna(0)

        # Melt the DataFrame to long-form for Plotly
        melted_df = monthly_counts.melt(
            id_vars='month_added',
            value_vars=emotion_columns,
            var_name='Emotion',
            value_name='Emotion_Intensity'
        )

        # Create stacked bar chart
        fig = px.bar(
            melted_df,
            x='month_added',
            y='Emotion_Intensity',
            color='Emotion',
            barmode='stack',
            title=f'Overall Emotion Distribution by Month ({year})',
            labels={'Emotion_Intensity': 'Emotion Intensity', 'month_added': 'Month'},
            color_discrete_sequence=px.colors.qualitative.Vivid
        )

        fig.update_layout(
            yaxis_title='Emotion Intensity',
            xaxis_title='Month',
            yaxis=dict(range=[0, y_max]),  
            plot_bgcolor='rgba(30, 215, 96, 0.1)',
            paper_bgcolor='rgba(30, 215, 96, 0.1)',
            font=dict(
                family="Arial, sans-serif",
                size=14,
                color="white"
            )
        )

    return fig

def create_overall_emotion_distribution(filtered_df, selected_emotions, magnitude_threshold, y_max, year):
    if not pd.api.types.is_datetime64_any_dtype(filtered_df['added_at']):
        filtered_df['added_at'] = pd.to_datetime(filtered_df['added_at'], errors='coerce')

    filtered_df['month_added'] = filtered_df['added_at'].dt.month_name()
    filtered_df['year_added'] = filtered_df['added_at'].dt.year

    year_df = filtered_df[filtered_df['year_added'] == year]
    fig = create_yearly_plot(year_df, year, selected_emotions, magnitude_threshold, y_max)

    return fig

def create_bar_chart_by_month(filtered_df, selected_emotions, magnitude_threshold):
    return create_overall_emotion_distribution(filtered_df, selected_emotions, magnitude_threshold)


def create_violin_plot(filtered_df):
    filtered_df['weekday_added'] = pd.Categorical(
        filtered_df['weekday_added'], 
        categories=[
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ], 
        ordered=True
    )

    fig = px.violin(
        filtered_df, 
        y='hour_added', 
        x='weekday_added', 
        color='weekday_added', 
        box=True, 
        points='all', 
        title='Violin Plot of Hour Added by Weekday'
    )

    fig.update_traces(hovertemplate=(
        "Hour Added: %{y}<br>"
        "Weekday: %{x}<br>"
        "Track: %{customdata[0]}<br>"
        "Artist: %{customdata[1]}<br>"
        "<extra></extra>"
    ))

    # Add custom data for track name and artist names
    fig.update_traces(customdata=filtered_df[['track_name', 'artist_names']].to_numpy())

    fig.update_layout(
        xaxis=dict(
            categoryorder='array', 
            categoryarray=[
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ]
        ),
        yaxis_title='Hour Added',
        xaxis_title='Weekday',
        plot_bgcolor='rgba(30, 215, 96, 0.1)',  
        paper_bgcolor='rgba(30, 215, 96, 0.1)',   
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="white"
        )
    )
    return fig

def create_cluster_bar_chart(cluster_df, cluster_id, y_axis_range=None):
    selected_cluster_df = cluster_df[cluster_df['cluster'] == cluster_id]
    
    # Calculate the mean of each feature for the cluster
    cluster_means = selected_cluster_df.filter(regex='^GPT_(?!.*_explanation$)').mean().reset_index().round(1)
    cluster_means.columns = ['Feature', 'Mean']
    
    # Calculate the overall mean of each feature across all clusters
    overall_means = cluster_df.filter(regex='^GPT_(?!.*_explanation$)').mean().reset_index().round(1)
    overall_means.columns = ['Feature', 'Overall Mean']
    
    # Bar chart for cluster means
    fig = px.bar(
        cluster_means, x='Feature', y='Mean',
        title=f'Feature Means for Cluster {cluster_id}',
        labels={'Mean': 'Average Value'},
        text='Mean'
    )
    
    # Points of overall means
    fig.add_scatter(
        x=overall_means['Feature'],
        y=overall_means['Overall Mean'],
        mode='markers',
        name='Overall Mean',
        marker=dict(size=8, color='rgba(255, 0, 0, 0.5)') 
    )
    
    # Update layout for aesthetics
    fig.update_layout(
        xaxis_title='Feature',
        yaxis_title='Average Value',
        plot_bgcolor='rgba(30, 215, 96, 0.1)', 
        paper_bgcolor='rgba(30, 215, 96, 0.1)',
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="white"
        )
    )
    
    return fig

def get_cluster_track_list(cluster_df, cluster_id):
    selected_cluster_df = cluster_df[cluster_df['cluster'] == cluster_id]
    track_list = selected_cluster_df['track_name'].tolist()
    track_list_str = "\n".join(track_list)
    
    return track_list_str

# Get track list for a specific group in the emotions plot
def get_emotion_track_list(filtered_df, group_by):
    if group_by == 'album_name':
        track_list = filtered_df['track_name'].unique().tolist()
    else:
        track_list = filtered_df['track_name'].tolist()
    track_list_str = "\n".join(track_list)
    return track_list_str

def create_gpt_emotions_plot(df, group_by):
    """
    Creates a plot to visualize GPT emotions grouped by a specified column.
    Automatically adjusts grouping based on the provided group_by parameter.
    """
    emotion_columns = [col for col in df.columns if col.startswith('GPT_') and pd.api.types.is_numeric_dtype(df[col])]
    
    # Modify grouping to include 'track_name' if grouping by 'album_name'
    if group_by == 'album_name':
        grouped_df = df.groupby([group_by, 'track_name'])[emotion_columns].mean().reset_index()
    else:
        grouped_df = df.groupby(group_by)[emotion_columns].mean().reset_index()

    # Update the Plotly express bar chart to include 'track_name'
    if group_by == 'album_name':
        fig = px.bar(
            grouped_df,
            x=group_by,
            y=emotion_columns,
            title=f'GPT Emotions by {group_by.replace("_", " ").title()} and Track',
            barmode='group'
        )
        fig.update_traces(hovertemplate=(
            "Emotion: %{fullData.name}<br>"
            "Emotion Value: %{y}<br>"
            "Album: %{x}<br>"
            "Track: %{customdata[0]}<br>"
            "Artist: %{customdata[1]}<br>"
            "<extra></extra>"
        ))

        # Add custom data for track name and artist names
        fig.update_traces(customdata=df[['track_name', 'artist_names']].to_numpy())


    else:
        fig = px.bar(
            grouped_df,
            x=group_by,
            y=emotion_columns,
            title=f'Average GPT Emotions by {group_by.replace("_", " ").title()}',
            labels={'value': 'Average Emotion Intensity', group_by: group_by.replace("_", " ").title()},
            barmode='group'
        )

    fig.update_layout(
        yaxis_title='Average Emotion Intensity',
        xaxis_title=group_by.replace("_", " ").title(),
        plot_bgcolor='rgba(30, 215, 96, 0.1)',  
        paper_bgcolor='rgba(30, 215, 96, 0.1)', 
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="white"
        )
    )
    return fig