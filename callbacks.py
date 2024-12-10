# This file contains the callbacks for the dashboard to update the plots.

from plots import (
    create_overall_emotion_distribution,
    create_violin_plot,
    create_gpt_emotions_plot,
    create_cluster_bar_chart,
    get_cluster_track_list
)

def update_bar_chart(event, widgets, df, bar_chart_panels):
    filtered = get_filtered_df(event, widgets, df)
    
    selected_emotions = widgets['emotion_multiselect'].value
    magnitude_threshold = widgets['emotion_magnitude_slider'].value
    y_max = widgets['yaxis_max_slider'].value 

    # Create the appropriate bar charts with updated parameters
    try:
        fig_2023 = create_overall_emotion_distribution(
            filtered, 
            selected_emotions, 
            magnitude_threshold, 
            y_max,
            2023
        )
        bar_chart_panels['2023'].object = fig_2023
    except:
        print('uh oh something break in update_bar_chart')

    try:
        fig_2024 = create_overall_emotion_distribution(
            filtered, 
            selected_emotions, 
            magnitude_threshold, 
            y_max,
            2024
        )
        bar_chart_panels['2024'].object = fig_2024
    except:
        print('uh oh something break in update_bar_chart')

        

def update_cluster_info(event, widgets, cluster_df, cluster_bar_chart_panel, track_list_area, y_axis_range):
    """
    Callback to update the cluster bar chart and track list when a new cluster is selected.
    """
    cluster_id = event.new
    cluster_bar_chart_panel.object = create_cluster_bar_chart(cluster_df, cluster_id, y_axis_range)
    track_list_area.value = get_cluster_track_list(cluster_df, cluster_id)

def update_violin_plot(event, widgets, df, violin_panel):
    """
    Callback to update the violin plot when search filters change.
    """
    filtered = get_filtered_df(event, widgets, df)
    violin_panel.object = create_violin_plot(filtered)

def update_gpt_emotions_plot(event, widgets, df, gpt_emotions_panel):
    """
    Callback to update the GPT emotions plot when search filters change.
    Dynamically determines the grouping based on active search filters.
    """
    filtered = get_filtered_df(event, widgets, df)
    
    # Determine the grouping column based on active search filters
    if widgets['artist_search_box'].value:
        group_by = 'artist_names'
    elif widgets['album_search_box'].value:
        group_by = 'album_name'
    elif widgets['track_search_box'].value:
        group_by = 'track_name'
    else:
        # Default grouping
        group_by = 'artist_names'
    
    gpt_emotions_panel.object = create_gpt_emotions_plot(filtered, group_by)

def get_filtered_df(event, widgets, df):
    filtered = df.copy()

    # Apply search filters
    if widgets['artist_search_box'].value:
        filtered = filtered[filtered['artist_names'].str.lower().str.contains(widgets['artist_search_box'].value.lower())]

    if widgets['album_search_box'].value:
        filtered = filtered[filtered['album_name'].str.lower().str.contains(widgets['album_search_box'].value.lower())]

    if widgets['track_search_box'].value:
        filtered = filtered[filtered['track_name'].str.lower().str.contains(widgets['track_search_box'].value.lower())]

    if widgets['lyrics_search_box'].value:
        filtered = filtered[filtered['lyrics'].str.lower().str.contains(widgets['lyrics_search_box'].value.lower(), na=False)]

    return filtered 