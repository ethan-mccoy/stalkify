# Main file for the visualization dashboard

import panel as pn
import pandas as pd
from widgets import (
    artist_search_box,
    album_search_box,
    track_search_box,
    lyrics_search_box,
    emotion_multiselect,
    emotion_magnitude_slider,
    yaxis_max_slider,
    cluster_slider
)
from plots import (
    create_overall_emotion_distribution,
    create_violin_plot,
    create_gpt_emotions_plot,
    create_cluster_bar_chart,
    get_cluster_track_list,
    get_emotion_track_list
)
from clustering import perform_clustering
from callbacks import (
    update_bar_chart,
    update_cluster_info,
    update_violin_plot,
    update_gpt_emotions_plot
)

df = pd.read_csv('data/merged_cleaned.csv')
print("DataFrame Columns:", df.columns)

# Clustering
cluster_df = perform_clustering(df)
y_axis_range = [0, yaxis_max_slider.value] 

# Initialize plots
initial_fig_2023 = create_overall_emotion_distribution(df, [], 0, y_max=yaxis_max_slider.value, year = 2023)
initial_fig_2024 = create_overall_emotion_distribution(df, [], 0, y_max=yaxis_max_slider.value, year = 2024)
initial_violin_fig = create_violin_plot(df)
initial_cluster_fig = create_cluster_bar_chart(cluster_df, cluster_slider.value, y_axis_range)
initial_gpt_emotions_fig = create_gpt_emotions_plot(df, 'artist_names')

bar_chart_panel_2023 = pn.pane.Plotly(initial_fig_2023, height=400)
bar_chart_panel_2024 = pn.pane.Plotly(initial_fig_2024, height=400)
violin_panel = pn.pane.Plotly(initial_violin_fig, height=400)
gpt_emotions_panel = pn.pane.Plotly(initial_gpt_emotions_fig, height=400)
cluster_bar_chart_panel = pn.pane.Plotly(initial_cluster_fig, height=400)

# Tracklist next to some plots
track_list_area = pn.widgets.TextAreaInput(value=get_cluster_track_list(cluster_df, cluster_slider.value), height=400, width=300, disabled=True)
emotion_track_list_area = pn.widgets.TextAreaInput(value='', height=400, width=300, disabled=True)

# Column for the search widgets in sidebar
search_sidebar = pn.Column(
    artist_search_box,
    album_search_box,
    track_search_box,
    lyrics_search_box,
    emotion_multiselect,
    emotion_magnitude_slider,
    yaxis_max_slider,
    cluster_slider
)

# Dictionary of widgets for callbacks
widgets = {
    'artist_search_box': artist_search_box,
    'album_search_box': album_search_box,
    'track_search_box': track_search_box,
    'lyrics_search_box': lyrics_search_box,
    'emotion_multiselect': emotion_multiselect,
    'emotion_magnitude_slider': emotion_magnitude_slider,
    'yaxis_max_slider': yaxis_max_slider,
    'cluster_slider': cluster_slider
}

bar_chart_panels = {
    '2023': bar_chart_panel_2023,
    '2024': bar_chart_panel_2024
}

# Set Up Observers with lambda to pass additional parameters
emotion_multiselect.param.watch(
    lambda event: update_bar_chart(event, widgets, df, bar_chart_panels), 
    'value'
)
emotion_magnitude_slider.param.watch(
    lambda event: update_bar_chart(event, widgets, df, bar_chart_panels), 
    'value'
)
yaxis_max_slider.param.watch(
    lambda event: update_bar_chart(event, widgets, df, bar_chart_panels), 
    'value'
)

# Watch for changes in the sidebar
for widget_name in ['artist_search_box', 'album_search_box', 'track_search_box', 'lyrics_search_box']:
    widgets[widget_name].param.watch(
        lambda event: update_bar_chart(event, widgets, df, bar_chart_panels), 
        'value'
    )
    widgets[widget_name].param.watch(
        lambda event: update_violin_plot(event, widgets, df, violin_panel), 
        'value'
    )
    widgets[widget_name].param.watch(
        lambda event: update_gpt_emotions_plot(event, widgets, df, gpt_emotions_panel), 
        'value'
    )

cluster_slider.param.watch(
    lambda event: update_cluster_info(event, widgets, cluster_df, cluster_bar_chart_panel, track_list_area, y_axis_range),
    'value'
)

# Watch for changes in the GPT Emotions plot to update the emotion_track_list_area
def update_emotion_track_list(event):
    selected_group = widgets['album_search_box'].value  
    if selected_group:
        filtered_df = df[df['album_name'] == selected_group]
        track_list = get_emotion_track_list(filtered_df, group_by='album_name')
        emotion_track_list_area.value = track_list
    else:
        emotion_track_list_area.value = ''

widgets['album_search_box'].param.watch(
    lambda event: [
        update_gpt_emotions_plot(event, widgets, df, gpt_emotions_panel),
        update_emotion_track_list(event)
    ],
    'value'
)

# Layout 
template = pn.template.FastListTemplate(
    title="Stalkify",
    theme='dark',
    header_background='#1DB954',
    sidebar=[search_sidebar],  
    main=[
        pn.Row(bar_chart_panel_2023, bar_chart_panel_2024),  
        pn.Row(gpt_emotions_panel, emotion_track_list_area),  
        pn.Row(violin_panel),  
        pn.Row(cluster_bar_chart_panel, track_list_area)
    ]  
)

template.servable() 