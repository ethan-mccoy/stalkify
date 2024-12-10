# Creates the widgets for the dashboard

import panel as pn

pn.extension("plotly")

# Search widgets for filtering data
artist_search_box = pn.widgets.TextInput(name='Search Artist', placeholder='Enter artist name')
album_search_box = pn.widgets.TextInput(name='Search Album', placeholder='Enter album name')
track_search_box = pn.widgets.TextInput(name='Search Track', placeholder='Enter track name')
lyrics_search_box = pn.widgets.TextInput(name='Search Lyrics', placeholder='Enter word in lyrics')

# MultiSelect and slider for emotion-based filtering
emotion_multiselect = pn.widgets.MultiSelect(
    name='Select Emotions', 
    options=[
        'GPT_Happy', 'GPT_Sad', 'GPT_Fear', 'GPT_Anger',
        'GPT_Hope', 'GPT_Happy_Love', 'GPT_Sad_Love', 'GPT_Growth',
        'GPT_Spirituality', 'GPT_Materialism', 'GPT_Emotion',
        'GPT_Humor', 'GPT_Ambition', 'GPT_Empowerment',
        'GPT_Self_Reflection', 'GPT_Creativity', 'GPT_Optimism',
        'GPT_Pessimism', 'GPT_Loneliness', 'GPT_Intensity',
        'GPT_Metaphorical'
    ],
    value=[],  # Default 
    size=10
)
emotion_magnitude_slider = pn.widgets.IntSlider(
    name='Emotion Magnitude', 
    start=0, 
    end=10, 
    step=1, 
    value=0
)

# Y-Axis Maximum Slider
yaxis_max_slider = pn.widgets.IntSlider(
    name='Y-Axis Maximum', 
    start=20, 
    end=400,  
    step=20, 
    value=300  
)

# Cluster Selection Slider
cluster_slider = pn.widgets.IntSlider(name='Select Cluster', start=0, end=10, step=1, value=0)

# Export all widgets
__all__ = [
    'artist_search_box',
    'album_search_box',
    'track_search_box',
    'lyrics_search_box',
    'emotion_multiselect',
    'emotion_magnitude_slider',
    'yaxis_max_slider',
    'cluster_slider'
] 