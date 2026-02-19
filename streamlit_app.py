import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP dashboard',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from pathlib import Path
    from math import radians, sin, cos, asin, sqrt


    st.set_page_config(page_title='Taxi Dashboard', page_icon='🚕', layout='wide')


    @st.cache_data
    def load_data(uploaded_file):
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            # No bundled taxi data in this workspace; return None so user can upload
            return None
        return df


    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 6371 * 2 * asin(sqrt(a))


    st.title('Taxi Data Explorer')

    st.markdown('Upload a NYC taxi CSV (columns like pickup/dropoff lat/lon, fare_amount, tpep_pickup_datetime, tpep_dropoff_datetime).')

    uploaded = st.file_uploader('Upload taxi CSV', type=['csv'])
    df = load_data(uploaded)

    if df is None:
        st.info('Please upload a taxi CSV to enable the dashboard. I can compute distances, durations, Monte Carlo fare simulation, and maps once data is provided.')
        st.stop()

    with st.spinner('Preparing data...'):
        # basic cleaning
        df = df.copy()
        # convert datetimes if present
        for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # compute duration
        if 'tpep_pickup_datetime' in df.columns and 'tpep_dropoff_datetime' in df.columns:
            df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

        # compute haversine distance
        if all(c in df.columns for c in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']):
            df['distance_km'] = df.apply(lambda r: haversine(r['pickup_longitude'], r['pickup_latitude'], r['dropoff_longitude'], r['dropoff_latitude']), axis=1)

        # sample for performance
        sample_n = min(50000, len(df))
        df_sample = df.sample(sample_n, random_state=42)

    st.subheader('Data preview')
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Trip distance vs Fare')
        if 'trip_distance' in df.columns:
            fig = px.scatter(df_sample, x='trip_distance', y='fare_amount', opacity=0.5, title='Trip Distance vs Fare')
        elif 'distance_km' in df.columns:
            fig = px.scatter(df_sample, x='distance_km', y='fare_amount', opacity=0.5, title='Distance (km) vs Fare')
        else:
            fig = None
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader('Trip duration vs Fare')
        if 'trip_duration' in df.columns:
            fig2 = px.scatter(df_sample, x='trip_duration', y='fare_amount', opacity=0.5, title='Trip Duration (min) vs Fare')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write('No trip duration column found')

    st.markdown('---')

    st.subheader('Monte Carlo Fare Simulation')
    with st.spinner('Running Monte Carlo...'):
        # derive simple fare stats
        if 'fare_amount' in df.columns:
            # use computed distance if present else fall back to trip_distance
            if 'distance_km' in df.columns:
                dist_col = 'distance_km'
            elif 'trip_distance' in df.columns:
                dist_col = 'trip_distance'
            else:
                dist_col = None

            fare_per_km = (df['fare_amount'] / df[dist_col]).median() if dist_col else df['fare_amount'].median()
            base_fare = df['fare_amount'].quantile(0.05)
            fare_std = df['fare_amount'].std()

            N_PATHS = st.sidebar.slider('Paths', 100, 1000, 300)
            N_STEPS = st.sidebar.slider('Steps', 50, 200, 100)
            max_distance = st.sidebar.slider('Max distance (km)', 5, 50, 20)

            DT = max_distance / N_STEPS
            np.random.seed(42)

            paths = np.zeros((N_PATHS, N_STEPS + 1))
            paths[:,0] = base_fare
            for step in range(N_STEPS):
                drift = (fare_per_km if dist_col else (df['fare_amount'].mean() / max_distance)) * DT
                volatility = (fare_std * 0.15) * np.sqrt(DT)
                shock = np.random.normal(0, volatility, N_PATHS)
                paths[:, step + 1] = np.maximum(paths[:, step] + drift + shock, base_fare)

            distances = np.linspace(0, max_distance, N_STEPS + 1)
            mean_path = paths.mean(axis=0)
            p10 = np.percentile(paths, 10, axis=0)
            p90 = np.percentile(paths, 90, axis=0)

            mc_fig = go.Figure()
            # lighter plot: only mean + bands
            mc_fig.add_trace(go.Scatter(x=distances, y=mean_path, line=dict(color='#FFE135', width=3), name='Mean'))
            mc_fig.add_trace(go.Scatter(x=np.concatenate([distances, distances[::-1]]), y=np.concatenate([p90, p10[::-1]]), fill='toself', fillcolor='rgba(0,200,255,0.08)', line=dict(color='rgba(0,0,0,0)'), name='P10-P90'))
            mc_fig.update_layout(title='Monte Carlo Fare Simulation', xaxis_title='Distance (km)', yaxis_title='Fare ($)', template='plotly_dark', height=450)
            st.plotly_chart(mc_fig, use_container_width=True)
        else:
            st.write('No `fare_amount` column found in the uploaded file.')

    st.markdown('---')

    st.subheader('Pickup map (sample)')
    if all(c in df_sample.columns for c in ['pickup_latitude','pickup_longitude','fare_amount']):
        map_fig = px.scatter_mapbox(df_sample, lat='pickup_latitude', lon='pickup_longitude', color='fare_amount', size_max=4, opacity=0.6, zoom=10, center=dict(lat=40.7128, lon=-74.0060), mapbox_style='carto-darkmatter', title='Pickup locations colored by fare')
        map_fig.update_layout(height=600)
        st.plotly_chart(map_fig, use_container_width=True)
    else:
        st.write('Upload data with pickup_latitude and pickup_longitude to view the map.')
