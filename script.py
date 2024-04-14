import os

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import datetime
import plotly.express as px
import plotly.graph_objects as go
from persiantools.jdatetime import JalaliDate

st.set_page_config(page_title="Traffic Counter Data", page_icon="😎")
# st.set_config('browser.uiDirection', 'RTL')

with open( "styles.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

def find_dates():
    dates = []
    for year in range(1395, 1403):
        for month in [1, 12]:
            if year == 1402 and month == 12:
                continue
            for day in range(1, 32):
                try:
                    JalaliDate(year, month, day)
                    dates.append(f"{year}-{month}-{day}")
                except:
                    pass
    return dates


@st.cache_resource
def load_data():
    path = "data/data.csv.zip"
    if not os.path.isfile(path):
        # path = f"Path to GitHub repository"
        path = None

    data = pd.read_csv(
        path,
        names = [
            'road code', 'road name', 'start time', 'end time', 'operation length (minutes)', 'class 1', 'class 2',
            'class 3', 'class 4', 'class 5', 'estimated number', 'province', 'start city', 'end city', 'edge name'
        ],
        skiprows=1,
        usecols=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19]
    )
    # Change "start time" to date using Jalali
    data['start date'] = data['start time'].apply(lambda x: JalaliDate(
        int(x.split(' ')[0].split('/')[0]),
        int(x.split(' ')[0].split('/')[1]),
        int(x.split(' ')[0].split('/')[2])
    ))

    data['end date'] = data['end time'].apply(lambda x: JalaliDate(
        int(x.split(' ')[0].split('/')[0]),
        int(x.split(' ')[0].split('/')[1]),
        int(x.split(' ')[0].split('/')[2])
    ))

    return data

@st.cache_resource
def load_coordinates():
    path = "data/coordinates.csv.zip"
    if not os.path.isfile(path):
        # path = f"Path to GitHub repository"
        path = None
    
    coordinates = pd.read_csv(
        path,
        names=[
            "city",
            "lat",
            "lon"
        ],
        skiprows=1,
        usecols=[0, 1, 2]
    )

    return coordinates

@st.cache_resource
def load_population():
    path = "data/population.csv.zip"
    if not os.path.isfile(path):
        # path = f"Path to GitHub repository"
        path = None
    
    population = pd.read_csv(
        path,
        names=[
            "city",
            "population"
        ],
        skiprows=1,
        usecols=[0, 1]
    )

    return population

def map(data, lat, lon, zoom):
    # Add column color to data
    st.write(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state={
                "latitude": lat,
                "longitude": lon,
                "zoom": zoom,
                "pitch": 50,
            },
            layers=[
                pdk.Layer(
                    "HexagonLayer",
                    data=data,
                    get_position=["lon", "lat"],
                    get_elevation_weight="tourist",
                    get_color_weight="tourist",
                    color_range=[
                        [215,48,39],
                        [252,141,89],
                        [254,224,139],
                        [217,239,139],
                        [145,207,96],
                        [26,152,80]
                    ],
                    radius=5000,
                    elevation_scale=6,
                    opacity=0.8,
                    elevation_range=[0, 8000],
                    pickable=True,
                    extruded=True,
                ),
            ],
        )
    )

@st.cache_data
def city_input(data, start_date, end_date):
    start_date = JalaliDate(
        int(start_date.split("-")[0]), int(start_date.split("-")[1]), int(start_date.split("-")[2]
    ))
    end_date = JalaliDate(
        int(end_date.split("-")[0]), int(end_date.split("-")[1]), int(end_date.split("-")[2]
    ))
    data = data[(data["start date"] >= start_date) & (data["end date"] <= end_date)]
    # Create a df with columns city, input, output, net
    city_input = pd.DataFrame(columns=["date", "city", "input", "output", "net", "tourist", "province"])
    cities = data["start city"].unique()
    date_range = pd.date_range(start_date.to_gregorian(), end_date.to_gregorian())
    jalali_range = [JalaliDate(date) for date in date_range]

    # Add a row for combinations of cities and jalali_range
    for city in cities:
        for date in jalali_range:
            city_input = pd.concat([city_input, pd.DataFrame({
                "date": [date],
                "city": [city],
                "input": [0],
                "output": [0],
                "net": [0],
                "tourist": [0],
                "province": [data[data["start city"] == city]["province"].values[0]]
            })])
    
    for _, row in data.iterrows():
        # Check if the city is in the df
        city_input.loc[ (city_input["city"] == row["start city"]) & (city_input["date"] == row["start date"]), "output"] += row["class 1"]
        city_input.loc[ (city_input["city"] == row["end city"]) & (city_input["date"] == row["end date"]), "input"] += row["class 1"]

    city_input["net"] = city_input["input"] - city_input["output"]
    city_input["tourist"] = city_input["net"].apply(lambda x: abs(x))
    # Groupby city
    city_input = city_input.groupby("city").agg({
        "input": "sum",
        "output": "sum",
        "net": "sum",
        "tourist": "sum",
        "province": "first"
    }).reset_index()
    city_input = city_input.merge(coordinates, on="city", how="left")
    city_input = city_input.merge(population, on="city", how="left")

    return city_input

@st.cache_data
def pca(data, start_date, end_date):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    data = aggregate_data(data, start_date, end_date)
    data = data[~data["city"].isin([
        'Ghazvin',
        'Gilan Border',
        'Golestan Forest',
        'Imamzadeh Hashem Tehran',
        'Kandovan',
        'Park Jangali',
        'Se-rah-e Kalaleh',
        'Sisangan'
    ])]
    X = data
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    data["pca-2d-one"] = X_2d[:,0]
    data["pca-2d-two"] = X_2d[:,1]

    return data

@st.cache_data
def aggregate_data(data, start_date, end_date):
    aggregate_data = city_input(data, start_date, end_date)
    aggregate_data["input per capita"] = aggregate_data["input"] / (aggregate_data["population"] + 1)
    aggregate_data["output per capita"] = aggregate_data["output"] / (aggregate_data["population"] + 1)
    aggregate_data["net per capita"] = aggregate_data["net"] / (aggregate_data["population"] + 1)
    aggregate_data["tourist per capita"] = aggregate_data["tourist"] / (aggregate_data["population"] + 1)
    return aggregate_data

@st.cache_data
def tsne(data, start_date, end_date):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    data = aggregate_data(data, start_date, end_date)
    data = data[~data["city"].isin([
        'Ghazvin',
        'Gilan Border',
        'Golestan Forest',
        'Imamzadeh Hashem Tehran',
        'Kandovan',
        'Park Jangali',
        'Se-rah-e Kalaleh',
        'Sisangan'
    ])]
    X = pca(data, start_date, end_date)
    X = StandardScaler().fit_transform(X)

    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)
    data["tsne-2d-one"] = X_2d[:,0]
    data["tsne-2d-two"] = X_2d[:,1]

    return data


@st.cache_data
def dbscan(data, start_date, end_date):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    data = aggregate_data(data, start_date, end_date)
    data = data[~data["city"].isin([
        'Ghazvin',
        'Gilan Border',
        'Golestan Forest',
        'Imamzadeh Hashem Tehran',
        'Kandovan',
        'Park Jangali',
        'Se-rah-e Kalaleh',
        'Sisangan'
    ])]
    X = data[["tourist per capita", "tourist"]].values
    X = StandardScaler().fit_transform(X)

    # Cluster to 3 clusters
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    data["cluster"] = db.labels_

    return data

@st.cache_data
def kmeans(data, start_date, end_date):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    data = aggregate_data(data, start_date, end_date)
    data = data[~data["city"].isin([
        'Ghazvin',
        'Gilan Border',
        'Golestan Forest',
        'Imamzadeh Hashem Tehran',
        'Kandovan',
        'Park Jangali',
        'Se-rah-e Kalaleh',
        'Sisangan'
    ])]
    X = data[["tourist per capita", "tourist"]].values
    X = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=8).fit(X)
    data["cluster"] = kmeans.labels_

    return data

@st.cache_data
def gmm(data, start_date, end_date):
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    data = aggregate_data(data, start_date, end_date)
    data = data[~data["city"].isin([
        'Ghazvin',
        'Gilan Border',
        'Golestan Forest',
        'Imamzadeh Hashem Tehran',
        'Kandovan',
        'Park Jangali',
        'Se-rah-e Kalaleh',
        'Sisangan'
    ])]
    X = data[["tourist per capita", "tourist"]].values
    X = StandardScaler().fit_transform(X)

    gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
    data["cluster"] = gmm.predict(X)

    return data

@st.cache_data
def mpoint(lat, lon):
    return (np.average(lat), np.average(lon))

@st.cache_data
def weekly_data(data, weeks):
    for i in range(len(weeks)):
        start_date = JalaliDate(
            int(weeks[i].split("-")[0]),
            int(weeks[i].split("-")[1]),
            int(weeks[i].split("-")[2])
        )
        end_date = start_date + datetime.timedelta(days=6)



data = load_data()
coordinates = load_coordinates()
population = load_population()

if not st.session_state.get("url_synced", False):
    try:
        start_date = st.query_params["start_date"][0]
        end_date = st.query_params["end_date"][0]
        st.session_state["start_date"] = start_date
        st.session_state["end_date"] = end_date
        st.session_state["url_synced"] = True
    except KeyError:
        pass


def update_query_params():
    date = st.session_state["date"]
    start_date = date[0]
    end_date = date[1]
    st.experimental_set_query_params(start_date=start_date, end_date=end_date)

st.title("شمال: قصه‌ای برای نوروز")
st.write(
"""
شمال ایران با طبیعتی بکر آب و هوای دل‌پذیرش همواره یکی از مقاصد گردشگری محبوب بوده است. از رشت تا مازندران، از تالش تا آستارا، هر شهر و ناحیه‌ای در این منطقه دارای جذابیت‌های منحصر به فردی است که هر ساله گردشگران بسیاری را جذب می‌کند. اما در این میان، برخی شهرها برتری بیشتری دارند و مقصدهایی پرطرفدارتر برای گردشگران محسوب می‌شوند.د.

در این پروژه، ما قصد داریم با بررسی داده های تردد شمار به  شناسایی و معرفی پرطرفدارترین شهرهای شمال ایران بپردازیم. این تحقیق به هدف ارائه اطلاعاتی کامل و مفید به گردشگران و علاقه‌مندان به سفر در این منطقه، و همچنین به عنوان یک منبع مرجع برای محققین و برنامه‌ریزان گردشگری و توسعه گردشگری ارائه می‌شود.

"""
)

st.header("آغاز سفر")
st.write("""
استان های شمالی انتخاب بسیاری از ایرانیان در همه فصل هاست اما بدون شک، بهار شمال چیز دیگری است! این فصل به همراه آغاز رونق طبیعت، آب و هوای معتدل و روزهای آفتابی، جذابیت خاصی به سفرها در این مناطق می‌بخشد. تعطیلات نوروز نیز بهانه‌ای عالی برای فرار از روزمرگی و پرداختن به سفرهای خانوادگی و دوستانه است.
""")
st.subheader("نمودار برای نمایش بیشتر بودن ترددها در هفته دوم")
st.write(
    """

با تحلیل داده‌های تردد شمارها، مشخص شده است که هفتهٔ دوم تعطیلات، یکی از موقعیت‌های محبوب برای  مسافرت‌هاست.
.این انتخاب ممکن است به دلایل مختلفی اتفاق بیافتد؛ به عنوان مثال، بسیاری از مسافران ترجیح می‌دهند که ابتدا  لحظه تحویل سال را در خانه‌های خودشان جشن گرفته و سپس به سفر بروند 

    """
)
start_date, end_date = st.select_slider(
    'Select a range for analysis',
    options=find_dates(),
    key='date',
    value=("1395-1-1", "1395-1-10"),
    on_change=update_query_params
)

city_input_data = city_input(data, start_date, end_date)
st.dataframe(city_input_data, height=300, width=800)

mazandaran = [36.5700, 51.5200]
zoom_level = 7
midpoint = mpoint(coordinates["lat"], coordinates["lon"])

st.subheader("**Distribution of Net tourist in Different Cities**")
map_data = city_input(data, start_date, end_date)
# Keep only lat, lon, and net columns
map_data = map_data[["lat", "lon", "net", "tourist"]]
map(map_data, mazandaran[0], mazandaran[1], zoom_level)
# Add divider
st.write("---")

row2_1, row2_2 = st.columns((1, 1))

with row2_1:
    # Show dbscan results
    st.subheader("**Clustering Cities using DBSCAN**")
    dbscan_data = dbscan(data, start_date, end_date)
    fig = px.scatter(dbscan_data, x="tourist", y="tourist per capita", color="cluster", hover_name="city", template='plotly')
    # Make the cluster colors discrete
    fig.update_traces(marker=dict(size=12, opacity=0.8))
    # Make the axes logarithmic
    fig.update_layout(
        yaxis_type="log"
    )
    st.plotly_chart(fig)

with row2_2:
    # st.write("**Clustering Cities using K-Means**") with Bigger font
    st.subheader("**Clustering Cities using K-Means**")
    kmeans_data = kmeans(data, start_date, end_date)
    fig = px.scatter(kmeans_data, x="tourist", y="tourist per capita", color="cluster", hover_name="city", template='plotly')
    # Make the cluster colors discrete
    fig.update_traces(marker=dict(size=12, opacity=0.8))
    # Make the axes logarithmic
    fig.update_layout(
        yaxis_type="log"
    )
    st.plotly_chart(fig)

row3_1, row2_3 = st.columns((1, 1))

with row3_1:
    # Show gmm results
    st.subheader("**Clustering Cities using Gaussian Mixture Model**")
    gmm_data = gmm(data, start_date, end_date)
    fig = px.scatter(gmm_data, x="tourist", y="tourist per capita", color="cluster", hover_name="city", template='plotly')
    # Make the cluster colors discrete
    fig.update_traces(marker=dict(size=12, opacity=0.8))
    # Make the axes logarithmic
    fig.update_layout(
        yaxis_type="log"
    )
    st.plotly_chart(fig)