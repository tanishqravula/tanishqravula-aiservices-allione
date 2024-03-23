import streamlit as st
import tensorflow as tf
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image, ImageOps
import google.generativeai as genai
import numpy as np
import openai
from duckduckgo_search import ddg_images
import requests
from requests.exceptions import ConnectionError, ReadTimeout
from moviepy.editor import *
from bs4 import BeautifulSoup
from summarizer import Summarizer
import textwrap
import time
import os
import yt_dlp
import base64
#from transformers import BartForConditionalGeneration, BartTokenizer
import pyttsx3
import geocoder
import leafmap.foliumap as leafmap
import folium
import requests
from math import radians, sin, cos, sqrt, atan2
from geopy.geocoders import Nominatim
import webbrowser
from numpy import clip
from googleapiclient.discovery import build

GOOGLE_API_KEY='AIzaSyAHoNfvJhI4SwWqC75VfLS33mueiK23g2w'
google_api_key = "AIzaSyDkd8FH7Un6h68wnzw-PdBkCbCynmlOhyU"  
search_engine_id = "94a9058e786854003"  
google_custom_search = build("customsearch", "v1", developerKey=google_api_key)

num_images=20


st.set_page_config(
    page_title="Reduce Reuse and Recycle",
    page_icon=":mango:",
    initial_sidebar_state='auto'
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

openai.api_key = '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'
num_images = 16
# Haversine formula for distance calculation between two coordinates

@st.cache_resource
def load_model() -> genai.GenerativeModel:
    """
    The function load_model() returns an instance of the genai.GenerativeModel class initialized with the model name
    'gemini-pro'.
    :return: an instance of the genai.GenerativeModel class.
    """
    model = genai.GenerativeModel('gemini-pro')
    return model
def generate_content(model_type, content):
    model = genai.GenerativeModel(model_type)
    response = model.generate_content(content)
    return response.text
# Function to calculate distances from user location to waste management facilities using haversine formula


# Function to find waste management facilities based on general amenity
def haversine(coord1, coord2):
    R = 6371.0  # Radius of the Earth in kilometers

    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    a = clip(a, 0.0, 1.0)  # Ensure a is within the valid range

    # Ensure the argument passed to sqrt is non-negative
    a = max(0, a)

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# Function to calculate distances from user location to waste management facilities
def calculate_distances(user_location, waste_facilities):
    distances = []
    for facility in waste_facilities:
        facility_lat, facility_lon = float(facility["lat"]), float(facility["lon"])
        distance = haversine(user_location, (facility_lat, facility_lon))
        distances.append(distance)
    return distances

# Function to find waste management facilities based on general amenity
def find_waste_facilities(user_location):
    osm_endpoint = "https://overpass-api.de/api/interpreter"
    query = (
        f'[out:json];'
        f'node["amenity"="waste_disposal"](around:{400000},{user_location[0]},{user_location[1]});'
        f'out center;'
    )

    headers = {"Content-Type": "text/xml"}
    osm_response = requests.post(osm_endpoint, data=query, headers=headers)
    osm_data = osm_response.json()

    waste_facilities = []
    if osm_data.get("elements"):
        waste_facilities = osm_data["elements"]

    return waste_facilities

# Function to display waste management facilities and map
def display_waste_facilities(user_location):
    waste_facilities = find_waste_facilities(user_location)

    if waste_facilities:
        distances = calculate_distances(user_location, waste_facilities)
        facilities_within_400km = [(facility, distance) for facility, distance in zip(waste_facilities, distances) if distance <= 400]
        facilities_within_400km.sort(key=lambda x: x[1])  # Sort facilities by distance (nearest first)

        m = leafmap.Map(locate_control=True, latlon_control=True, draw_export=True, minimap_control=True)

        # Add a marker for the user's location
        folium.Marker(location=user_location, popup="Your Location", icon=folium.Icon(color="blue")).add_to(m)

        for facility, distance in facilities_within_400km:
            facility_name = facility.get("tags", {}).get("name", "Unnamed Facility")
            facility_lat = facility["lat"]
            facility_lon = facility["lon"]
            popup_text = f"{facility_name}<br>Distance: {distance:.2f} km"

            # Add a marker for the waste management facility
            leafmap.folium.Marker(location=(facility_lat, facility_lon), popup=popup_text, icon=folium.Icon(color="red")).add_to(m)

            # Draw a line from the input location to the waste management facility
            folium.PolyLine([user_location, (facility_lat, facility_lon)], color="green", weight=2.5, opacity=1).add_to(m)

        m.save("waste_management_map.html")

        # Display the map using st.components
        with open("waste_management_map.html", "r") as f:
            map_html = f.read()
        st.components.v1.html(map_html, height=700)

        if facilities_within_400km:
            # Display facility names and distances in the frontend
            st.subheader("Waste Management Facilities within 400 km:")
            for facility, distance in facilities_within_400km:
                facility_name = facility.get("tags", {}).get("name", "Unnamed Facility")

                # Check if the distance is less than a threshold to show in meters
                if distance < 1.0:
                    distance_display = f"{distance*1000:.2f} meters"
                else:
                    distance_display = f"{distance:.2f} km"

                st.write(f"- {facility_name}: {distance_display}")

            # Open Google Maps Directions to the nearest facility within 400 km
            nearest_facility = facilities_within_400km[0][0]
            #open_google_maps_directions(f"{user_location[0]},{user_location[1]}", f"{nearest_facility['lat']},{nearest_facility['lon']}")
        else:
            st.warning("No waste management facilities found within 400 km.")
    else:
        st.warning("No waste management facilities found within 100 km.")

def get_coordinates_from_location_name(location_name):
    geolocator = Nominatim(user_agent="waste_management_app")
    location = geolocator.geocode(location_name)
    if location:
        return location.latitude, location.longitude
    else:
        st.error(f"Unable to find coordinates for location: {location_name}")
        return None
# Function to open Google Maps Directions to the specified destination
def open_google_maps_directions(origin, destination):
    google_maps_url = f"https://www.google.com/maps/dir/{origin}/{destination}"
    webbrowser.open_new_tab(google_maps_url)
    st.write(f"Redirecting to Google Maps for directions...")


def scrape_duckduckgo_and_summarize(query):
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"}
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()

        # Add a delay to avoid sending too many requests too quickly
        time.sleep(2)

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract and combine snippets from the search results
        search_results = soup.find_all("a", class_="result__url")
        content = ""
        for result in search_results:
            snippet = result.find_next(class_="result__snippet")
            if snippet:
                snippet_text = snippet.text
                content += snippet_text + "\n"  # Add a newline for each snippet

        # Summarize the content to about 300-350 words
        model = Summarizer()
        summary = model(content, ratio=0.2)  # Adjust the ratio for the desired word count

        return summary
    except Exception as e:
        st.error(f"An error occurred while scraping DuckDuckGo: {str(e)}")
        return ""
genai.configure(api_key=GOOGLE_API_KEY)

model = load_model()



def bing_image_search(query, num_images):
    # Perform a Bing image search using BeautifulSoup
    # Replace this section with your actual Bing image search implementation

    image_urls = []

    try:
        # Replace this URL with your actual Bing image search URL
        url = f"https://www.bing.com/images/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            image_tags = soup.find_all("img")

            for image_tag in image_tags:
                image_src = image_tag.get("src")
                if image_src:
                    image_urls.append(image_src)
                if len(image_urls) >= num_images:
                    break
    except Exception as e:
        st.warning(f"Failed to fetch images from Bing. Error: {str(e)}")

    return image_urls
def concat_google_images(query, start_index):
    image_urls = []

    try:
      
        response = google_custom_search.cse().list(
            q=query,
            cx=search_engine_id,
            num=10, 
            start=start_index,
            searchType="image"
        ).execute()

        items = response.get("items", [])
        for item in items:
            image_urls.append(item["link"])

    except Exception as e:
        st.warning(f"Failed to fetch images from Google. Error: {str(e)}")

    return image_urls


def generate_pdf_summary(summary_text):
    # Create a PDF file
    pdf_filename = "summary.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)

    # Set the font and size for the PDF content
    c.setFont("Helvetica", 12)

    # Add the summary text to the PDF
    c.drawString(50, 750, "Summary:")
    text_lines = textwrap.wrap(summary_text, width=70)
    y_position = 730
    for line in text_lines:
        c.drawString(50, y_position, line)
        y_position -= 15  # Adjust for line spacing

    # Save the PDF
    c.showPage()
    c.save()

    return pdf_filename


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/mp4;base64,{b64}" download="{file_label}.mp4">Download {file_label}</a>'
    return href

def get_pdf_file_downloader_html(pdf_file_path, file_label='Download PDF'):
    with open(pdf_file_path, 'rb') as f:
        data = f.read()
    base64_pdf = base64.b64encode(data).decode()
    pdf_download_link = f'<a href="data:application/pdf;base64,{base64_pdf}" download="{file_label}.pdf">{file_label}</a>'
    return pdf_download_link


def text_to_audio(text):
    engine = pyttsx3.init()
    engine.save_to_file(text, 'output.mp3')
    engine.runAndWait()


def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction) == clss:
            return key


with st.sidebar:
    st.title("DIY Recommender system")
    st.subheader("DIY ideas for trashnet dataset")


#st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache_data()
def load_model():
    model = tf.keras.models.load_model('model_EfficientnetB0.h5')
    return model


with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # DIY ideas for dataset of 12 classes
         """
         )

files = st.file_uploader("Upload multiple images", type=["jpg", "png"], accept_multiple_files=True)


def import_and_predict(image_data, model):
    size = (224, 224)
    
    # Check if image_data is already a JpegImageFile object
    if isinstance(image_data, Image.Image):
        image = image_data
    else:
        # Convert the bytes-like object to a PIL Image object
        image = Image.open(io.BytesIO(image_data))
    
    # Resize the image using ImageOps.fit() if necessary
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert the image to numpy array
    img = np.asarray(image)
    
    # Reshape the numpy array for model prediction
    img_reshape = img[np.newaxis, ...]
    
    # Make prediction
    prediction = model.predict(img_reshape)
    
    return prediction
if files is None:
    st.text("Please upload an image file")
for file in files:
    st.session_state.video_generated = False  # Reset the variable when a new image is uploaded
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    #x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    #st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper',
                   'plastic', 'shoes', 'trash', 'white-glass']
    predicted_class = class_names[np.argmax(predictions)]
    string = "Detected waste : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'trash':
        try:
            content="DIY ideas for reusing and recycling trash waste in more than 16 points"
            response =generate_content("gemini-pro", content)
            # Display the chatbot's response
            st.text_area("DIY IDEAS:", value=response, height=1000)
        except Exception as e:
            st.error("An error occurred: {}".format(e))
        user_query = "DIY ideas for reusing and recycling trash waste"
        search_query = "DIY ideas for reusing and recycling trash  management "
        if "video_generated" not in st.session_state:
            st.session_state.video_generated = False
        if search_query and not st.session_state.video_generated:
            try:
                ydl_opts = {
                    'format': 'best',
                    'quiet': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    if search_query.isdigit():
                        video_url = f'https://www.youtube.com/watch?v={search_query}'
                    else:
                        info_dict = ydl.extract_info(f"ytsearch:{search_query}", download=False)
                        video_url = info_dict['entries'][0]['url']
                st.video(video_url)
                st.session_state.video_generated = True
            except Exception as e:
                st.error(f"An error occurred: {e}")        
        summary = generate_content("gemini-pro", content)
        pdf_filename = generate_pdf_summary(summary)
        st.success(f"PDF summary generated successfully. Check the {pdf_filename} file.")
        pdf_file_path = f"./{pdf_filename}"
        pdf_download_link = get_pdf_file_downloader_html(pdf_file_path, "Download PDF")
        st.markdown(pdf_download_link, unsafe_allow_html=True)
        waste_type = 'nearby trash waste management'
        if waste_type:
            geolocator = geocoder.ip("me")
            if geolocator.latlng is not None:
                user_location = (geolocator.latlng[0], geolocator.latlng[1])
                st.write(f"Your Location: {user_location[0]}, {user_location[1]}")
                display_waste_facilities(user_location)
        
    if class_names[np.argmax(predictions)]== 'plastic':
        try:
            content="DIY ideas for reusing and recycling plastic waste in more than 16 points"
            response =generate_content("gemini-pro", content)
            # Display the chatbot's response
            st.text_area("DIY IDEAS:", value=response, height=1000)
        except Exception as e:
            st.error("An error occurred: {}".format(e))
        user_query="DIY ideas for reusing and recycling plastic waste"
        search_query="DIY ideas for reusing and recycling plastic waste"
        if "video_generated" not in st.session_state:
            st.session_state.video_generated = False
        if search_query and not st.session_state.video_generated:
            try:
                ydl_opts = {
            'format': 'best',
            'quiet': True,}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    if search_query.isdigit():
                        video_url = f'https://www.youtube.com/watch?v={search_query}'
                    else:
                        info_dict = ydl.extract_info(f"ytsearch:{search_query}", download=False)
                        video_url = info_dict['entries'][0]['url']
                st.video(video_url)
                st.session_state.video_generated = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
        summary = generate_content("gemini-pro", content)
        pdf_filename = generate_pdf_summary(summary)
        st.success(f"PDF summary generated successfully. Check the {pdf_filename} file.")
        pdf_file_path = f"./{pdf_filename}"
        pdf_download_link = get_pdf_file_downloader_html(pdf_file_path, "Download PDF")
        st.markdown(pdf_download_link, unsafe_allow_html=True)
        waste_type = 'nearby plastic waste management'
        if waste_type:
            geolocator = geocoder.ip("me")
            if geolocator.latlng is not None:
                user_location = (geolocator.latlng[0], geolocator.latlng[1])
                st.write(f"Your Location: {user_location[0]}, {user_location[1]}")
                display_waste_facilities(user_location)


    if class_names[np.argmax(predictions)] == 'paper':
        try:
            content="DIY ideas for reusing and recycling paper waste in more than 16 points"
            response =generate_content("gemini-pro", content)
            # Display the chatbot's response
            st.text_area("DIY IDEAS:", value=response, height=1000)
        except Exception as e:
            st.error("An error occurred: {}".format(e))
        user_query="DIY ideas for reusing and recycling paper waste"
        search_query="DIY ideas for reusing and recycling paper waste"
        if "video_generated" not in st.session_state:
            st.session_state.video_generated = False
        if search_query and not st.session_state.video_generated:
            try:
                ydl_opts = {
            'format': 'best',
            'quiet': True,}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    if search_query.isdigit():
                        video_url = f'https://www.youtube.com/watch?v={search_query}'
                    else:
                        info_dict = ydl.extract_info(f"ytsearch:{search_query}", download=False)
                        video_url = info_dict['entries'][0]['url']
                st.video(video_url)
                st.session_state.video_generated = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
        summary = generate_content("gemini-pro", content)
        pdf_filename = generate_pdf_summary(summary)
        st.success(f"PDF summary generated successfully. Check the {pdf_filename} file.")
        pdf_file_path = f"./{pdf_filename}"
        pdf_download_link = get_pdf_file_downloader_html(pdf_file_path, "Download PDF")
        st.markdown(pdf_download_link, unsafe_allow_html=True)
        waste_type = 'nearby paper waste management'
        if waste_type:
            geolocator = geocoder.ip("me")
            if geolocator.latlng is not None:
                user_location = (geolocator.latlng[0], geolocator.latlng[1])
                st.write(f"Your Location: {user_location[0]}, {user_location[1]}")
                display_waste_facilities(user_location)
    if class_names[np.argmax(predictions)] == 'metal':
        try:
            content="DIY ideas for reusing and recycling metal waste in more than 16 points"
            response =generate_content("gemini-pro", content)
            # Display the chatbot's response
            st.text_area("DIY IDEAS:", value=response, height=1000)
        except Exception as e:
            st.error("An error occurred: {}".format(e))
        user_query="DIY ideas for reusing and recycling metal waste"
        search_query="DIY ideas for reusing and recycling metal waste"
        if "video_generated" not in st.session_state:
            st.session_state.video_generated = False
        if search_query and not st.session_state.video_generated:
            try:
                ydl_opts = {
            'format': 'best',
            'quiet': True,}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    if search_query.isdigit():
                        video_url = f'https://www.youtube.com/watch?v={search_query}'
                    else:
                        info_dict = ydl.extract_info(f"ytsearch:{search_query}", download=False)
                        video_url = info_dict['entries'][0]['url']
                st.video(video_url)
                st.session_state.video_generated = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
        summary = generate_content("gemini-pro", content)
        pdf_filename = generate_pdf_summary(summary)
        st.success(f"PDF summary generated successfully. Check the {pdf_filename} file.")
        pdf_file_path = f"./{pdf_filename}"
        pdf_download_link = get_pdf_file_downloader_html(pdf_file_path, "Download PDF")
        st.markdown(pdf_download_link, unsafe_allow_html=True)
        waste_type = 'nearby metal waste management'
        if waste_type:
            geolocator = geocoder.ip("me")
            if geolocator.latlng is not None:
                user_location = (geolocator.latlng[0], geolocator.latlng[1])
                st.write(f"Your Location: {user_location[0]}, {user_location[1]}")
                display_waste_facilities(user_location)
    if class_names[np.argmax(predictions)] == 'cardboard':
        try:
            content="DIY ideas for reusing and recycling cardboard waste in more than 16 points"
            response =generate_content("gemini-pro", content)
            # Display the chatbot's response
            st.text_area("DIY IDEAS:", value=response, height=1000)
        except Exception as e:
            st.error("An error occurred: {}".format(e))
        user_query="DIY ideas for reusing and recycling cardboard waste"
        search_query="DIY ideas for reusing and recycling cardboard waste"
        if "video_generated" not in st.session_state:
            st.session_state.video_generated = False
        if search_query and not st.session_state.video_generated:
            try:
                ydl_opts = {
            'format': 'best',
            'quiet': True,}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    if search_query.isdigit():
                        video_url = f'https://www.youtube.com/watch?v={search_query}'
                    else:
                        info_dict = ydl.extract_info(f"ytsearch:{search_query}", download=False)
                        video_url = info_dict['entries'][0]['url']
                st.video(video_url)
                st.session_state.video_generated = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
        summary = generate_content("gemini-pro", content)
        pdf_filename = generate_pdf_summary(summary)
        st.success(f"PDF summary generated successfully. Check the {pdf_filename} file.")
        pdf_file_path = f"./{pdf_filename}"
        pdf_download_link = get_pdf_file_downloader_html(pdf_file_path, "Download PDF")
        st.markdown(pdf_download_link, unsafe_allow_html=True)
        waste_type = 'nearby cardboard waste management'
        if waste_type:
            geolocator = geocoder.ip("me")
            if geolocator.latlng is not None:
                user_location = (geolocator.latlng[0], geolocator.latlng[1])
                st.write(f"Your Location: {user_location[0]}, {user_location[1]}")
                display_waste_facilities(user_location)
    if class_names[np.argmax(predictions)] == 'shoes':
        try:
            content="DIY ideas for reusing and recycling shoes waste in more than 16 points"
            response =generate_content("gemini-pro", content)
            # Display the chatbot's response
            st.text_area("DIY IDEAS:", value=response, height=1000)
        except Exception as e:
            st.error("An error occurred: {}".format(e))
        user_query="DIY ideas for reusing and recycling shoes waste"
        search_query="DIY ideas for reusing and recycling shoes waste"
        if "video_generated" not in st.session_state:
            st.session_state.video_generated = False
        if search_query and not st.session_state.video_generated:
            try:
                ydl_opts = {
            'format': 'best',
            'quiet': True,}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    if search_query.isdigit():
                        video_url = f'https://www.youtube.com/watch?v={search_query}'
                    else:
                        info_dict = ydl.extract_info(f"ytsearch:{search_query}", download=False)
                        video_url = info_dict['entries'][0]['url']
                st.video(video_url)
                st.session_state.video_generated = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
        summary = generate_content("gemini-pro", content)
        pdf_filename = generate_pdf_summary(summary)
        st.success(f"PDF summary generated successfully. Check the {pdf_filename} file.")
        pdf_file_path = f"./{pdf_filename}"
        pdf_download_link = get_pdf_file_downloader_html(pdf_file_path, "Download PDF")
        st.markdown(pdf_download_link, unsafe_allow_html=True)
        waste_type = 'nearby cardboard waste management'
        if waste_type:
            geolocator = geocoder.ip("me")
            if geolocator.latlng is not None:
                user_location = (geolocator.latlng[0], geolocator.latlng[1])
                st.write(f"Your Location: {user_location[0]}, {user_location[1]}")
                display_waste_facilities(user_location)
    if class_names[np.argmax(predictions)] == 'clothes':
        try:
            content="DIY ideas for reusing and recycling clothes waste in more than 16 points"
            response =generate_content("gemini-pro", content)
            # Display the chatbot's response
            st.text_area("DIY IDEAS:", value=response, height=1000)
        except Exception as e:
            st.error("An error occurred: {}".format(e))
        user_query="DIY ideas for reusing and recycling clothes waste"
        search_query="DIY ideas for reusing and recycling clothes waste"
        if "video_generated" not in st.session_state:
            st.session_state.video_generated = False
        if search_query and not st.session_state.video_generated:
            try:
                ydl_opts = {
            'format': 'best',
            'quiet': True,}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    if search_query.isdigit():
                        video_url = f'https://www.youtube.com/watch?v={search_query}'
                    else:
                        info_dict = ydl.extract_info(f"ytsearch:{search_query}", download=False)
                        video_url = info_dict['entries'][0]['url']
                st.video(video_url)
                st.session_state.video_generated = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
        summary = generate_content("gemini-pro", content)
        pdf_filename = generate_pdf_summary(summary)
        st.success(f"PDF summary generated successfully. Check the {pdf_filename} file.")
        pdf_file_path = f"./{pdf_filename}"
        pdf_download_link = get_pdf_file_downloader_html(pdf_file_path, "Download PDF")
        st.markdown(pdf_download_link, unsafe_allow_html=True)
        waste_type = 'nearby clothes waste management'
        if waste_type:
            geolocator = geocoder.ip("me")
            if geolocator.latlng is not None:
                user_location = (geolocator.latlng[0], geolocator.latlng[1])
                st.write(f"Your Location: {user_location[0]}, {user_location[1]}")
                display_waste_facilities(user_location)
    if class_names[np.argmax(predictions)] == 'battery':
        try:
            content="DIY ideas for reusing and recycling battery waste in more than 16 points"
            response =generate_content("gemini-pro", content)
            # Display the chatbot's response
            st.text_area("DIY IDEAS:", value=response, height=1000)
        except Exception as e:
            st.error("An error occurred: {}".format(e))
        user_query="DIY ideas for reusing and recycling battery waste"
        search_query="DIY ideas for reusing and recycling battery waste"
        if "video_generated" not in st.session_state:
            st.session_state.video_generated = False
        if search_query and not st.session_state.video_generated:
            try:
                ydl_opts = {
            'format': 'best',
            'quiet': True,}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    if search_query.isdigit():
                        video_url = f'https://www.youtube.com/watch?v={search_query}'
                    else:
                        info_dict = ydl.extract_info(f"ytsearch:{search_query}", download=False)
                        video_url = info_dict['entries'][0]['url']
                st.video(video_url)
                st.session_state.video_generated = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
        summary = generate_content("gemini-pro", content)
        pdf_filename = generate_pdf_summary(summary)
        st.success(f"PDF summary generated successfully. Check the {pdf_filename} file.")
        pdf_file_path = f"./{pdf_filename}"
        pdf_download_link = get_pdf_file_downloader_html(pdf_file_path, "Download PDF")
        st.markdown(pdf_download_link, unsafe_allow_html=True)
        waste_type = 'nearby battery waste management'
        if waste_type:
            geolocator = geocoder.ip("me")
            if geolocator.latlng is not None:
                user_location = (geolocator.latlng[0], geolocator.latlng[1])
                st.write(f"Your Location: {user_location[0]}, {user_location[1]}")
                display_waste_facilities(user_location)
    if class_names[np.argmax(predictions)] == 'biological':
        try:
            content="DIY ideas for reusing and recycling biological waste in more than 16 points"
            response =generate_content("gemini-pro", content)
            # Display the chatbot's response
            st.text_area("DIY IDEAS:", value=response, height=1000)
        except Exception as e:
            st.error("An error occurred: {}".format(e))
        user_query="DIY ideas for reusing and recycling biological waste"
        search_query="DIY ideas for reusing and recycling biological waste"
        if "video_generated" not in st.session_state:
            st.session_state.video_generated = False
        if search_query and not st.session_state.video_generated:
            try:
                ydl_opts = {
            'format': 'best',
            'quiet': True,}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    if search_query.isdigit():
                        video_url = f'https://www.youtube.com/watch?v={search_query}'
                    else:
                        info_dict = ydl.extract_info(f"ytsearch:{search_query}", download=False)
                        video_url = info_dict['entries'][0]['url']
                st.video(video_url)
                st.session_state.video_generated = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
        summary = generate_content("gemini-pro", content)
        pdf_filename = generate_pdf_summary(summary)
        st.success(f"PDF summary generated successfully. Check the {pdf_filename} file.")
        pdf_file_path = f"./{pdf_filename}"
        pdf_download_link = get_pdf_file_downloader_html(pdf_file_path, "Download PDF")
        st.markdown(pdf_download_link, unsafe_allow_html=True)

        waste_type = 'nearby biological waste management'
        if waste_type:
            geolocator = geocoder.ip("me")
            if geolocator.latlng is not None:
                user_location = (geolocator.latlng[0], geolocator.latlng[1])
                st.write(f"Your Location: {user_location[0]}, {user_location[1]}")
                display_waste_facilities(user_location)
    if class_names[np.argmax(predictions)] == 'brown-glass':
        try:
            content="DIY ideas for reusing and recycling brown-glass waste in more than 16 points"
            response =generate_content("gemini-pro", content)
            # Display the chatbot's response
            st.text_area("DIY IDEAS:", value=response, height=1000)
        except Exception as e:
            st.error("An error occurred: {}".format(e))
        user_query="DIY ideas for reusing and recycling brown glass"
        search_query="DIY ideas for reusing and recycling brown glass"
        if "video_generated" not in st.session_state:
            st.session_state.video_generated = False
        if search_query and not st.session_state.video_generated:
            try:
                ydl_opts = {
            'format': 'best',
            'quiet': True,}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    if search_query.isdigit():
                        video_url = f'https://www.youtube.com/watch?v={search_query}'
                    else:
                        info_dict = ydl.extract_info(f"ytsearch:{search_query}", download=False)
                        video_url = info_dict['entries'][0]['url']
                st.video(video_url)
                st.session_state.video_generated = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
        summary = generate_content("gemini-pro", content)
        pdf_filename = generate_pdf_summary(summary)
        st.success(f"PDF summary generated successfully. Check the {pdf_filename} file.")
        pdf_file_path = f"./{pdf_filename}"
        pdf_download_link = get_pdf_file_downloader_html(pdf_file_path, "Download PDF")
        st.markdown(pdf_download_link, unsafe_allow_html=True)
        waste_type = 'nearby brown glass waste management'
        if waste_type:
            geolocator = geocoder.ip("me")
            if geolocator.latlng is not None:
                user_location = (geolocator.latlng[0], geolocator.latlng[1])
                st.write(f"Your Location: {user_location[0]}, {user_location[1]}")
                display_waste_facilities(user_location)
    if class_names[np.argmax(predictions)] == 'green-glass':
        try:
            content="DIY ideas for reusing and recycling green-glass waste in more than 16 points"
            response =generate_content("gemini-pro", content)
            # Display the chatbot's response
            st.text_area("DIY IDEAS:", value=response, height=1000)
        except Exception as e:
            st.error("An error occurred: {}".format(e))
        user_query="DIY ideas for reusing and recycling green glass"
        search_query="DIY ideas for reusing and recycling green waste"
        if "video_generated" not in st.session_state:
            st.session_state.video_generated = False
        if search_query and not st.session_state.video_generated:
            try:
                ydl_opts = {
            'format': 'best',
            'quiet': True,}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    if search_query.isdigit():
                        video_url = f'https://www.youtube.com/watch?v={search_query}'
                    else:
                        info_dict = ydl.extract_info(f"ytsearch:{search_query}", download=False)
                        video_url = info_dict['entries'][0]['url']
                st.video(video_url)
                st.session_state.video_generated = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
        summary = generate_content("gemini-pro", content)
        pdf_filename = generate_pdf_summary(summary)
        st.success(f"PDF summary generated successfully. Check the {pdf_filename} file.")
        pdf_file_path = f"./{pdf_filename}"
        pdf_download_link = get_pdf_file_downloader_html(pdf_file_path, "Download PDF")
        st.markdown(pdf_download_link, unsafe_allow_html=True)
        waste_type = 'nearby green glass waste management'
        if waste_type:
            geolocator = geocoder.ip("me")
            if geolocator.latlng is not None:
                user_location = (geolocator.latlng[0], geolocator.latlng[1])
                st.write(f"Your Location: {user_location[0]}, {user_location[1]}")
                display_waste_facilities(user_location)
    if class_names[np.argmax(predictions)] == 'white-glass':
        try:
            content="DIY ideas for reusing and recycling white-glass waste in more than 16 points"
            response =generate_content("gemini-pro", content)
            # Display the chatbot's response
            st.text_area("DIY IDEAS:", value=response, height=1000)
        except Exception as e:
            st.error("An error occurred: {}".format(e))
        user_query="DIY ideas for reusing and recycling white glass"
        search_query="DIY ideas for reusing and recycling white glass"
        if "video_generated" not in st.session_state:
            st.session_state.video_generated = False
        if search_query and not st.session_state.video_generated:
            try:
                ydl_opts = {
            'format': 'best',
            'quiet': True,}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    if search_query.isdigit():
                        video_url = f'https://www.youtube.com/watch?v={search_query}'
                    else:
                        info_dict = ydl.extract_info(f"ytsearch:{search_query}", download=False)
                        video_url = info_dict['entries'][0]['url']
                st.video(video_url)
                st.session_state.video_generated = True
            except Exception as e:
                st.error(f"An error occurred: {e}")        
        summary = generate_content("gemini-pro", content)
        pdf_filename = generate_pdf_summary(summary)
        st.success(f"PDF summary generated successfully. Check the {pdf_filename} file.")
        pdf_file_path = f"./{pdf_filename}"
        pdf_download_link = get_pdf_file_downloader_html(pdf_file_path, "Download PDF")
        st.markdown(pdf_download_link, unsafe_allow_html=True)
        waste_type = 'nearby white glass waste management'
        if waste_type:
            geolocator = geocoder.ip("me")
            if geolocator.latlng is not None:
                user_location = (geolocator.latlng[0], geolocator.latlng[1])
                st.write(f"Your Location: {user_location[0]}, {user_location[1]}")
                display_waste_facilities(user_location)

    if class_names[np.argmax(predictions)] == 'trash':
        st.snow()
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'plastic':
        st.snow()
        st.sidebar.warning(string)

    elif class_names[np.argmax(predictions)] == 'paper':
        st.balloons()
        st.sidebar.warning(string)
    elif class_names[np.argmax(predictions)] == 'metal':
        st.balloons()
        st.sidebar.warning(string)
        
    elif class_names[np.argmax(predictions)] == 'clothes':
        st.balloons()
        st.sidebar.warning(string)
        
    elif class_names[np.argmax(predictions)] == 'cardboard':
        st.balloons()
        st.sidebar.warning(string)
        
    elif class_names[np.argmax(predictions)] == 'shoes':
        st.balloons()
        st.sidebar.success(string)
    elif class_names[np.argmax(predictions)] == 'battery':
        st.balloons()
        st.sidebar.success(string)
    elif class_names[np.argmax(predictions)] == 'biological':
        st.balloons()
        st.sidebar.success(string)
    elif class_names[np.argmax(predictions)] == 'brown-glass':
        st.balloons()
        st.sidebar.success(string)
    elif class_names[np.argmax(predictions)] == 'green-glass':
        st.balloons()
        st.sidebar.success(string)
    elif class_names[np.argmax(predictions)] == 'white-glass':
        st.balloons()
        st.sidebar.success(string)
st.markdown('''<html lang="en">
<head>
</head>
<body>
<a href="https://funny-torrone-bd8eb7.netlify.app/">Send Email for any queries</a>
</body>
</html>
''',unsafe_allow_html=True) 