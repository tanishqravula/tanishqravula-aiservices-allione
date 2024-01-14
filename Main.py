import streamlit as st
import pandas as pd
import google.generativeai as genai
import re
import openpyxl
from PIL import Image
import pdf2image
import pytesseract
from pytesseract import Output, TesseractError
from functions import convert_pdf_to_txt_pages, convert_pdf_to_txt_file, save_pages, displayPDF, images_to_txt
import requests
import PyPDF2 
from docx import Document
from pptx import Presentation
import io
from bs4 import BeautifulSoup
import textwrap
from youtube_transcript_api import YouTubeTranscriptApi
import speech_recognition as sr
from io import StringIO

#Je t'aime plus que les mots,
#Plus que les sentiments,
#Plus que la vie elle-même
GOOGLE_API_KEY='AIzaSyBT3xgt_NZCLx2Auoyw0Dn3FBkgOIJqql4'

st.set_page_config(
    page_title="Tanishq AI Chat",
    page_icon="https://seeklogo.com/images/G/google-ai-logo-996E85F6FD-seeklogo.com.png",
    layout="wide",
)
# Path: Main.py
#Author: Sergio Demis Lopez Martinez
#------------------------------------------------------------
#HEADER
st.markdown('''
Powered by Tanishq <img src="https://seeklogo.com/images/G/google-ai-logo-996E85F6FD-seeklogo.com.png" width="20" height="20">
''', unsafe_allow_html=True)
st.caption("By Tanishq Ravula")

#------------------------------------------------------------
#LANGUAGE
langcols = st.columns([0.2,0.8])
with langcols[0]:
  lang = st.selectbox('Select your language',
  ('Español', 'English', 'Français', 'Deutsch',
  'Italiano', 'Português', 'Polski', 'Nederlands',
  'Русский', '日本語', '한국어', '中文', 'العربية',
  'हिन्दी', 'Türkçe', 'Tiếng Việt', 'Bahasa Indonesia',
  'ภาษาไทย', 'Română', 'Ελληνικά', 'Magyar', 'Čeština',
  'Svenska', 'Norsk', 'Suomi', 'Dansk', 'हिन्दी', 'हिन् '),index=1)

if 'lang' not in st.session_state:
    st.session_state.lang = lang
st.divider()

#------------------------------------------------------------
#FUNCTIONS
def extract_graphviz_info(text: str) -> list[str]:
  """
  The function extract_graphviz_info takes in a text and returns a list of graphviz code blocks found in the text.

  :param text: The text parameter is a string that contains the text from which you want to extract Graphviz information
  :return: a list of strings that contain either the word "graph" or "digraph". These strings are extracted from the input
  text.
  """

  graphviz_info  = text.split('```')

  return [graph for graph in graphviz_info if ('graph' in graph or 'digraph' in graph) and ('{' in graph and '}' in graph)]

def append_message(message: dict) -> None:
    """
    The function appends a message to a chat session.

    :param message: The message parameter is a dictionary that represents a chat message. It typically contains
    information such as the user who sent the message and the content of the message
    :type message: dict
    :return: The function is not returning anything.
    """
    st.session_state.chat_session.append({'user': message})
    return

@st.cache_resource
def load_model() -> genai.GenerativeModel:
    """
    The function load_model() returns an instance of the genai.GenerativeModel class initialized with the model name
    'gemini-pro'.
    :return: an instance of the genai.GenerativeModel class.
    """
    model = genai.GenerativeModel('gemini-pro')
    return model

@st.cache_resource
def load_modelvision() -> genai.GenerativeModel:
    """
    The function load_modelvision loads a generative model for vision tasks using the gemini-pro-vision model.
    :return: an instance of the genai.GenerativeModel class.
    """
    model = genai.GenerativeModel('gemini-pro-vision')
    return model
def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

# Function to generate content using GenAI model
def generate_content(model_type, content):
    model = genai.GenerativeModel(model_type)
    response = model.generate_content(content)
    return response.text
def transcribe_video(video_url):
    recognizer = sr.Recognizer()
    audio = None

    with sr.AudioFile(video_url) as source:
        audio = recognizer.record(source)

    return recognizer.recognize_google(audio)
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                text += cell.text + "\t"
            text += "\n"
    return text




#------------------------------------------------------------
#CONFIGURATION
genai.configure(api_key=GOOGLE_API_KEY)

model = load_model()

vision = load_modelvision()

if 'chat' not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = []

#st.session_state.chat_session

#------------------------------------------------------------
#CHAT

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'welcome' not in st.session_state or lang != st.session_state.lang:
    st.session_state.lang = lang
    welcome  = model.generate_content(f'''
    Da un saludo de bienvenida al usuario y sugiere que puede hacer
    (Puedes describir imágenes, responder preguntas, leer archivos texto, leer tablas,generar gráficos con graphviz, etc)
    eres un chatbot en una aplicación de chat creada en streamlit y python. generate the answer in {lang}''')
    welcome.resolve()
    st.session_state.welcome = welcome

    with st.chat_message('ai'):
        st.write(st.session_state.welcome.text)
else:
    with st.chat_message('ai'):
        st.write(st.session_state.welcome.text)

if len(st.session_state.chat_session) > 0:
    count = 0
    for message in st.session_state.chat_session:

        if message['user']['role'] == 'model':
            with st.chat_message('ai'):
                st.write(message['user']['parts'])
                graphs = extract_graphviz_info(message['user']['parts'])
                if len(graphs) > 0:
                    for graph in graphs:
                        st.graphviz_chart(graph,use_container_width=False)
                        if lang == 'Español':
                          view = "Ver texto"
                        else:
                          view = "View text"
                        with st.expander(view):
                          st.code(graph, language='dot')
        else:
            with st.chat_message('user'):
                st.write(message['user']['parts'][0])
                if len(message['user']['parts']) > 1:
                    st.image(message['user']['parts'][1], width=200)
        count += 1



#st.session_state.chat.history

cols=st.columns(7)

with cols[0]:
    if lang == 'Español':
      image_atachment = st.toggle("Adjuntar imagen", value=False, help="Activa este modo para adjuntar una imagen y que el chatbot pueda leerla")
    else:
      image_atachment = st.toggle("Attach image", value=False, help="Activate this mode to attach an image and let the chatbot read it")

with cols[1]:
    if lang == 'Español':
      txt_atachment = st.toggle("Adjuntar archivo de texto", value=False, help="Activa este modo para adjuntar un archivo de texto y que el chatbot pueda leerlo")
    else:
      txt_atachment = st.toggle("Attach text file", value=False, help="Activate this mode to attach a text file and let the chatbot read it")
with cols[2]:
    if lang == 'Español':
      csv_excel_atachment = st.toggle("Adjuntar CSV o Excel", value=False, help="Activa este modo para adjuntar un archivo CSV o Excel y que el chatbot pueda leerlo")
    else:
      csv_excel_atachment = st.toggle("Attach CSV or Excel", value=False, help="Activate this mode to attach a CSV or Excel file and let the chatbot read it")
with cols[3]:
    if lang == 'Español':
      graphviz_mode = st.toggle("Modo graphviz", value=False, help="Activa este modo para generar un grafo con graphviz en .dot a partir de tu mensaje")
    else:
      graphviz_mode = st.toggle("Graphviz mode", value=False, help="Activate this mode to generate a graph with graphviz in .dot from your message")
with cols[4]:
    if lang == 'Español':
        doc_atachment = st.toggle("Adjuntar PDF, PPT, DOCX", value=False, help="Activa este modo para adjuntar un archivo PDF, PPT o DOCX y que el chatbot pueda leerlo")
    else:

        doc_atachment = st.toggle("Attach PDF, PPT, DOCX", value=False, help="Activate this mode to attach a PDF, PPT, or DOCX file and let the chatbot read it")
with cols[5]:
    if lang == 'Español':
        website_chat = st.toggle("Chatear con sitios web", value=False,
                                 help="Activa este modo para chatear con un sitio web y resumir su contenido")
    else:
        website_chat = st.toggle("Chat with websites", value=False,
                                 help="Activate this mode to chat with a website and summarize its content")
with cols[6]:
    if lang == 'Español':
        youtube_chat = st.toggle("Chatear con youtube urls", value=False,
                                 help="Activa este modo para chatear con un sitio yotube resumir su contenido")
    else:
        youtube_chat = st.toggle("Chat with youtube urls(Only English videos are allowed)", value=False,
                                 help="Activate this mode to chat with a website and summarize its content")


if image_atachment:
    if lang == 'Español':
      image = st.file_uploader("Sube tu imagen", type=['png', 'jpg', 'jpeg'])
      url = st.text_input("O pega la url de tu imagen")
    else:
      image = st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg'])
      url = st.text_input("Or paste your image url")
else:
    image = None
    url = ''



if txt_atachment:
    if lang == 'Español':
      txtattachment = st.file_uploader("Sube tu archivo de texto", type=['txt'])
    else:
      txtattachment = st.file_uploader("Upload your text file", type=['txt'])
else:
    txtattachment = None
if doc_atachment:
    if lang == 'Español':
        docattachment = st.file_uploader("Sube tu archivo PDF, PPT, DOCX", type=['pdf', 'ppt', 'pptx', 'doc', 'docx'])
    else:
        docattachment = st.file_uploader("Upload your PDF, PPT, DOCX file", type=['pdf', 'ppt', 'pptx', 'doc', 'docx'])
else:
    docattachment = None

if csv_excel_atachment:
    if lang == 'Español':
      csvexcelattachment = st.file_uploader("Sube tu archivo CSV o Excel", type=['csv', 'xlsx'])
    else:
      csvexcelattachment = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])
else:
    csvexcelattachment = None
if youtube_chat:
    if lang == 'Español':
        youtube_url = st.text_input("Introduce la URL del sitio youtube:")
    else:
        youtube_url = st.text_input("Enter the URL of the youtube:")
    if youtube_url:
        video_id = youtube_url.split("=")[1]  # Extracting the video ID

        try:
            video_transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if video_transcript:
                full_transcript = ' '.join([x['text'] for x in video_transcript])
            else:
                # If no transcript available, attempt to transcribe the speech
                # Assuming the video has no transcript available
                st.warning("No transcript available. Attempting to transcribe the speech...")
                full_transcript = transcribe_video(youtube_url)

            #st.subheader("Video Transcript")
            #st.write(full_transcript)

            #st.subheader("Summary")
            #summarized_text = summarize_text(full_transcript)
            #st.write(summarized_text)

        except Exception as e:
            st.error(f"An error occurred: {e}")


if website_chat:
    if lang == 'Español':
        website_url = st.text_input("Introduce la URL del sitio web:")
    else:
        website_url = st.text_input("Enter the URL of the website:")

    if website_url:
        try:
            website_response = requests.get(website_url)
            website_html = website_response.text

            # Use Beautiful Soup to extract and summarize text content
            soup = BeautifulSoup(website_html, 'html.parser')
            paragraphs = soup.find_all('p')
            website_text = ' '.join([paragraph.get_text() for paragraph in paragraphs])
            content=f'display this content:{website_text} without missing even one word from the text fetched from information:{website_text} and complete the whole generated content'
            content1=f'organize the content: {website_text} into  tables '
            result = generate_content("gemini-pro", content)
            result1=generate_content("gemini-pro",content1)


            # Summarize the text if needed
            # (You can use a summarization library or method here)

            # Display the summarized text in the chat
            with st.chat_message('user'):
                st.write(f"Content: {website_url}")
            with st.chat_message('model'):
                st.markdown(to_markdown(result))
                st.markdown(to_markdown(result1))
                st.write(f'Extracted content from website:{website_text}')



        except Exception as e:
            with st.chat_message('model'):
                st.write(f"The website  does not allow to collect and fetch information according to the websites privacy and confidential information.You can try another website urls {str(e)}")
if lang == 'Español':
  prompt = st.chat_input("Escribe tu mensaje")
else:
  prompt = st.chat_input("Write your message")

if prompt:
    txt = ''
    if txtattachment:
        txt = txtattachment.getvalue().decode("utf-8")
        if lang == 'Español':
          txt = '   Archivo de texto: \n' + txt
        else:
          txt = '   Text file: \n' + txt
    if docattachment:
        path =docattachment.read()
        file_extension = docattachment.name.split('.')[-1].lower()
        try:
            if file_extension == 'pdf':
                
                if docattachment is not None:
                    reader = PyPDF2.PdfReader(docattachment)
                    doc_content = ""
                    for page_num in range(len(reader.pages)):
                        doc_content += reader.pages[page_num].extract_text()
                    texts, nbPages = images_to_txt(path, 'eng')
                    text_data_f = "\n\n".join(texts)
                    #text_data_text, nbPages_text = convert_pdf_to_txt_file(docattachment)
                    doc_content+=text_data_f
            elif file_extension == 'docx':
                docx_content = extract_text_from_docx(docattachment)
            elif file_extension == 'ppt' or file_extension == 'pptx':
                ppt_content = ""
                ppt_file = docattachment.read()
                presentation = Presentation(io.BytesIO(ppt_file))
                for slide_num, slide in enumerate(presentation.slides):
                    ppt_content += f'Slide {slide_num + 1}:\n'
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                             ppt_content += shape.text + '\n'
            else:
                raise ValueError(f'Unsupported file format: {file_extension}')

        except Exception as e:
            st.error(f"Error processing {file_extension} file: {e}")
        # Add more specific handling/logging if needed

        else:
            if file_extension == 'pdf':
                if lang == 'Español':
                    txt += f'   Archivo adjunto (PDF): \n{doc_content}'
                else:
                    txt += f'   Attached file (PDF): \n{doc_content}'
            elif file_extension == 'docx':
                if lang == 'Español':
                    txt += f'   Archivo adjunto (DOCX): \n{docx_content}'
                else:
                    txt += f'   Attached file (DOCX): \n{docx_content}'
            elif file_extension == 'ppt' or file_extension == 'pptx':
                if lang == 'Español':
                    txt += f'   Archivo adjunto (PPT): \n{ppt_content}'
                else:
                    txt += f'   Attached file (PPT): \n{ppt_content}'
    




    if csvexcelattachment:
        file_name = csvexcelattachment.name
        file_extension = file_name.split('.')[-1].lower()

        if file_extension == 'csv':
             df = pd.read_csv(csvexcelattachment)
             txt += '   Dataframe: \n' + str(df)
        elif file_extension in ['xlsx', 'xls']:
            wb = openpyxl.load_workbook(csvexcelattachment)
            sheet = wb.active
            for row in sheet.iter_rows():
            # extract each cell value
                for cell in row:
                    txt += str(cell.value)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
    if website_chat:
        txt=website_text
        prmt  = {'role': 'user', 'parts':[prompt+txt]}
    else:
        st.write('')
        prmt  = {'role': 'user', 'parts':[prompt+txt]}
    


    if graphviz_mode:
        if lang == 'Español':
          txt += '   Genera un grafo con graphviz en .dot \n'
        else:
          txt += '   Generate a graph with graphviz in .dot \n'
    if youtube_chat:
        txt=full_transcript 
        prmt  = {'role': 'user', 'parts':[prompt+txt]}
    else:
        st.write('')
        prmt  = {'role': 'user', 'parts':[prompt+txt]}

    if len(txt) > 900000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000:
        txt = txt[:900000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000] + '...'
    if image or url != '':
        if url != '':
            img = Image.open(requests.get(url, stream=True).raw)
        else:
            img = Image.open(image)
        prmt  = {'role': 'user', 'parts':[prompt+txt, img]}
    else:
        prmt  = {'role': 'user', 'parts':[prompt+txt]}

    append_message(prmt)

    if lang == 'Español':
      spinertxt = 'Espera un momento, estoy pensando...'
    else:
      spinertxt = 'Wait a moment, I am thinking...'
    with st.spinner(spinertxt):
        if len(prmt['parts']) > 1:
            response = vision.generate_content(prmt['parts'],stream=True,safety_settings=[
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_LOW_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_LOW_AND_ABOVE",
        },
    ]
)
            response.resolve()
        else:
            response = st.session_state.chat.send_message(prmt['parts'][0])

        try:
          append_message({'role': 'model', 'parts':response.text})
        except Exception as e:
          append_message({'role': 'model', 'parts':f'{type(e).name}: {e}'})


        st.rerun()



#st.session_state.chat_session
