# Python In-built packages
from pathlib import Path
import PIL
import supervision as sv
from pydub.playback import play
from playsound import playsound
# External packages
import streamlit as st

# Local Modules
import settings
import helper

def play_sound():
    sound_file = 'mixkit-classic-alarm-995.wav'  # Replace with the path to your sound file
    playsound(sound_file)

# Setting page layout
st.set_page_config(
    page_title="Crowd and Violence Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Crowd and Violence Detection")

# Sidebar
st.sidebar.header("Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Crowd Detection', 'Violence Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Crowd Detection':
    model_path = Path(settings.DETECTION_MODEL1)
elif model_type == 'Violence Detection':
    model_path = Path(settings.DETECTION_MODEL2)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                names = res[0].names
                class_detections_values = []
                for k, v in names.items():
                    class_detections_values.append(res[0].boxes.cls.tolist().count(k))
                # create dictionary of objects detected per class
                classes_detected = dict(zip(names.values(), class_detections_values))
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                if 'people' in classes_detected:
                    no_of_people = classes_detected['people']
                #st.title(f"No of People: {no_of_people}")
                    if no_of_people:
                        if no_of_people == 0:
                            st.info("Crowd Density: Very Low")
                            st.success("No of people: {}".format(no_of_people))
                        elif 1< no_of_people < 10:
                            st.info("Crowd Density: Low")
                            st.success("No of people: {}".format(no_of_people))
                        elif 10< no_of_people <50:
                            st.info("Crowd Density: Medium")
                            st.success("No of people: {}".format(no_of_people))
                            play_sound()
                            st.warning("Alert Sound Played 🚨")
                        if no_of_people > 50:
                            st.title("Crowd Density: High")
                            st.title("No of people: {}".format(no_of_people))
                            play_sound()
                    else:
                        st.title("No People")
                elif 'violence' in classes_detected:
                    st.info("Violence Detected")
                    play_sound()
                    st.warning("Alert Sound Played 🚨")
                    #st.title(matches)
                #st.title(f"No of People: {classes_detected}")
                #st.text(res)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
