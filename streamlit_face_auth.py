import streamlit as st
from PIL import Image
import io
import cv2 as cv
from scipy.spatial.distance import cosine
import requests
import os
from dotenv import load_dotenv
import numpy as np
from deepface import DeepFace
from supabase import create_client , Client
from ultralytics import YOLO

load_dotenv()

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize YOLOv11 nano model for face detection
@st.cache_resource
def load_yolo_model():
    """Load YOLOv11 nano model and cache it for performance"""
    try:
        # Try to load YOLOv11 nano model
        model = YOLO('yolo11n.pt')  # YOLOv11 nano model
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv11 model: {e}")
        return None

yolo_model = load_yolo_model()

def detect_faces_yolo(image_rgb):
    """
    Detect faces using YOLOv11 with fallback strategies
    Returns: list of face rectangles as (x, y, w, h)
    """
    if yolo_model is None:
        return []
    
    try:
        # Strategy 1: Try to detect persons and extract head/face region
        results = yolo_model(image_rgb, verbose=False)
        
        faces = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class and confidence
                    cls = int(box.cls[0].cpu().numpy())
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Check if it's a person (class 0) with good confidence
                    if cls == 0 and confidence > 0.4:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Extract upper portion of person as potential face region
                        person_w = int(x2 - x1)
                        person_h = int(y2 - y1)
                        
                        # For face detection, take upper 30% of person height
                        # and center it horizontally with 70% of person width
                        face_h = int(person_h * 0.3)
                        face_w = int(person_w * 0.7)
                        
                        # Center the face region
                        face_x = int(x1 + (person_w - face_w) / 2)
                        face_y = int(y1)
                        
                        # Ensure coordinates are valid
                        face_x = max(0, min(face_x, image_rgb.shape[1] - face_w))
                        face_y = max(0, min(face_y, image_rgb.shape[0] - face_h))
                        face_w = max(0, min(face_w, image_rgb.shape[1] - face_x))
                        face_h = max(0, min(face_h, image_rgb.shape[0] - face_y))
                        
                        # Validate minimum face size
                        if face_w >= 50 and face_h >= 50:
                            faces.append((face_x, face_y, face_w, face_h))
        
        # If no person detected, try a different approach
        if not faces:
            # Strategy 2: Look for any object that could be face-like in the center region
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = box.conf[0].cpu().numpy()
                        
                        if confidence > 0.3:  # Lower threshold for backup strategy
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x, y = int(x1), int(y1)
                            w, h = int(x2 - x1), int(y2 - y1)
                            
                            # Check if it's reasonably face-sized and positioned
                            aspect_ratio = w / h if h > 0 else 0
                            img_h, img_w = image_rgb.shape[:2]
                            
                            # Face should be in upper half of image and have reasonable aspect ratio
                            if (0.5 <= aspect_ratio <= 2.0 and 
                                w >= 40 and h >= 40 and 
                                y < img_h * 0.7):  # Upper 70% of image
                                faces.append((x, y, w, h))
                                break  # Take first reasonable match
        
        return faces
        
    except Exception as e:
        st.warning(f"YOLOv11 detection failed: {e}")
        return []

if "page" not in st.session_state :
    st.session_state["page"] = "home"

def after_login() :
    st.session_state["page"] = "login"

def after_sign() :
    st.session_state["page"] = "signup"

def go_home():
    st.session_state["page"] = "home"

def successful_sign() :
    st.session_state["page"] = "success" 

with st.container():
    col1, col2 = st.columns([6, 1])
    with col2:
        st.button("Home ⬅️", on_click=go_home, use_container_width=True)

page = st.session_state["page"]

if page == "home" :
    st.title("Welcome to :orange[FaceAuth]  :man: ")
    st.write("")
    st.write("")
    st.write("")
    st.subheader("\nThis uses :green[DeepFace], a wrapper around FaceNet to verify and authenticate." , divider = "gray")

    st.write("") 
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    st.button(":blue[Sign up ]" , on_click = after_sign , type = "secondary" , use_container_width = True)
    st.button(":violet[Login]" , on_click = after_login ,  type = "secondary" , use_container_width = True)

elif page == "signup" :
    st.markdown('<h2 style="color:#20EDC4;"> Sign Up ☺️</h2>', unsafe_allow_html=True)
    st.write("")
    st.write("")

    username = st.text_input("Enter Username   :" , placeholder = "John Doe" )
    st.write(f"Current Username : {username}")
    st.write("")

    flag_name = True
    camera_allow = st.checkbox("Enable Camera")
    if(camera_allow) :
        if(username == "") :
            flag_name = False
            st.warning("Kindly give a User name ! ")
    st.info("The image you are sharing will NOT be stored. Peace ✌🏻 ")
    img = st.camera_input("Keep your face in the center :pray:", disabled = not (camera_allow and 
                                                                                 flag_name))

    if img is not None :
        image = Image.open(img)
        img_rgb = np.array(image.convert("RGB"))
        faces = detect_faces_yolo(img_rgb)
        if len(faces) == 0 :
            st.warning("No faces captured. ")
        else:
            try:
                face = faces[0]
                x,y,w,h = face
                face_image = img_rgb[y:y+h , x:x+w]
                embedding_obj = DeepFace.represent(face_image, model_name = "Facenet512",
                                                   enforce_detection = False)[0]
                embedding = embedding_obj["embedding"]

                st.success("Face embedding generated successfully!")

                st.image(face_image, caption = "Captured Face", channels = "RGB")

            except Exception as e:
                st.error(f"Error extracting embedding: {e}")
    if(username.strip == "") :
        st.warning("Kindly give a Name ! ")
    else :
        try:
            data = {
                "username": username,
                "embedding": embedding  
            }

            response = supabase.table("users").insert(data).execute()

            if response:
                st.success(f"User '{username}' registered successfully 🤗")
                st.balloons()
            else:
                st.error(f"Failed to store in database !")

        except Exception as e:
            if(camera_allow and flag_name) :
                st.error(f"Error storing data in Supabase: {e}")

elif page == "success" :
    st.balloons()
    st.success("You have signed in successfully ! ")

elif page == "login" :
    st.markdown('<h2 style="color:#0691DB;"> Login ☺️</h2>', unsafe_allow_html = True)
    st.write("")
    st.write("")
    flag_name_sign = True
    images = []
    username_sign = st.text_input("Enter Username   :" , placeholder = "John Doe" )
    camera_allow_sign = st.checkbox("Enable Camera : ")
    if(username_sign == "") :
        flag_name_sign = False
        if(camera_allow_sign) :
            st.warning("Give username !")
    img_count = st.slider("Capture more frames for better accuracy !" , 
              min_value = 1 , max_value = 10 , step = 1 , 
              disabled = not (camera_allow_sign and flag_name_sign) , width = "stretch")
    num_photos = img_count

    name_present = False    
    if(username_sign.strip()) :
        response = supabase\
            .table("users")\
            .select("embedding")\
            .eq("username", username_sign.strip())\
            .limit(1)\
            .execute()

        if not response.data:
            st.error("User not found. Please check your username.")
        else :
            name_present = True

    if(camera_allow_sign and flag_name_sign and name_present) :
        st.info(f"Capture {img_count} photos ! ")
        for i in range(img_count):
            img = st.camera_input(f"Frame {i+1}", key = f"cam_{i}")
            if img:
                images.append(img)

    face_embeddings = []
    if len(images) == num_photos:
        st.info("Processing captured images...")

        for i, img in enumerate(images):
            image = Image.open(img)
            img_rgb = np.array(image.convert("RGB"))
            faces = detect_faces_yolo(img_rgb)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_img = img_rgb[y:y+h, x:x+w]

                try:
                    emb = DeepFace.represent(face_img , model_name = "Facenet512" ,
                                              enforce_detection = False)[0]["embedding"]
                    face_embeddings.append(emb)
                    st.image(face_img , caption = f"Face {i+1}" , channels = "RGB")
                except Exception as e:
                    st.warning(f"Frame {i+1}: Embedding failed - {e}")
            else:
                st.warning(f"Frame {i+1}: No face detected.")

        if len(face_embeddings) == num_photos:
            avg_embedding = np.mean(np.array(face_embeddings) , axis = 0)
            st.success("Starting Comparison.")

        if (avg_embedding is not None and username_sign.strip()):
            try:
                response = supabase\
                    .table("users")\
                    .select("embedding")\
                    .eq("username", username_sign.strip())\
                    .limit(1)\
                    .execute()

                if not response.data:
                    st.error("User not found. Please check your username.")
                else:
                    user_data = response.data[0]
                    stored_embedding = np.array(user_data["embedding"])

                    distance = cosine(avg_embedding, stored_embedding)
                    threshold = 0.20

                    if distance < threshold:
                        st.success(f"Welcome {username_sign}!  ")
                        st.write(f"Similarity score: {1 - distance:.2f}")
                        st.button(":green[Continue]" , use_container_width = True , on_click = successful_sign)
                        
                    else:
                        st.error("Face does not match the username. NOT Authenticated ! ")

            except Exception as e:
                st.error(f"Error during login verification: {e}")




    















