# face-auth

https://face-unlock.streamlit.app/

This uses YOLOv11 nano for fast and accurate face detection in images. Then uses FaceNet512 to extract the face embeddings and stores it securely. 

During login, it compares the current embedding with the stored (if any) embedding and does a cosine similarity check to authenticate.  
