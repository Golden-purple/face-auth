# face-auth

https://face-unlock.streamlit.app/

This uses a HaarCascade to quickly identify faces in the image. Then uses FaceNet512 to extract the face embeddings and stores it securely. 

During login, it compares the current embedding with the stored (if any) embedding and does a cosine similarity check to authenticate.  
