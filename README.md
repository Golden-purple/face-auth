# face-auth

https://face-unlock.streamlit.app/

Using frontalface haarcascade, quickly captures the face of the user.
We store the face’s embeddings from FaceNet512 securely in a table in supabase. Each embedding is a 512-dimensional abstract vector that captures the distinctive features of the face.
Storing the embeddings in a database is safe and harder to trace back to the person.
Instead of training the model every time a user signs up, making it a classification model, this captures the features and simply compares the stored vector, and the vector obtained real-time using cosine similarity (a dot product of the two vectors showing how similar they are).


