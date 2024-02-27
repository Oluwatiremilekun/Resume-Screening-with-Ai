
import streamlit as st
import joblib
import PyPDF2
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model
loaded_model = joblib.load("knn_model.pkl")  # Replace with your filename

# Create the app's title and file uploader
st.title("Resume Classification App")
uploaded_file = st.file_uploader("Upload your resume")

def process_resume_file(uploaded_file):
    if uploaded_file is not None:
        try:
            # Get the content of the uploaded file
            file_content = uploaded_file.read()
            
            # Display the raw content of the resume
            st.subheader("Raw Content of the Resume:")
            st.text(file_content.decode("latin-1"))

            # Save the file locally
            file_path = "uploaded_resume." + uploaded_file.type.split("/")[1]
            with open(file_path, "wb") as f:
                f.write(file_content)

            # Extract text based on file type
            if file_path.endswith(".pdf"):
                with open(file_path, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    # Replace the deprecated getPage method with pages
                    page = pdf_reader.pages[0]
                    resume_text = page.extract_text().decode("utf-8")
            elif file_path.endswith((".doc", ".docx")):
                resume_text = docx2txt.process(file_path)
            else:
                st.error("Unsupported file format. Please upload a PDF or DOC/DOCX file.")
                return

            # Preprocessing and prediction
            vectorizer = TfidfVectorizer()
            resume_features = vectorizer.transform([resume_text])
            prediction = loaded_model.predict(resume_features)
            predicted_class = prediction[0]

            # Display the predicted class
            st.subheader("Prediction:")
            st.write("Predicted Class:", predicted_class)

        except Exception as exc:
            st.error(f"Error processing file: {exc}")

# Call the function to process the resume file
process_resume_file(uploaded_file)
