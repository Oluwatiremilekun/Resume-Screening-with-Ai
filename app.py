


import streamlit as st
import joblib
import PyPDF2  # For PDF reading
import docx2txt  # For DOC/DOCX reading
import os  # For file path handling
from sklearn.feature_extraction.text import TfidfVectorizer  # Assuming you used TF-IDF


# Load the saved model
loaded_model = joblib.load("knn_model.pkl")  # Replace with your filename

# Create the app's title and file uploader
st.title("Resume Classification App")
uploaded_file = st.file_uploader("Upload your resume")


def process_resume_file():

    if uploaded_file is not None:
        file_path = uploaded_file.getvalue()
        file_name = os.path.basename(file_path)
        file_ext = file_name.split(".")[-1].replace(" ", "").casefold()

        try:
            if file_ext == "pdf":
                try:
                    with open(file_path, "rb") as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        page = pdf_reader.get_page(0)
                        resume_text = page.extract_text().decode("utf-8")  # Adjust codec if needed
                except Exception as exc:
                    st.error("Error reading PDF file:", exc)
            elif file_ext in ("doc", "docx"):
                try:
                    resume_text = docx2txt.process(file_path)
                except Exception as exc:
                    st.error("Error reading DOC/DOCX file:", exc)
            else:
                st.error("Unsupported file format. Please upload a PDF or DOC/DOCX file.")
                return
            
            # Preprocessing and prediction here
            vectorizer = TfidfVectorizer()
            resume_features = vectorizer.transform([resume_text])
            prediction = loaded_model.predict(resume_features)
            predicted_class = prediction[0]
            st.write("Predicted Class:", predicted_class)

             # Display the predicted class
            st.subheader("Prediction:")
            st.write("Predicted Class:", predicted_class)


        except Exception as exc:
            st.error("Error processing file:", exc)

# Call the function to process the resume file
process_resume_file()
