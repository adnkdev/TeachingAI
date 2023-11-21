import streamlit as st
#loads env files
from dotenv import load_dotenv
from PyPDF2 import PdfReader

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text




def main():
    load_dotenv()
    st.set_page_config(page_title="TeachingAI", page_icon=":books:")
    st.header("Generate exam questions from your learning material:books:")
    st.text_input("Ask a question:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            #generates a loading spinner while processing pdf
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)

                #get text chunks

                #creae vector stor




if __name__ == '__main__':
    main()