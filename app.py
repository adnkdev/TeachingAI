import streamlit as st
#loads env files
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

#vectors and embeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

#create memory for context
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain



#converts the pdf into a string
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#returns a list of chunked text
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200,length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

#upload chunks into cpu vector database
def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_converstation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conveersation_chain =




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

                # get text chunks
                chunks = get_text_chunks(raw_text)

                #create vector store
                vectorstore = get_vectorstore(chunks)

                #create conversation chain
                conversation = get_converstation_chain(vectorstore)





if __name__ == '__main__':
    main()