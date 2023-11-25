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

from langchain.chat_models import ChatOpenAI
from openai import OpenAI


from htmlTemplates import css,bot_template,user_template

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
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

#upload chunks into cpu vector database
def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):

    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    #get chat history so there is context for follow up queries
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def generate_topics():
    response = st.session_state.questions({'question': "list the key topics in one sentence and number each topic and ensure a space after each topic"})
    return response["answer"]

def generate_context(number):

    if int(number) > 0:
        response = st.session_state.questions({'question': f"list information about {st.session_state.topic_list[int(number)-1]} in less than 300 tokens"})
        st.write(response["answer"])
    else:
        st.write("Error: please input correct number")



def test():
    client = OpenAI()
    prompt = [{"role":"system", "content": "write a 100 word story about a white cat in a white house"}]
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=prompt,
        temperature=0.5,
        max_tokens=1024
    )

    generated_text = response.choices[0].message.content
    st.write(generated_text)




def get_topics(topics):

    topic_list = []
    curr_string = ""
    for letter in topics:
        if letter != "\n":
            curr_string += letter
        else:
            topic_list.append(curr_string)
            curr_string = ""

    #removes empty list items
    for topic in topic_list:
        if topic == "":
            topic_list.remove(topic)

    return topic_list


def display_topics():

    if st.session_state.topic_list is not None:
        for topic in st.session_state.topic_list:
            st.write(bot_template.replace("{{MSG}}", topic), unsafe_allow_html=True)
    return 1


def main():
    load_dotenv()
    st.set_page_config(page_title="TeachingAI", page_icon=":books:")
    #import css
    st.write(css,unsafe_allow_html=True)

    #if "conversation" not in st.session_state:
        #st.session_state.conversation = None
    if "topic_list" not in st.session_state:
        st.session_state.topic_list = None
    if "questions" not in st.session_state:
        st.session_state.questions = None

    st.header("Generate exam questions from your learning material:books:")
    if st.button("Start"):
        topics = generate_topics()
        st.session_state.topic_list = get_topics(topics)


    display_topics()

    number = st.text_input("Input the topic number")
    if number:
        generate_context(number)
        # st.write(context_list)
        # test()


    with st.sidebar:
        st.subheader("Upload your pdf")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            #generates a loading spinner while processing pdf
            with st.spinner("Processing"):
                #get pdf text
                data = get_pdf_text(pdf_docs)

                # get text chunks
                chunks = get_text_chunks(data)

                #create vector store
                vectorstore = get_vectorstore(chunks)

                #create conversation chain that is not re-intialised
                #st.session_state.conversation = get_converstation_chain(vectorstore)

                st.session_state.questions = get_conversation_chain(vectorstore)

    #session_state allows for out of scope usage



if __name__ == '__main__':
    main()