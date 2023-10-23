import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain 
import os


# Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
     -[streamlit](https://streamlit.io/)
    -[Langchain](https://www.langchain.com/)
    -[OpenAI](https://openai.com/)
                

    ''')

    add_vertical_space(5)
    st.write('Made with love by [Rishabh](https://youtube.com/playlist?list=PLdMtv-iP-2mQh_6HUSQmGxn6ByOAYd4bY&si=iiyDaltdXERhOHGv )')



def main():
    st.write("hello")
    st.header("chat with pdf")

    #upload a pdf file

    pdf = st.file_uploader("upload your pdf here",type="pdf")
    st.write(pdf.name)

    if pdf is not None:
        pdf_reader= PdfReader(pdf)

        text =""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
         
        #embeddings
        
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"  ):
             with open(f "{store_name}.pkl","rb" as f  ) :
                 VectorStore = pickle.load(f)
      #st.write("embeddings loaded from the disk")
        
    else:
        embeddings = OpenAIEmbeddings()
        
        VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
        with open(f"{store_name}.pkl","wt" ) as f:
                  pickle.dump(VectorStore,f)

        st.write("embeddings computation completed")

    #accept user question query
    query = st.text_input("ask questions about your pdf file")
    st.write(query)

    if query: 
        VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
        docs = VectoreStore.similarity_search(query=query,k=3)

        llm = OpenAI(temperature=0)
chain = load_qa_chain(llm=llm,chain_type="stuff")


        #st.write(docs)







#st.write(chunks)


if __name__ == "__main__":
    main()
