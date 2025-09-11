#%% imports
from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

#%% Main

### For Testing______________________________
# def main():
#     print("Main: Under Development")
#     load_dotenv()
#     print(os.getenv("OLLAMA_BASE_URL"))
#     print(os.getenv("OLLAMA_CHAT_MODEL"))

def main():
    load_dotenv()
    st.set_page_config(page_title="Single PDF Database")
    st.header("Ask Your PDF")

    #* Uploading the PDF file
    pdf = st.file_uploader("Upload Your PDF Here", type="pdf")
    
    #* Extracting PDF
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()        
        # st.write(text)  # DEBUG
        
        #* split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.write(chunks)  # DEBUG



if __name__ == '__main__':
    main()

# %%
