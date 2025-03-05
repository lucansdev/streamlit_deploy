import streamlit as st
import os
import tempfile

# Configuração inicial
st.set_page_config(page_title="Chat com Documentos", layout="wide")

# Funções de processamento
def process_input(uploaded_files):
    """Processa todos os inputs e retorna texto consolidado"""

    
    # Processar arquivos
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file.flush()
            loader = FactoryLoader()
            processor = loader.get_loader(file.type,temp_file.name)
    
    return processor

# Interface principal
def main():
    st.title("📚 Chat Inteligente com Documentos")
    
    # Sidebar para upload
    with st.sidebar:
        st.header("Carregar Documentos")
        uploaded_files = st.file_uploader(
            "Arraste arquivos PDF/TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        if st.button("Processar Documentos"):
            with st.spinner("Processando..."):
                st.session_state.full_text = process_input(uploaded_files)
                st.success("Documentos prontos para consulta!")
                

    # Área de chat
    if "full_text" not in st.session_state:
        st.info("Carregue documentos para começar")
        return

    # Histórico de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibir mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt := st.chat_input("Faça sua pergunta sobre os documentos"):
        # Adicionar mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gerar resposta
        with st.spinner("Pensando..."):
            try:
                # Substituir por sua integração com LangChain
                ia = process_input(uploaded_files)
                response = ia.run(prompt)
                
            except Exception as e:
                response = f"Erro: {str(e)}"

        # Adicionar e exibir resposta
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


from abc import ABC,abstractmethod
import os
import dotenv
from langchain_community.document_loaders.pdf import PyPDFLoader#type:ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter#type:ignore
from langchain_community.vectorstores.chroma import Chroma#type:ignore
from langchain_community.embeddings import HuggingFaceEmbeddings#type:ignore
from langchain_openai.llms import OpenAI#type:ignore
from langchain.chains.retrieval_qa.base import RetrievalQA#type:ignore
from langchain_community.document_loaders import TextLoader

dotenv.load_dotenv()


class FileLoader(ABC):

    @abstractmethod
    def processor_file(self):
        pass
    
    @abstractmethod
    def splitting_text(self):
        pass
    
    @abstractmethod
    def embedding_vector_store(self):
        pass

    @abstractmethod
    def call_ai(self):
        pass



class PdfLoader(FileLoader):
    def __init__(self,arquivos):
        self.file = arquivos

    def processor_file(self):

        loader = PyPDFLoader(self.file)
        arquivo = loader.load()

        return arquivo
    
    def splitting_text(self):

        split = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50,
            separators=["\n\n","\n","","."," "]

        )

        docs = split.split_documents(self.processor_file())
        return docs
    
    def embedding_vector_store(self):

        embedding = HuggingFaceEmbeddings()

        vector_db = Chroma.from_documents(documents=self.splitting_text(),embedding=embedding)

        return vector_db

    

    def call_ai(self):

        vector = self.embedding_vector_store()
        
        llm = OpenAI(api_key=st.secrets["openaiKey"])
        retriever = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector.as_retriever(search_type="mmr"),
            chain_type="refine",
            verbose=True
        )
            
        

        return retriever
        

class TxtLoader(FileLoader):
    def __init__(self,arquivos):
        self.file = arquivos

    def processor_file(self):

        loader = TextLoader(self.file)
        arquivo = loader.load()

        return arquivo
    
    def splitting_text(self):

        split = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50,
            separators=["\n\n","\n","","."," "]

        )

        docs = split.split_documents(self.processor_file())
        return docs
    
    def embedding_vector_store(self):


        embedding = HuggingFaceEmbeddings()

        vector_db = Chroma.from_documents(documents=self.splitting_text(),embedding=embedding)

        return vector_db

    
    
    def call_ai(self):

        vector = self.embedding_vector_store()
        
        llm = OpenAI(api_key=st.secrets["openaiKey"])
        retriever = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector.as_retriever(search_type="mmr"),
            chain_type="refine",
            verbose=True
        )
            
        
        return retriever
    


class FactoryLoader:
    def __init__(self):
        ...

    def get_loader(self,type,arquivo):
        if type == "application/pdf":
            pdf = PdfLoader(arquivo)
            final_pdf = pdf.call_ai()
            return final_pdf
        else:

            if type == "text/plain":
                txt = TxtLoader(arquivo)
                final_txt = txt.call_ai()
                return final_txt
            

if __name__ == "__main__":
    main()