import os
# To clone the repo
from git import Repo
# For context aware splitting 
from langchain.text_splitter import Language
# GenericLoader is needed to upload github repo.
from langchain_community.document_loaders.generic import GenericLoader 
from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain



#clone any github repositories 
def repo_ingestion(repo_url):
    os.makedirs("repo", exist_ok=True)
    repo_path = "repo/"
    Repo.clone_from(repo_url, to_path=repo_path)



#Loading repositories as documents
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                        glob = "**/*",
                                       suffixes=[".py"],
                                       parser = LanguageParser(language=Language.PYTHON, parser_threshold=500))    
    documents = loader.load()
    return documents




#Creating text chunks 
def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language = Language.PYTHON,
                                                             chunk_size = 2000,
                                                             chunk_overlap = 200)
    
    text_chunks = documents_splitter.split_documents(documents)
    return text_chunks



#loading embeddings model
def load_embedding():
    embeddings=OpenAIEmbeddings(disallowed_special=())
    return embeddings