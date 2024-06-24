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

