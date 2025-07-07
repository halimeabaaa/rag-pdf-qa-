from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import time
import weaviate
import os

load_dotenv()

wcs_cluster_url = os.getenv("WEAVIATE_URL")
wcs_api_key = os.getenv("WEAVIATE_API_KEY")

pdf_loader = PyPDFLoader("your document.pdf")
docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n", "\n\n", ".", "?", "!", " "]
)
chunks = text_splitter.split_documents(docs)
embedding_s = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcs_cluster_url,
    auth_credentials=weaviate.auth.AuthApiKey(api_key=wcs_api_key)
)


db = WeaviateVectorStore.from_documents(chunks, embedding_s, client=weaviate_client)

retriever = db.as_retriever()


llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.1)

system_prompt = ("""
    Use the text provided to answer the question. 
    If you don't know the answer, say you don't know. 
    Context: {context}
    """
)
prompt_template = ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human", "{question}")
])


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt":prompt_template}
)

question = "pdf ne hakkında?"

start_time=time.time()
response = qa_chain.run(question)
end_time = time.time()

total_time=end_time-start_time

char_count=len(response)
total_token=char_count*0.7



print(response)
print("\n\n Aradan geçen zaman: ",total_time)
print(" \n\n metinde kullanılan harf sayısı: ",char_count)
print("\n\n total token değeri: ",total_token)




