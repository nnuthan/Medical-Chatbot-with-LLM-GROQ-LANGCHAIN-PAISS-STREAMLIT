import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


#from langchain_huggingface import HuggingFaceEndpoint
#from langchain_core.prompts import PromptTemplate
#from langchain.chains import RetrievalQA

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv
load_dotenv()


# Step 1: Setup LLM (Mistral with HuggingFace)
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME="llama-3.1-8b-instant"

llm = ChatGroq(
    model=GROQ_MODEL_NAME,
    temperature=0.5,
    max_tokens=512,
    api_key=GROQ_API_KEY,
)

# Step 2: Connect LLM with FAISS and Create chain

DB_FAISS_PATH="vectorstore/db_faiss"  #to tell LLM about the location/path of db to access the data or retrieve 
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True) #load db


#step 3: Build RAG chain
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

#document combiner chain (stuff document into prompt)
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

#Retrieval chain (retriever+ doc combiner)
rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={'k':3}), combine_docs_chain)

# Create QA chain
"""qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)"""

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=rag_chain.invoke({'input': user_query})
print("RESULT: ", response["answer"])
print("\nSOURCE DOCUMENTS:")
for doc in response["context"]:
    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")
