# import streamlit as st
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser

# # Load PDF and split text (can be cached to avoid reloads on every run)
# @st.cache_resource
# def setup_rag_chain():
#     loader = PyPDFLoader("reciepe generation.pdf")
#     documents = loader.load()

#     splitter = RecursiveCharacterTextSplitter(chunk_size=500)
#     splitted_text = splitter.split_documents(documents=documents)

#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     texts = [doc.page_content for doc in splitted_text]
#     embeddings = model.encode(texts)

#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(embeddings)

#     prompt = PromptTemplate(template="""
#     You are a helpful assistant. 
#     Give answer of the question only from the given context. 
#     If you think answer is not present in the given context then just say 'I don't know'.
    
#     Question: {question}
#     Context: {context}
#     """, input_variables=["question", "context"])

#     llm = HuggingFaceEndpoint(
#         repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#         task="text-generation"
#     )
#     chat_model = ChatHuggingFace(llm=llm)
#     parser = StrOutputParser()

#     def retrival_process(q):
#         q_em = model.encode(q)
#         D, I = index.search(np.array([q_em]), k=5)
#         context = "\n".join([texts[i] for i in I[0]])
#         return context

#     retrival_runnable = RunnableLambda(retrival_process)
#     parallel_chain = RunnableParallel({
#         "context": retrival_runnable,
#         "question": RunnablePassthrough()
#     })

#     rag_chain = parallel_chain | prompt | chat_model | parser

#     return rag_chain

# rag_chain = setup_rag_chain()

# # --- Streamlit UI ---
# st.title("📚RAG Chatbot")

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# with st.form("chat_form", clear_on_submit=True):
#     user_input = st.text_input("Ask a question from the Document:")
#     submitted = st.form_submit_button("Ask")

# if submitted and user_input:
#     with st.spinner("Thinking..."):
#         response = rag_chain.invoke(user_input)
#         st.session_state.chat_history.append(("You", user_input))
#         st.session_state.chat_history.append(("Bot", response))

# # Display chat history
# for sender, message in st.session_state.chat_history:
#     if sender == "You":
#         st.markdown(f"**🧑‍💬 You:** {message}")
#     else:
#         st.markdown(f"**🤖 Bot:** {message}")


import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

# --- Streamlit UI ---
st.title("📚 RAG Chatbot: Ask Your Document")

uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

if uploaded_file:
    if "rag_chain" not in st.session_state:
        with st.spinner("Setting up RAG chain..."):

            # Save uploaded file to temp file
            temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Load PDF
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()

            # Split
            splitter = RecursiveCharacterTextSplitter(chunk_size=500)
            splitted_text = splitter.split_documents(documents=documents)
            texts = [doc.page_content for doc in splitted_text]

            # Embeddings
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(texts)

            # Build FAISS index
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

            # Prompt
            prompt = PromptTemplate(template="""
            You are a helpful assistant. 
            Answer the question only using the given context. 
            If the answer is not present, say "I don't know."

            Question: {question}
            Context: {context}
            """, input_variables=["question", "context"])

            # LLM
            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                task="text-generation"
            )
            chat_model = ChatHuggingFace(llm=llm)
            parser = StrOutputParser()

            def retrival_process(q):
                q_em = model.encode(q)
                D, I = index.search(np.array([q_em]), k=5)
                context = "\n".join([texts[i] for i in I[0]])
                return context

            retrival_runnable = RunnableLambda(retrival_process)
            parallel_chain = RunnableParallel({
                "context": retrival_runnable,
                "question": RunnablePassthrough()
            })

            rag_chain = parallel_chain | prompt | chat_model | parser
            st.session_state.rag_chain = rag_chain
            st.session_state.chat_history = []

# Chat Interface
if "rag_chain" in st.session_state:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question from your document:")
        submitted = st.form_submit_button("Ask")

    if submitted and user_input:
        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.invoke(user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response))

    # Show chat history
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**🧑‍💬 You:** {message}")
        else:
            st.markdown(f"**🤖 Bot:** {message}")
