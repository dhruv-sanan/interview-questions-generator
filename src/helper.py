from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from src.prompt import *
import os
from dotenv import load_dotenv


# OpenAI authentication
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def file_processing(file_path):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    pdf_data = ''

    for page in data:
        pdf_data += page.page_content

    splitter = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size=5000,
        chunk_overlap=50,
    )

    chunks_ques_gen = splitter.split_text(pdf_data)

    document_ques_gen = splitter.create_documents(chunks_ques_gen)

    new_splitter = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size=1000,
        chunk_overlap=10,
    )

    document_answer_gen = new_splitter.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path):

    document_ques_gen, document_answer_gen = file_processing(file_path)

    llm_ques_gen_pipeline = ChatOpenAI(
        temperature=0.9,  # high temperature means more creativity
        model="gpt-3.5-turbo",
    )

    prompt_ques = PromptTemplate(
        template=prompt_template, input_variables=["text"])
    # recommended by langchain
    prompt = PromptTemplate.from_template(prompt_template)

    refined_prompt_ques = PromptTemplate(
        template=refine_template, input_variables=["text", "existing_answer"])
    refined_prompt = PromptTemplate.from_template(refine_template)

    ques_gen_chain = load_summarize_chain(llm=llm_ques_gen_pipeline,
                                          chain_type="refine",
                                          verbose=True,
                                          question_prompt=prompt_ques,
                                          refine_prompt=refined_prompt_ques)

    ques = ques_gen_chain.run(document_ques_gen)

    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    llm_answer_gen = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith(
        '?') or element.endswith('.')]

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen,
                                                          chain_type="stuff",
                                                          retriever=vector_store.as_retriever())

    return answer_generation_chain, filtered_ques_list
