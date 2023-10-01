from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS, Chroma
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain import LLMChain
import os
import time
import re


def answer(query: list[Document], llm: LLMChain) -> str:
    answer_template = """
        Transcript:
        ------------
        {text}
        ------------
        You are an expert in extracting video content information.
        Your task is to extract the following information from the given video text information:
                1. title
                2. author
                3. link
                4. description
                5. categories of math problems
        And output the extracted information in the specified order.
        Formatted as JSON
    """

    PROMPT_ANSWER = PromptTemplate(template=answer_template, input_variables=["text"])
    # Initialize the retrieval chain
    answer_chain = LLMChain(llm=llm, prompt=PROMPT_ANSWER)
    # chain中优化输出
    res = answer_chain.predict(text=str(query))
    return res


def vector_store(embeddings: OpenAIEmbeddings) -> Chroma:
    store = Chroma(persist_directory="/content/chroma_vector", embedding_function=embeddings)
    return store


def extract_chain(llm: LLMChain) -> LLMChain:
    extract_skills_template = """
        Transcript:
        ------------
        {text}
        ------------
        You are an expert in extracting the topics taught in math problem-solving videos.
        Your task is to extract the following information from the given video text information:
            1. Categories of math problems
        Output the extracted information.
    """
    PROMPT_EXTRACT_SKILLS = PromptTemplate(template=extract_skills_template, input_variables=["text"])
    return LLMChain(llm=llm, prompt=PROMPT_EXTRACT_SKILLS)


def extract_skills(query: str) -> str:
    res = extract_chain(query)
    return res


def skills(query: str) -> str:
    videos = extract_skills(query)
    extracted_skills = extract_chain.predict(text=videos[0].page_content)
    return extracted_skills


def output_chain(llm: LLMChain) -> LLMChain:
    output_template = """
        Transcript:
        ------------
        {text}
        ------------
        You are an expert in extracting information from the video descriptions of mathematical problem-solving videos.
        The input text contains information summaries for three videos.
        Your task is to extract the following information for each of the three videos from the input text:
                1. title
                2. author
                3. link
                4. description
                5. categories of math problems
        Please provide the outputs in the order of the videos.
        Formatted as JSON
    """
    PROMPT_OUTPUT = PromptTemplate(template=output_template, input_variables=["text"])
    chain = LLMChain(llm=llm, prompt=PROMPT_OUTPUT)
    return chain


# def bot(extracted_skills: str, output_chain=output_chain(), llm : LLMChain) -> None:
#     keywords = [
#         "Operations and Algebraic Thinking",
#         "Number and Operations—Fractions",
#         "Number and Operations in Base Ten",
#         "Measurement and Data",
#         "Geometry"
#     ]
#
#     sentence = str(extracted_skills)
#     for keyword in keywords:
#         if re.search(re.escape(keyword), sentence):
#             time.sleep(20)
#             videos = vector_store.similarity_search(keyword)
#             print()
#             print(f"{keyword}相关问题链接: ")
#             vedios3 = output_chain.predict(
#                 text=videos[0].page_content + videos[1].page_content + videos[2].page_content)
#             print(vedios3)
#             time.sleep(20)
