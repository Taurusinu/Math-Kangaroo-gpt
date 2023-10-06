import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
import submission.chains as chains
import submission.getImg as getIMG
import re
import time
import os
import json

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.set_page_config(page_title="📚 📐 Math-Kangaroo")

openai_api_key = st.sidebar.text_input('OpenAI API Key', type="password")
os.environ["OPENAI_API_KEY"] = openai_api_key

st.title('📚 📐 Math-Kangaroo')


def generate_response(input_text):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0)
    embeddings = OpenAIEmbeddings(request_timeout=600)


    vector_store = Chroma(persist_directory="chroma_vector", embedding_function=embeddings)
    vc = vector_store.similarity_search(input_text)
    mathProblems = chains.answer(vc[0].page_content, llm)
    data = json.loads(mathProblems)
    st.title("Solution to your question :bookmark_tabs:")
    c1, c2 = st.columns(2,gap="large")
    vedioURL = getIMG.get_youtube_thumbnail(data["link"])
    with c1:
        st.markdown(f'<a href="{data["link"]}"><img src="{vedioURL}" width ="500" height="400" alt="centered image"></a>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''
                        <section class="feature-post solution">
                          <div class="post-content">
                            <h2>{data["title"]}</h2>
                            <p>{data["author"]}</p>
                            <p>{data["description"]}</p>
                          </div>
                        </section>
                      ''', unsafe_allow_html=True)


    extracted_skills = chains.extract_chain(llm)

    keywords = [
        "Operations and Algebraic Thinking",
        "Number and Operations—Fractions",
        "Number and Operations in Base Ten",
        "Measurement and Data",
        "Geometry"
    ]

    st.title("Recommended videos on related topics :speech_balloon:")
    sentence = mathProblems
    for keyword in keywords:
        if re.search(re.escape(keyword), sentence):
            time.sleep(20)
            videos = vector_store.similarity_search(keyword)
            st.info(f"{keyword}related problems link: ")
            output_chain = chains.output_chain(llm)
            vedios3 = output_chain.predict(
                text=videos[0].page_content + videos[1].page_content + videos[2].page_content)
            data = json.loads(vedios3)

            col1, col2, col3 = st.columns(3, gap="large")
            with col1:
                img1URL = getIMG.get_youtube_thumbnail(data["videos"][0]["link"])
                st.markdown(f'<a href="{data["videos"][0]["link"]}"><img src="{img1URL}" width ="300" height="200" alt="centered image"></a>',
                            unsafe_allow_html=True)
                st.markdown(f'''
                        <section class="feature-post solution">
                          <div class="post-content">
                            <h2>{data["videos"][0]["title"]}</h2>
                            <p>{data["videos"][0]["author"]}</p>
                            <p>{data["videos"][0]["description"]}</p> 
                          </div>
                        </section>
                      ''', unsafe_allow_html=True)
            with col2:
                img2URL = getIMG.get_youtube_thumbnail(data["videos"][1]["link"])
                st.markdown(
                    f'<a href="{data["videos"][1]["link"]}"><img src="{img2URL}" width ="300" height="200" alt="centered image"></a>',
                    unsafe_allow_html=True)
                st.markdown(f'''
                        <section class="feature-post solution">
                            <div class="post-content">
                                <h2>{data["videos"][1]["title"]}</h2>
                                <p>{data["videos"][1]["author"]}</p>
                                <p>{data["videos"][1]["description"]}</p>
                            </div>
                        </section>
                        ''', unsafe_allow_html=True)
            with col3:
                img3URL = getIMG.get_youtube_thumbnail(data["videos"][2]["link"])
                st.markdown(
                    f'<a href="{data["videos"][2]["link"]}"><img src="{img3URL}" width ="300" height="200" alt="centered image"></a>',
                    unsafe_allow_html=True)
                st.markdown(f'''
                            <section class="feature-post solution">
                              <div class="post-content">
                                <h2>{data["videos"][2]["title"]}</h2>
                                <p>{data["videos"][2]["author"]}</p>
                                <p>{data["videos"][2]["description"]}</p>
                              </div>
                            </section>
                          ''', unsafe_allow_html=True)


with st.form('my_form'):
    text = st.text_area('Enter text:', 'Please input your math problem here.')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
if submitted and openai_api_key.startswith('sk-'):
    generate_response(text)
