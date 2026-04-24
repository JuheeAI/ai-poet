from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0.1)

prompt = ChatPromptTemplate.from_messages([
  ("system", "You are a helpful assistant."),
  ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# content = "코딩"
# result = chain.invoke({"input": content + "에 대한 시를 써줘"})
# print(result)

st.title("인공지능 시인")

content = st.text_input("시의 주제를 제시해주세요.")
st.write("시의 주제는", content)

if st.button("시 작성 요청하기"):
  with st.spinner("시를 작성하는 중입니다..."):
    result = chain.invoke({'input': content + "에 대한 시를 써줘"})
    st.write(result)
