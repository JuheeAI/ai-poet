import streamlit as st
from langchain_core.prompts import PromptTemplate # 프롬프트 템플릿 생성 클래스
from langchain_community.llms import CTransformers # 오픈소스 모델 지원
from langchain_ollama import OllamaLLM # Ollama 모델 지원

def getLLMResponse(form_input, email_sender, email_recipient, language):
  """
  getLLMResponse 함수는 주어진 입력을 사용하여 LLM으로부터 이메일 응답을 생성합니다.
  
  매개변수:
  - form_input: 사용자가 입력한 이메일 주제.
  - email_sender: 이메일을 보낸 사람의 이름.
  - email_recipient: 이메일을 받는 사람의 이름.
  - language: 이메일이 생성될 언어(한국어 또는 영어)

  반환값:
  - LLM이 생성한 이메일 응답 텍스트.
  """

  if language == '한국어':
    template = """
    {email_topic} 주제를 포함한 이메일을 작성해주세요. \n\n보낸 사람: {sender}\n받는 사람: {recipient} 전부 {language}로 번역해서 작성해주세요. 한문은 내용에서 제외해주세요.
    \n\n이메일 내용:
"""
  else:
    template = """
    Write an email including the topic of {email_topic}.\n\nSender: {sender}\nRecipient: {recipient} Please write the entire email in {language}. \n\nEmail content:"""

  prompt = PromptTemplate(
    input_variables=['email_topic', 'sender', 'recipient', 'language'],
    template=template
  )

  # llm을 사용하여 응답 생성
  response = llm.invoke(prompt.format(email_topic=form_input, sender=email_sender, recipient=email_recipient, language=language))
  print(response)

  return response

# llm = CTransformers(model='./llama-2-7b-chat.ggmlv3.q8_0.bin',
#                     model_type='llama',
#                     config={'max_new_tokens':512,
#                             'temperature':0.01,})

# Ollama 기반의 Llama 3.1 설정
llm = OllamaLLM(model="llama3.1:8b", temperature=0.7)

st.set_page_config(
  page_title="이메일 생성기",
  page_icon="📧",
  layout="centered",
  initial_sidebar_state="collapsed"
)
st.header("이메일 생성기")

language_choice = st.selectbox("이메일 언어를 선택하세요.", ("한국어", "English"))

form_input = st.text_area("이메일 주제를 입력하세요.", height=100)

col1, col2 = st.columns([10, 10])
with col1:
  email_sender = st.text_input("보낸 사람 이름을 입력하세요.")
with col2:
  email_recipient = st.text_input("받는 사람 이름을 입력하세요.")
submit = st.button("이메일 생성하기")

if submit:
  with st.spinner("이메일을 생성하는 중입니다..."):
    response = getLLMResponse(form_input, email_sender, email_recipient, language_choice)
    st.subheader("생성된 이메일:")
    st.write(response)