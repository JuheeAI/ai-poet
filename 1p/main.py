from langchain_openai import ChatOpenAI # OpenAI API와 상호작용하는 LLM 클래스
from langchain_core.prompts import ChatPromptTemplate # 프롬프트 템플릿 생성 클래스
from langchain_core.output_parsers import StrOutputParser # 출력 파서 클래스(깔끔하게 추출하는 용도)
from dotenv import load_dotenv # .env 파일에서 환경변수 불러오기
load_dotenv() 

llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0.1, max_tokens=5)
# result = llm.invoke("hello")
# print(result)

prompt = ChatPromptTemplate.from_messages([ # 프롬프트 템플릿 생성
  ("system", "You are a helpful assistant."),
  ("user", "{input}")
])

# chain = prompt | llm # 체인으로 연결
# result = chain.invoke({"input": "hi"})
# print(result)

output_parser = StrOutputParser() # 출력 파서 생성

chain = prompt | llm | output_parser # 체인으로 연결
result = chain.invoke({"input": "hi"})
print(result)