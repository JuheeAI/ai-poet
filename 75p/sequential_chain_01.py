from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai = ChatOpenAI(model="gpt-5.4-nano", temperature=0.7, api_key=OPENAI_API_KEY)

# 프롬프트 템플릿 정의
# prompt1 = 리뷰를 한 문장으로 요약
# prompt2 = 리뷰를 바탕으로 0~10점 사이 점수 매김
# prompt3 = 요약된 리뷰에 대해 공손한 답변 작성

prompt1 = PromptTemplate(
  input_variables=['review'],
  template="다음 숙박 시설 리뷰를 한글로 번역하세요. \n\n{review}"
)

prompt2 = PromptTemplate.from_template(
  "다음 숙박 시설 리뷰를 한 문장으로 요약하세요.\n\n{translation}"
)

prompt3 = PromptTemplate.from_template(
  "다음 숙박 시설 리뷰를 읽고 0점부터 10점 사이에서 부정/긍정 점수를 매기세요. 숫자만 대답하세요. \n\n{translation}"
)

prompt4 = PromptTemplate.from_template(
  "다음 숙박 시설 리뷰에 사용된 언어가 무엇인가요? 언어 이름만 답하세요. \n\n{review}"
)

prompt5 = PromptTemplate.from_template(
  "다음 숙박 시설 리뷰 요약에 대해 공손한 답변을 작성하세요. \n답변 언어:{language}\n리뷰 요약:{summary}"
)

prompt6 = PromptTemplate.from_template(
  "다음 생성된 답변을 한국어로 번역해주세요. \n 리뷰 번역 {reply1}"
)

# LCEL의 RunnablePassthrough.assign()을 사용하여 체인 컴포넌트 생성
translate_chain_component = prompt1 | openai | StrOutputParser() # 번역
summarize_chain_component = prompt2 | openai | StrOutputParser() # 요약
sentiment_score_chain_component = prompt3 | openai | StrOutputParser() # 감정 점수
language_chain_component = prompt4 | openai | StrOutputParser() # 언어 감지
reply1_chain_component = prompt5 | openai | StrOutputParser() # 답변 생성
reply2_chain_component = prompt6 | openai | StrOutputParser() # 답변 번역

combined_lcel_chain = (
  RunnablePassthrough.assign(
    translation=lambda x: translate_chain_component.invoke({"review": x["review"]}),
  )
  |
  RunnablePassthrough.assign(
    summary=lambda x: summarize_chain_component.invoke({"translation": x["translation"]}),
    sentiment_score=lambda x: sentiment_score_chain_component.invoke({"translation": x["translation"]}),
    language=lambda x: language_chain_component.invoke({"review": x["review"]})
  )
  |
  RunnablePassthrough.assign(
    reply1=lambda x: reply1_chain_component.invoke({"language": x["language"], "summary": x["summary"]})
  )
  |
  RunnablePassthrough.assign(
    reply2=lambda x: reply2_chain_component.invoke({"reply1": x["reply1"]})
  )
)

review_text = """
The hotel was clean and the staff were very helpful.
The location was convenient, close to many attractions.
However, the room was a bit small and the breakfast options were limited.
Overall, a decent stay but there is room for improvement.
"""

try:
  result = combined_lcel_chain.invoke(input={"review": review_text})
  print(f'translation 결과: {result.get("translation", "N/A")} \n')
  print(f'summary 결과: {result.get("summary", "N/A")} \n')
  print(f'sentiment_score 결과: {result.get("sentiment_score", "N/A")} \n')
  print(f'language 결과: {result.get("language", "N/A")} \n')
  print(f'reply1 결과: {result.get("reply1", "N/A")} \n')
  print(f'reply2 결과: {result.get("reply2", "N/A")} \n')
except Exception as e:
  print(f"Error: {e}")