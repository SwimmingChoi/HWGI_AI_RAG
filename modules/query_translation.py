from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class QueryTranslation:
    def __init__(self, config: dict):
        self.config = config
        self.model = ChatOpenAI(model=config['openai']['gpt_model'], temperature=0, api_key=config['openai']['api_key'])

    def translate(self, question: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "해당 질문에 답하기 위한 보험 약관 단락을 작성해주세요. \n 질문: {question} \n 단락:"
        )
        chain = prompt | self.model | StrOutputParser()
        generated_docs_for_retrieval = chain.invoke({"question": question})
        print(generated_docs_for_retrieval)
        return generated_docs_for_retrieval
