from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from typing import Optional, Union, Annotated, TypedDict

from modules.query_translation import HyDE, QueryRewrite
from modules.reranker import Reranker
from modules.gpt4o_judge import RelevanceGrader
from utils.logger import setup_logger

import json

class QAChain:
    def __init__(
        self, 
        openai_config: dict,
        retriever: str = None,
        search_type: str = "sparse",  # "bm25" 또는 "vector"
        query_translation_type: str = None,
        top_k: int = 3,
        use_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-base",
        reranker_top_k: int = 3
    ):
        """
        :param openai_config: OpenAI 설정
        :param retriever: Retriever 인스턴스
        :param search_type: 검색 방식 선택 ("sparse" 또는 "dense")
        :param top_k: 검색할 문서 수
        """
        self.search_type = search_type
        self.query_translation_type = query_translation_type
        self.llm = ChatOpenAI(
            model_name=openai_config['gpt_model'],
            api_key=openai_config['api_key'],
            max_tokens=1024,
            temperature=0.2
        )
        self.hyde = HyDE(openai_config)
        self.query_rewrite = QueryRewrite(openai_config)
        self.relevance_grader = RelevanceGrader(openai_config)
        
        # 검색 방식에 따른 retriever 설정
        if not retriever:
            raise ValueError("검색을 위해서는 retriever가 필요합니다.")

        retriever_options = {
            "sparse": lambda: retriever if use_reranker else retriever.as_retriever(search_kwargs={"k": top_k}),
            "dense": lambda: retriever.as_retriever(search_type="similarity", search_kwargs={"k": top_k}),
            "hybrid": lambda: retriever
            }

        try:
            self.active_retriever = retriever_options[search_type]()
            
        except KeyError:
            raise ValueError("search_type은 'sparse' 또는 'dense' 또는 'hybrid' 여야 합니다.")


        # Reranker 적용
        if use_reranker:
            self.active_retriever = Reranker(
                base_retriever=self.active_retriever,
                reranker_model=reranker_model,
                reranker_top_k=reranker_top_k
            ).get_retriever()


        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        
        self.qa_chain = self._create_qa_chain()
        self.logger = setup_logger()

    def _create_qa_chain(self):
        condense_question_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("assistant", "위 대화 기록을 고려하여, 다음 질문에 답변하기 위해 필요한 정보를 검색하기 위한 질문을 작성해주세요.")
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """
주어진 보험문서를 바탕으로 사용자의 질문에 답변해주세요. 
보험문서 내 답변이 존재합니다. 
보험문서를 자세히 읽고 추론하여 정확한 답변을 제공해주세요.
답변을 보험문서 내에서 추론할 수 없을 경우, '모르겠습니다.'라고 답변해주세요."""),
            ("assistant", "보험문서: {context}"),
            ("user", "{input}"),
        ])

        retriever_chain = create_history_aware_retriever(
            llm=self.llm,
            retriever=self.active_retriever,
            prompt=condense_question_prompt
        )

        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=qa_prompt,
        )

        return create_retrieval_chain(
            retriever_chain,
            document_chain,
        )
    
    def multi_step_qa(self, question: str, reset_memory: bool = False):
        """
        답변과 질문의 Relevance 평가
        Relevance 값이 True일 경우 종료
        조건 미충족 시 재시도 (n = 5까지)
        """
        n = 0
        while n < 5:
            result = self.process_question(question, reset_memory)
            relevance_score = self.relevance_grader.grade(result['question'], result['answer'])
            if relevance_score[0] == True:
                return result
            else:
                n += 1
                if n == 5:
                    self.logger.info(f"오답입니다. {relevance_score[1]} 5번 시도를 마치고 종료합니다.")
                    return result
                self.logger.info(f"오답입니다. {relevance_score[1]} 답변 생성을 다시 시도합니다. {n}번째 시도")

        
    def process_question(self, question: str, reset_memory: bool = True) -> dict:
        """
        query_translation_type에 따라 적절한 질문 처리 함수를 호출하는 메서드
        """    
        if self.query_translation_type == "HyDE":
            
            generated_docs_for_retrieval = self.hyde.translate(question)
            
            return self._ask_question_with_hyde(
                question=question,
                generated_docs_for_retrieval=generated_docs_for_retrieval,
                reset_memory=reset_memory
            )
        else:
            if self.query_translation_type == "QueryRewrite":
                question = self.query_rewrite.rewrite(question)
            else:
                question = question
            return self._ask_question(
                question=question,
                reset_memory=reset_memory
            )

    def _ask_question_with_hyde(self, question: str, generated_docs_for_retrieval: str, reset_memory: bool = False) -> dict:
        
        try:
            normalized_question = " ".join(question.strip().split())

            if reset_memory:
                self.memory.clear()

            # 이미 생성된 가상 문서를 직접 retriever에 전달
            retrieved_docs = self.active_retriever.invoke(generated_docs_for_retrieval)
            
            # QA 체인에 검색된 문서와 원래 질문을 전달
            response = self.qa_chain.invoke({
                "chat_history": self.memory.chat_memory.messages if self.memory else [],
                "input": normalized_question,
                "context": retrieved_docs
            })
            
            context_docs = response.get("context", [])
            context_pages = [
                doc.metadata.get('page', 'Unknown')
                for doc in context_docs
            ]
            
            context_pages_content = [   
                doc.page_content
                for doc in context_docs
            ]
            
            result = {
                'question': question,   
                'answer': response['answer'],
                'context_pages': context_pages,
                'context_pages_content': context_pages_content
            }
            
            if self.memory: 
                self.memory.chat_memory.add_user_message(question)
                self.memory.chat_memory.add_ai_message(response['answer'])
                
            return result
        
        except Exception as e:
            self.logger.error(f"질문 처리 중 오류 발생: {str(e)}")
            raise
    
    def _ask_question(self, question: str, reset_memory: bool = False) -> dict:
        """
        질문을 받아 답변과 참조 문서를 반환하는 함수
        """
        try:
            normalized_question = " ".join(question.strip().split())
        
            if reset_memory:
                self.memory.clear()
            
            response = self.qa_chain.invoke({
                "chat_history": self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else [],
                "input": normalized_question
            })
            
            context_docs = response.get("context", [])
            context_pages = [
                doc.metadata.get('페이지', 'Unknown') 
                for doc in context_docs
            ]
            context_pages = [int(pages) for pages in context_pages]
            
            context_pages_content = [
                doc.page_content
                for doc in context_docs
            ]

            result = {
                'question': question,
                'answer': response['answer'],
                'context_pages': context_pages,
                'context_pages_content': context_pages_content
            }
            
            
            if self.memory:
                self.memory.chat_memory.add_user_message(question)
                self.memory.chat_memory.add_ai_message(response['answer'])
                
            return result
            
        except Exception as e:
            self.logger.error(f"질문 처리 중 오류 발생: {str(e)}")
            raise
