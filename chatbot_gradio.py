# [2팀] 2팀 코드 - Gradio 챗봇 코드

from modules.qa_chain import QAChain
from modules.config_loader import load_config
from modules.sparse_retrieval import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from modules.dense_retrieval import CustomEmbeddings
from modules.document_processor import DocumentProcessor
from utils.logger import setup_logger
from modules.faq import FAQ

import os
import torch
import argparse
import gradio as gr
import logging
import json

# [2팀] Gradio 챗봇 실행 (사용자의 질문에 대해 답변을 제공합니다. 로직은 rag_main.py와 동일합니다.)
def main():
    
    parser = argparse.ArgumentParser(description='Enter user name for the config script')
    parser.add_argument('--config', type=str, required=False, default='config')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(current_dir, "config", f"{args.config}.yaml"))
    logger = setup_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("general insurace uploading....")


    # [2팀] STEP 1: 입력 document 처리
    doc_processor = DocumentProcessor()
    docs, metadata = doc_processor.load_and_process_documents(
        file_path=config['dataset']['document'],
        mode=config['dataset']['document_type'],
        do_chunking=config['preprocessing']['chunking'],
        chunk_size=config['preprocessing']['chunk_size'],
        chunk_overlap=config['preprocessing']['chunk_overlap']
    )

    # [2팀] STEP 2: FAQ 데이터셋 (json파일) 내에서 질문들만 추출하여 question_dictionary 리스트에 저장
    # → 이후 사용자의 질문과 FAQ 데이터셋 내 질문을 비교하기 위해 사용
    faq_data_path=os.path.join(current_dir, "documents", config['dataset']['faq'])
    with open(faq_data_path, 'r') as f:
        faq_data = json.load(f)
    question_dictionary=[]
    for i in range(len(faq_data)):
        question_dictionary.append(faq_data[i]['original_question'])
        
        
    # [2팀] STEP 3: Hybrid Retriever 설정
    retrieversp = BM25Retriever(docs) 
    faiss_index_path = config['dataset']['document'].replace(
        '.json',
        f"_{config['dense_model']['model_type']}_" +
        ('chunk.bin' if config['preprocessing']['chunking'] else 'full.bin')
    )
    embedding_service = CustomEmbeddings(
        model_type=config['dense_model']['model_type'],
        model_path=config['dense_model']['model_cache'],
        device=device,
        embedding_config=config,
        faiss_index_path=faiss_index_path,
    )
    retrieverde = embedding_service.create_or_load_vectorstore(docs)
    retrieverde = retrieverde.as_retriever(search_kwargs={
        "k": config['retrieval']['top_k'],
        "search_type": config['retrieval']['search_type']
    })
    retriever_hybrid = EnsembleRetriever(
        retrievers=[retrieversp, retrieverde],
        weights=[0.4, 0.6])
    print("retrieval type saved")


    # [2팀] STEP 4: QA chain 설정
    qa_chain = QAChain(
        openai_config=config['openai'],
        retriever= retriever_hybrid,
        search_type=config['retrieval']['type'], 
        query_translation_type=config['query_translation']['type'],
        top_k= config['retrieval']['top_k'],
        use_reranker=config['retrieval']['reranker'],
        reranker_model=config['retrieval']['reranker_model'],
        reranker_top_k=config['retrieval']['reranker_top_k'],
        num = config['multi_step_iteration']['num']
    )
    print("qa-chain saved")
    
    
     # [2팀] STEP 5: 고객상담센터 연락처 저장
     # →'모르겠습니다' 답변 생성 경우를 대비하여 준비하였음
    hanwha=f'''
한화손해보험 고객상담센터(1566-8000)로 문의해주세요.
<u>한화손해보험 고객상담센터 바로가기</u>
'''

    def chat(message, history):
        logger.info(f"사용자가 질문을 입력했습니다.")
        """
        [2팀] 사용자 질문을 처리하고 챗봇의 답변을 반환하는 함수
        
        Parameters:
            message (str): 사용자 질문
            history (list): 이전 대화 기록
            
        Returns:
            list: 업데이트된 대화 기록 (사용자 질문과 챗봇 답변 포함)
        """
        
        # [2팀] STEP 6: FAQ 데이터셋 질문들 중 사용자 입력 질문과 의미적으로 동일한 질문이 있는지 확인
        infaq=FAQ(api_key=config['openai']['api_key'])
        answer_faq = infaq.run_faq(question=message, question_dictionary=question_dictionary, faq_data=faq_data)
        # [2팀] STEP 7: FAQ 데이터셋 내 사용자의 질문과 의미적으로 동일한 질문이 있는 경우 답변 생성
        if answer_faq!=False:
            logger.info(f"FAQ 데이터셋을 활용하여 답변 생성을 완료했습니다.")
            return history + [[message, answer_faq]]
        
        # [2팀] STEP 8: FAQ 데이터셋 내 사용자의 질문과 의미적으로 동일한 질문이 없는 경우 RAG 실행
        else:
            total_result = qa_chain.multi_step_qa(message, reset_memory=True)
            answer_result=total_result['answer']
            logger.info(f"RAG 실행 후, 답변 생성을 완료했습니다.")        
        # [2팀] STEP9: 답변 생성
        # →'모르겠습니다'가 답변 내에 존재하는 경우, 고객상담센터 연락처도 제공
        # →'모르겠습니다'가 답변 내에 존재하지 않는 경우, 생성된 답변만 제공
        # → 질문과 답변을 history로 저장
            if '모르겠습니다' in answer_result:
                return history + [[message, answer_result+hanwha]]
            else:
                return history + [[message, answer_result]]
    
    # [2팀] Gradio 인터페이스
    with gr.Blocks(css="""
.message {
  font-size: 1.2em !important;
}

.message-content {
  font-size: 1.2em !important;
}

.message-wrap {
  max-width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
}

/* 사용자와 봇 메시지 공통 스타일 */
.message.user, 
.message.bot {
  padding: 0.5rem !important;
  margin: 0 !important;
}

/* 사용자 메시지 정렬 */
.chat .user-message {
  justify-content: flex-end !important;
  padding: 0 !important;
  margin: 0 !important;
}

.chat .user-message .message-wrap {
  margin-left: auto !important;
}

/* 메시지 레이블 공통 스타일 */
.message.bot::before,
.message.user::before {
  display: block;
  margin-bottom: 0.7rem;
  font-size: 0.6em;
  color: #666;
}

/* 봇 메시지 레이블 */
.message.bot::before {
  content: "🤖한화손해보험 챗봇";
}

/* 사용자 메시지 레이블 */
.message.user::before {
  content: "🤷고객님";
  text-align: right;
}
    """) as demo:
        gr.Markdown("<div align='center'>\n\n# 🚗 한화 자동차 보험\n\n</div>")
        gr.Markdown("<div align='center'>\n\n## 🤖안녕하세요. 한화 자동차 보험 챗봇입니다. 오늘도 친절하게 답변드리겠습니다!\n\n</div>")
        
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(label="궁금하신 점이 있다면 언제든 편하게 여쭤보세요.", placeholder="여기에 메시지를 입력하세요...")
        clear = gr.Button("대화내용 전체 삭제")
        
        msg.submit(chat, [msg, chatbot], [chatbot]).then(
            lambda: "", None, [msg]
        )
        
        clear.click(lambda: None, None, chatbot, queue=False)
    
    demo.launch()

if __name__ == "__main__":
    main()
