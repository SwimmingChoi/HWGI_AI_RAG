from modules.qa_chain import QAChain
from modules.config_loader import load_config
from modules.sparse_retrieval import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from modules.dense_retrieval import CustomEmbeddings
from modules.document_processor import DocumentProcessor

import os
import torch
import argparse
import gradio as gr


def main():
    
    parser = argparse.ArgumentParser(description='Enter user name for the config script')
    parser.add_argument('--config', type=str, required=False, default='config')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(current_dir, "config", f"{args.config}.yaml"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("general insurace uploading....")


    # 문서 처리
    doc_processor = DocumentProcessor()
    docs, metadata = doc_processor.load_and_process_documents(
        file_path=config['dataset']['document'],
        mode=config['dataset']['document_type'],
        do_chunking=config['preprocessing']['chunking'],
        chunk_size=config['preprocessing']['chunk_size'],
        chunk_overlap=config['preprocessing']['chunk_overlap']
    )


    # hybrid retriever
    retrieversp = BM25Retriever(docs) #tokenizer=config['retrieval']['tokenizer']
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


    # qa-chain
    qa_chain = QAChain(
        openai_config=config['openai'],
        retriever= retriever_hybrid,
        search_type=config['retrieval']['type'], 
        query_translation_type=config['query_translation']['type'],
        top_k= config['retrieval']['top_k'],
        use_reranker=config['retrieval']['reranker'],
        reranker_model=config['retrieval']['reranker_model'],
        reranker_top_k=config['retrieval']['reranker_top_k']
    )
    print("qa-chain saved")
    
    hanwha=f'''
한화손해보험 고객상담센터(1566-8000)로 문의해주세요.
<u>한화손해보험 고객상담센터 바로가기</u>
'''

    def chat(message, history):
        """
        사용자 질문을 처리하고 챗봇의 답변을 반환하는 함수
        
        Parameters:
            message (str): 사용자 질문
            history (list): 이전 대화 기록
            
        Returns:
            list: 업데이트된 대화 기록 (사용자 질문과 챗봇 답변 포함)
        """
        total_result = qa_chain.process_question(message, reset_memory=True)
        answer_result=total_result['answer']
        if '모르겠습니다' in answer_result:
            return history + [[message, answer_result+hanwha]]
        else:
            return history + [[message, answer_result]]
    
    # 대화 인터페이스
    with gr.Blocks(css="""
        .message { font-size: 1.2em !important; }
        .message-wrap { 
            max-width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .message.user, .message.bot { 
            padding: 0.5rem !important;
            margin: 0 !important;
        }
        .message-content { font-size: 1.2em !important; }
        .chat .user-message { 
            justify-content: flex-end !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .chat .user-message .message-wrap { margin-left: auto !important; }
        
        .message.bot::before {
            content: "🤖한화손해보험 챗봇";
            display: block;
            margin-bottom: 0.7rem;
            font-size: 0.6em;
            color: #666;
        }
        
        .message.user::before {
            content: "🤷고객님";
            display: block;
            margin-bottom: 0.7rem;
            font-size: 0.6em;
            color: #666;
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
