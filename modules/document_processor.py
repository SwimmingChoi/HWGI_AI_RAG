import json
import pickle
import os
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
import logging

class DocumentProcessor:
    def load_and_process_documents(
        self,
        file_path: str,
        mode: str = "pdf",
        do_chunking: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ) -> Tuple[List[LangchainDocument], Dict]:
        """
        문서를 로드하고 Langchain 문서로 변환하여 처리하는 함수
        """
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        try:
            mode = mode.lower()
            if mode not in ["pdf", "html", "json"]:
                raise ValueError(f"지원하지 않는 문서 모드입니다: {mode}")
                
            if mode == "pdf":
                # PDF 처리
                pdf_name = os.path.basename(file_path).split('.')[0]
                pickle_path = os.path.join(current_dir, "documents", f"{pdf_name}_pymupdf4llm.pkl")
                
                with open(pickle_path, 'rb') as f:
                    documents = pickle.load(f)
                    
                if not documents:
                    logging.warning("PDF 문서가 비어있습니다.")
                    return [], {}
                    
                langchain_docs = [
                    LangchainDocument(
                        page_content=doc.get_content(),
                        metadata={"page": doc.page if hasattr(doc, 'page') else 'Unknown'}
                    )
                    for doc in documents
                ]
                
            else:
                # JSON 처리
                with open(os.path.join(current_dir, "documents", file_path), 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    
                if not json_data:
                    logging.warning("JSON 문서가 비어있습니다.")
                    return [], {}
                
                langchain_docs = [
                    LangchainDocument(
                        page_content=value,
                        metadata={"page": key}
                    )
                    for key, value in json_data.items()
                ]
            
            # Text Chunking
            if do_chunking:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                )
                processed_docs = text_splitter.split_documents(langchain_docs)
                logging.info(f"문서 청크 분할 완료: {len(processed_docs)} 청크 생성")
            else:
                processed_docs = langchain_docs
                logging.info(f"청크 분할 없이 처리 완료: {len(processed_docs)} 문서")
            
            # Metadata Mapping
            doc_metadata = {
                i: {
                    "content": doc.page_content,
                    "page": doc.metadata.get('page', 'Unknown')
                }
                for i, doc in enumerate(processed_docs)
            }
            
            logging.info(f"문서 처리 완료 (모드: {mode}, 청크 분할: {do_chunking})")
            return processed_docs, doc_metadata
            
        except Exception as e:
            logging.error(f"문서 처리 중 오류 발생: {str(e)}")
            raise