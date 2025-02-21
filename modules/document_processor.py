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
        chunk_overlap: int = 100,
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
                
            elif mode == "json":
                # JSON 처리
                with open(os.path.join(current_dir, "documents", file_path), 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    
                if not json_data:
                    logging.warning("JSON 문서가 비어있습니다.")
                    return [], {}
                
                langchain_docs = []
                for item in json_data:
                    # 메타데이터 추출
                    content = "#" + item["Index"] + " " + item["Content"]
                    metadata = {
                        "index": item["Index"] if "Index" in item else "",
                        "page": item["Page"] if "Page" in item else "",
                        "content": content,
                        "type": item["Type"] if "Type" in item else "",
                        "article": item["Article"] if "Article" in item else "",
                        "part": item["Part"] if "Part" in item else "",
                        "chapter": item["Chapter"] if "Chapter" in item else "",
                        "section": item["Section"] if "Section" in item else "",
                        "table": item["Table"] if "Table" in item else "",
                        "table_content": item["Table_Content"] if "Table_Content" in item else "",
                        "source": file_path,
                        "document_type": "json"
                    }
                    
                    # 문서 내용과 메타데이터를 포함한 Document 객체 생성
                    doc = LangchainDocument(
                        page_content=content,
                        metadata=metadata
                    )
                    langchain_docs.append(doc)
            # Text Chunking
            if do_chunking:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                )
                processed_docs = text_splitter.split_documents(langchain_docs)
                
                # 청크에 대한 메타데이터 추가
                for i, doc in enumerate(processed_docs):
                    doc.metadata.update({
                        "chunk_index": i,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "total_chunks": len(processed_docs)
                    })
                
                logging.info(f"문서 청크 분할 완료: {len(processed_docs)} 청크 생성")
            else:
                processed_docs = langchain_docs
                logging.info(f"청크 분할 없이 처리 완료: {len(processed_docs)} 문서")
            
            # Metadata Mapping
            doc_metadata = {
                i: {
                    "index": doc.metadata.get("index", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "content": doc.page_content,
                    "type": doc.metadata.get("type", "Unknown"),
                    "article": doc.metadata.get("article", "Unknown"),
                    "part": doc.metadata.get("part", "Unknown"),
                    "chapter": doc.metadata.get("chapter", "Unknown"),
                    "section": doc.metadata.get("section", "Unknown"),
                    "table": doc.metadata.get("table", "Unknown"),
                    "table_content": doc.metadata.get("table_content", "Unknown"),
                    
                }
                for i, doc in enumerate(processed_docs)
            }
            
            logging.info(f"문서 처리 완료 (모드: {mode}, 청크 분할: {do_chunking})")
            return processed_docs, doc_metadata
            
        except Exception as e:
            logging.error(f"문서 처리 중 오류 발생: {str(e)}")
            raise