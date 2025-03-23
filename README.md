# 보험 약관 기반 RAG 챗봇

이 프로젝트는 **RAG(Retrieval-Augmented Generation)** 기법을 활용하여 **보험 약관 문서를 기반으로 하는 챗봇**을 구현한 것입니다. 사용자는 챗봇을 통해 보험 관련 문서의 내용을 질의응답 형식으로 쉽게 확인할 수 있습니다.

## 주요 기능

- **RAG 기반 질의응답**: 보험 약관 문서와 QnA 데이터셋을 기반으로 정보를 검색하고 답변을 생성합니다.
- **결과 평가 기능**: 생성된 응답 결과에 대한 평가 기능을 제공합니다.
- **Gradio UI**: 웹 기반 데모 인터페이스로 챗봇을 체험할 수 있습니다.

---

## 폴더 구조

```
.
├── rag_main.py              # RAG 실행 및 응답 저장
├── evaluate_results.py      # RAG 결과 평가
├── chatbot_gradio.py        # Gradio 기반 챗봇 데모 실행
├── config
   └── config.yaml           # RAG configuration 파일
└── documents/
    ├── QnA_dataset.xlsx     # QnA 질의응답 데이터셋
    └── car_document.json    # 보험 약관 문서 (자동차 보험 등)
```

> ⚠️ `config/config.yaml`, `documents/QnA_dataset.xlsx`, `documents/car_document.json` 파일은 비공개 데이터로, 배포 버전에는 포함되지 않습니다.

---

## 실행 방법

1. **RAG 실행 및 응답 저장**
   ```bash
   python rag_main.py
   ```

2. **결과 평가**
   ```bash
   python evaluate_results.py
   ```

3. **Gradio 데모 실행**
   ```bash
   python chatbot_gradio.py
   ```

실행 후 로컬 웹 페이지가 열리며 챗봇을 직접 체험할 수 있습니다.

---

## 참고 사항

- 본 프로젝트는 연구/프로토타입 목적입니다.
- OpenAI API 키가 필요하며, `config/config.yaml` 파일에 관련 설정이 포함되어 있어야 합니다.
