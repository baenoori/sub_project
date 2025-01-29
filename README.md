## 다중언어지원 Q&A 서비스 
주제 : 사용자가 입력한 질문과 업로드한 문서를 기반으로 답변을 생성하는 RAG 기반 다중언어지원 Q&A 서비스 설계

기간 : 2024.12.28 - 2025.01.07

인원 : 1명 (배누리)

<br>

## 프로젝트 소개
### 다중언어지원 Q&A 서비스 

<img width="787" alt="Image" src="https://github.com/user-attachments/assets/09bad039-a5e0-4a47-9723-0fdb43ac6c39" />

- 사용자가 업로드한 문서와 Hugging Face 데이터를 벡터화하여 통합 관리하고 벡터 DB 구축
- 사용자 질문을 OpenAI를 활용해 영어로 번역 후 임베딩을 생성해 벡터 DB에서 관련 데이터 검색
- 검색된 데이터를 기반으로 RAG를 활용해 답변 생성
- 생성된 답변은 OpenAI를 활용해 질문자의 원래 언어로 번역하여 출력
- HuggingFace Dataset 기반 일반 지식 Q&A 뿐만 아니라 PDF 업로드 데이터를 포함한 문서 기반 Q&A 지원

<br>

## 구현결과 

### PDF 업로드 O – Hugging face dataset + PDF 데이터 기반 답변
<img width="1320" alt="Image" src="https://github.com/user-attachments/assets/b9154862-1b01-45cb-9995-b9c0697f4754" />

### PDF 업로드 X – Hugging face dataset 기반 답변
<img width="1377" alt="Image" src="https://github.com/user-attachments/assets/ab891ebd-b330-45d1-999e-6ba0260a8f3a" />
