# 물류법 RAG 시스템 개요

## 1. 목적
이 프로젝트는 폴더 내 물류 관련 법령 PDF를 자동으로 전처리하고, 검색 질의에 가장 적합한 조항을 찾아 인용과 함께 보고서를 생성하는 Retrieval-Augmented Generation(RAG) 파이프라인을 제공합니다. 사용자는 CLI 또는 FastAPI 기반 웹 인터페이스를 통해 한국어 질의를 입력하고, 인덱싱된 법령 조항으로 구성된 답변을 받을 수 있습니다.

## 2. 주요 구성 요소
- `split_articles.py` : 원본 PDF를 조항 단위로 분할하여 `article_splits/법령명/00X_제Y조.pdf` 구조를 생성합니다.
- `rag_core.py` : 인덱싱·검색·재랭킹·LLM 호출을 담당하는 코어 모듈입니다.
- `rag_report.py` : CLI 앱으로, 단일 질의에 대한 보고서를 콘솔에 출력합니다.
- `web_app.py` : FastAPI 서버. 세션 기반 대화 히스토리를 관리하며, 관련 PDF 링크를 UI에 제공합니다.
- `templates/index.html` : 웹 UI 템플릿. 질의 입력, 응답, 근거 문서를 한 화면에 제공합니다.

## 3. 처리 파이프라인
### 3.1 데이터 준비
1. `split_articles.py --output article_splits` 실행 시 폴더 내 모든 PDF를 스캔합니다.
2. 조항 머리글(`제X조`, `제X조의Y` 패턴)을 인식하여 페이지 범위를 추출하고, 각 조항을 개별 PDF로 저장합니다.
3. 파일명에는 정렬을 위한 3자리 인덱스와 조항명이 포함됩니다.

### 3.2 인덱싱
1. `rag_core.build_or_load_index`가 `article_splits` 이하 PDF를 재귀 탐색합니다.
2. `infer_law_metadata`가 경로에서 법령군(법률·시행령·시행규칙)을 파악하고, `LAW_GRAPH` 정보를 이용해 태그(동의어/연관 법령/키워드)를 부여합니다.
3. 페이지 텍스트를 추출 후 1,500자 단위로 슬라이딩 윈도우를 적용하여 문단 기반 청크를 생성합니다.
4. OpenAI `text-embedding-3-small` 모델로 청크를 벡터화하여 `mcp_embeddings.npy`에 저장하고, 인덱스 메타데이터(`mcp_index.json`)에는 청크 정보와 임베딩 인덱스만 기록합니다.
5. 파일 해시와 인덱스 버전을 저장해 변경 여부를 감지하며, 재실행 시 변경이 없으면 즉시 캐시를 재사용합니다.

### 3.3 질의 처리
1. 사용자의 자연어 질의를 임베딩하여 코사인 유사도로 상위 후보(최소 20개)를 검색합니다.
2. `detect_query_tags`가 질의 내 법령명·동의어를 탐지해 물류법 그래프에서 관련 태그를 수집합니다.
3. `rerank_contexts`가 태그 일치도와 동일 법령군 여부를 기반으로 점수를 재조정하여, 질의와 연관성이 높은 시행령/시행규칙 후보가 상위에 오도록 만듭니다.
4. 최종 Top-K(기본 6~12개) 청크가 OpenAI `gpt-4o-mini` 모델에 컨텍스트로 전달되어, 개요 → 핵심 조항 → 시사점 구조의 한국어 보고서를 생성합니다.
5. 응답에는 각 주장 뒤 `[문서, 페이지]` 형태의 출처가 포함됩니다.

## 4. 법령 그래프 기반 재랭킹
`LAW_GRAPH`는 법률 이름을 노드로 두고, 시행령·시행규칙·일반적으로 쓰이는 키워드를 에지(동의어/연관어)로 연결한 작은 지식 그래프입니다. 질의가 지입·화물차 등 특정 도메인 키워드를 포함하면:
- 해당 노드(예: `화물자동차 운수사업법`)가 활성화됩니다.
- 연관 시행령/시행규칙 청크는 +0.3 가중치, 태그가 다수 일치하면 추가 가중치를 받아 검색 결과 상위로 이동합니다.
- 반대로 관련법이 이미 존재하는데 연관성이 낮은 항목은 -0.1 감점을 받아 노이즈가 줄어듭니다.

이 단순 그래프 기반 재랭킹은 딥러닝 모델을 추가 학습하지 않고도 도메인별 법령군을 묶어주는 효과가 있어, “지입제 양도 양수 서류” 같은 질의가 자연스럽게 화물자동차 운수사업법 계열 결과로 연결됩니다.

## 5. 웹 애플리케이션
- FastAPI + Jinja2로 구현되며, `sessions` 딕셔너리에 세션별 히스토리를 저장합니다.
- 각 턴에는 사용자 질의, 모델 응답, 참고 문서 목록이 포함되며, PDF 정적 파일 `/pdf/경로`로 바로 이동할 수 있습니다.
- `.env`나 PowerShell 세션에서 `OPENAI_API_KEY`가 설정되어 있어야 하며, `uvicorn web_app:app --reload` 명령으로 실행합니다.

## 6. 성능 및 캐싱 전략
- 임베딩 행렬을 `.npy` 파일로 분리 저장하고, 프로세스 내 `EMBED_MATRIX_CACHE`에서 재사용하여 질의당 디스크 I/O 및 JSON 파싱 부담을 줄였습니다.
- `split_articles.py`로 조항 분할을 선행하면 각 파일이 수 페이지 이내에 머물러 추출 속도가 빨라지고, 검색 결과가 문서 단위보다 조항 단위로 세밀해집니다.
- 재인덱싱 시에는 `--rebuild_index` 플래그를 제공하여 필요할 때만 임베딩을 새로 계산합니다.

## 7. 실행 절차 요약
1. 가상환경 준비
   ```bash
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. PDF 조항 분할
   ```bash
   python split_articles.py --output article_splits
   ```
3. CLI 보고서 생성
   ```bash
   python rag_report.py --root article_splits --query "지입제 양도 양수 서류" --top_k 6
   ```
4. 웹 서버 실행
   ```bash
   uvicorn web_app:app --reload --port 8000
   ```
5. OpenAI 키는 환경변수 `OPENAI_API_KEY`에 설정해야 합니다(예: PowerShell에서 `$env:OPENAI_API_KEY="sk-..."`).

## 8. 참고 논문 및 문헌
- Patrick Lewis et al., **Retrieval-Augmented Generation for Knowledge-Intensive NLP**, NeurIPS 2020.
- Vladimir Karpukhin et al., **Dense Passage Retrieval for Open-Domain Question Answering**, EMNLP 2020.
- Gautier Izacard & Édouard Grave, **Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering**, arXiv 2020.
- Ikuya Yamada et al., **Efficient Passage Retrieval with Hashing for Open-domain Question Answering**, ACL 2021.
- Kedar Tata et al., **Graph-augmented Dense Retrieval for Open-Domain Question Answering**, Findings of ACL 2022.
- Jae-Hyun Seo et al., **Domain-specific Legal Information Retrieval using Heterogeneous Graphs**, Journal of Information Science Theory and Practice 2021.

위 논문들은 RAG 구조, 밀집 벡터 검색, 그래프 기반 재랭킹 등 이 프로젝트가 활용한 핵심 개념을 다루고 있으며, 특히 법령/도메인 특화 그래프 적용 아이디어는 마지막 논문과 유사한 흐름을 따릅니다.

