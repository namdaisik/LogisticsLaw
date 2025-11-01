# 물류관련법 MCP (Logistics Law RAG System)

한국의 물류 관련 법률을 RAG(Retrieval-Augmented Generation) 시스템으로 검색하고 분석할 수 있는 웹 애플리케이션입니다.

## 📋 포함된 법률

- 물류시설의 개발 및 운영에 관한 법률 (법률, 시행령, 시행규칙)
- 물류정책기본법 (법률, 시행령, 시행규칙)
- 생활물류서비스산업발전법 (법률, 시행령, 시행규칙)
- 철도사업법 (법률, 시행령, 시행규칙)
- 화물자동차 운수사업법 (법률, 시행령, 시행규칙)

## 🚀 주요 기능

- **법률 검색**: 특정 키워드나 질문으로 관련 법률 조항 검색
- **상세 분석**: 검색된 법률 조항에 대한 상세 분석 및 설명
- **웹 인터페이스**: 사용하기 쉬운 웹 기반 인터페이스

## 🛠️ 기술 스택

- **Backend**: FastAPI, Python
- **RAG**: Sentence Transformers, NumPy
- **Frontend**: HTML, CSS, JavaScript

## 📦 설치 방법

1. 저장소 클론:
```bash
git clone https://github.com/namdaisik/LogisticsLaw.git
cd LogisticsLaw
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 🎯 사용 방법

1. 웹 애플리케이션 실행:
```bash
python web_app.py
```

2. 브라우저에서 `http://localhost:8000` 접속

3. 검색창에 질문 입력 (예: "화물운송업 등록 요건은?")

## 📁 프로젝트 구조

```
물류관련법MCP/
├── web_app.py           # FastAPI 웹 애플리케이션
├── rag_core.py          # RAG 핵심 로직
├── rag_report.py        # 리포트 생성
├── split_articles.py    # 법률 문서 분할 및 인덱싱
├── requirements.txt     # Python 패키지 목록
├── templates/           # HTML 템플릿
│   └── index.html
├── article_splits/      # 법률 문서 및 인덱스
└── docs/               # 문서
```

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

## 👤 작성자

namdaisik

## 🙏 감사의 말

이 프로젝트는 한국의 물류 관련 법률 정보에 대한 접근성을 높이기 위해 개발되었습니다.
