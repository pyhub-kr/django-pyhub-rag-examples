# Streamlit을 활용한 공정거래법 법령 유사 문서 검색

SQLite 데이터베이스를 Vector Store로서 활용했으며,
SQLite 데이터베이스에 저장된 공정거래법 법령 문서를 검색할 수 있는 Streamlit 애플리케이션입니다.

문서 임베딩과 데이터베이스 생성은 `django-pyhub-rag` 라이브러리 기반으로
장고를 통해 수행했습니다.

데이터베이스 생성 과정이 궁금하신 분은 [장고로 만드는 RAG 웹 채팅 서비스](https://ai.pyhub.kr/hands-on-lab/django-webchat-rag/)
문서를 참고하세요.

