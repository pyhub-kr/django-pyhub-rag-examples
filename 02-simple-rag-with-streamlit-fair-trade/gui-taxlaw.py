import os
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
import streamlit as st

from pyhub.llm import LLM
from pyhub.rag.db.sqlite_vec import similarity_search


def main(
    db_path: Union[Path, str],
    table_name: str,
    system_prompt: str,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
):
    default_query = (
        "수출하는 경우 영세율 첨부 서류로 수출실적명세서가 없는 경우 해결 방법"
    )

    st.set_page_config(layout="wide")
    st.title("📚 RAG Demo")
    with st.container():
        search_query = st.text_input(
            "🔍 검색 내용을 입력해주세요.",
            placeholder=default_query,
        ).strip()
        search_button = st.button("유사 문서 찾기")

    if search_button:
        if not search_query:
            search_query = default_query

        st.markdown(f"### 질문과 유사한 내용의 문서")

        with st.spinner("찾는 중 ..."):
            doc_list = similarity_search(
                db_path=db_path,
                table_name=table_name,
                query=search_query,
                embedding_model="text-embedding-3-large",
                api_key=openai_api_key,
            )
        st.markdown(f"{len(doc_list)} 개의 문서를 찾았습니다.")

        if doc_list:
            for doc in doc_list:
                title = " ".join(doc.page_content.splitlines()[:3])
                title = title.replace("##", "/")

                with st.expander(title.strip()):
                    st.markdown(doc.page_content)
                    st.markdown(doc.metadata)
        else:
            st.info("No documents found matching your search criteria.")

        # 지식 + 질의를 LLM에게 전달하여 응답 생성
        지식 = str(doc_list)
        chat_llm = LLM.create(
            model="claude-3-7-sonnet-latest",
            api_key=anthropic_api_key,
            # model="gpt-4o-mini",
            # api_key=openai_api_key,
            system_prompt=system_prompt + f"\n\n<context>{지식}</context>",
            max_tokens=4000,
        )

        st.markdown(f"### LLM 응답 (모델: {chat_llm.model})")

        with st.spinner("LLM 응답 생성 중 ..."):
            # 응답을 표시할 빈 컴포넌트 생성
            response_container = st.empty()

            text = ""
            for reply in chat_llm.reply(f"Question: {search_query}", stream=True):
                text += reply.text
                # 누적된 텍스트로 컴포넌트 업데이트
                response_container.markdown(text)

    # Footer
    st.markdown("---")
    st.markdown(
        """
<div style="text-align: center; color: gray; font-size: 0.8em;">
파이썬사랑방 (<a href="mailto:me@pyhub.kr">me@pyhub.kr</a>)
</div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    load_dotenv()

    try:
        with open("system_prompt_taxlaw.txt", "rt", encoding="utf-8") as f:
            loaded_system_prompt = f.read()
    except IOError:
        loaded_system_prompt = None

    main(
        db_path="./taxlaw-sample.sqlite3",
        table_name="documents",
        system_prompt=loaded_system_prompt,
        # .env 파일에서 로드된 환경변수 사용
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
