import json
import os
import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Generator, List, Literal, TypeAlias, Union

import openai
import sqlite_vec
import streamlit as st
from dotenv import load_dotenv
from openai.types import ChatModel as OpenAIChatModel
from typing_extensions import Optional

OpenAIEmbeddingModel: TypeAlias = Union[
    Literal[
        "text-embedding-3-small",  # 1536 차원
        "text-embedding-3-large",  # 3072 차원
    ],
    str,
]

LLMChatModel: TypeAlias = Union[str, OpenAIChatModel]


# https://platform.openai.com/docs/pricing#embeddings
def get_embedding_price(
    model: OpenAIEmbeddingModel, tokens: int
) -> tuple[Decimal, Decimal]:
    # 2025년 3월 기준
    price_per_1m = {
        "text-embedding-3-small": Decimal("0.02"),
        "text-embedding-3-large": Decimal("0.13"),
    }[model]
    usd = (Decimal(tokens) * price_per_1m) / Decimal("1000000")
    krw = usd * Decimal("1500")
    return usd, krw


# https://platform.openai.com/docs/pricing#latest-models
def get_chat_price(
    model: LLMChatModel, input_tokens: int, output_tokens: int
) -> tuple[Decimal, Decimal]:
    input_price_per_1m, output_price_per_1m = {
        "gpt-4o": (Decimal("2.5"), Decimal("10.0")),
        "gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
        "o1": (Decimal("15"), Decimal("60.00")),
        "o3-mini": (Decimal("1.10"), Decimal("4.40")),
        "o1-mini": (Decimal("1.10"), Decimal("4.40")),
    }[model]

    input_usd = (Decimal(input_tokens) * input_price_per_1m) / Decimal("1000000")
    output_usd = (Decimal(output_tokens) * output_price_per_1m) / Decimal("1000000")
    usd = input_usd + output_usd

    input_krw = input_usd * Decimal("1500")
    output_krw = output_usd * Decimal("1500")
    krw = input_krw + output_krw

    return usd, krw


st.set_page_config(
    page_title="simple rag with streamlit (Powered by django-pyhub-rag)",
    page_icon="🔍",
    layout="wide",
)

st.title("📚 세법 해석례 유사 문서 검색")
st.markdown(
    """
    파이썬사랑방 [장고로 만드는 RAG 웹 채팅 서비스](https://ai.pyhub.kr/hands-on-lab/django-webchat-rag/) 튜토리얼을 통해
    생성된 sqlite 데이터베이스를 활용한 유사 문서 검색 서비스입니다.
    (참고: [국세법령정보시스템](https://taxlaw.nts.go.kr/)에는
    [13만 건이 넘는 세법해석례 질답 데이터](https://taxlaw.nts.go.kr/qt/USEQTJ001M.do)가 있습니다.)
    """
)


@dataclass
class Document:
    id: int
    page_content: str
    metadata: dict
    distance: float

    def __post_init__(self):
        """초기화 후 실행되는 메서드로 필요한 속성들을 설정합니다."""
        # page_content에서 필요한 속성들을 파싱하여 할당
        obj = json.loads(self.page_content)
        self.문서ID = obj["문서ID"]
        self.제목 = obj["제목"]
        self.문서번호 = obj["문서번호"]
        self.법령분류 = obj["법령분류"]
        self.요지 = obj["요지"]
        self.회신 = obj["회신"]
        self.파일내용 = obj["파일내용"]
        self.공개여부 = obj["공개여부"]
        self.문서분류 = obj["문서분류"]
        self.생성일시 = obj["생성일시"]
        self.수정일시 = obj["수정일시"]
        self.url = self.metadata.get("url", None)


class Rag:
    def __init__(
        self,
        db_path: str,
        table_name: str,
        embedding_model: OpenAIEmbeddingModel,
        chat_model: LLMChatModel,
        openai_api_key: Optional[str] = None,
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.openai_api_key = openai_api_key

    def embed(self, input: str) -> tuple[list[float], int]:
        client = openai.Client(api_key=self.openai_api_key)
        response = client.embeddings.create(
            input=input,
            model=self.embedding_model,
        )
        return response.data[0].embedding, response.usage.prompt_tokens

    def make_reply(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> Generator[str, None, None]:
        text_output = ""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = openai.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        # 생성 결과를 실시간 스트리밍하면서 누적 처리
        usage = None
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                text_output += chunk.choices[0].delta.content
                yield text_output

            if chunk.usage:
                usage = chunk.usage

        if usage:
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            usd, krw = get_chat_price(self.chat_model, input_tokens, output_tokens)
            text_output += f"\n\n(입력 토큰: {input_tokens}, 출력 토큰: {output_tokens}, 비용: ${usd}, ₩{krw})"
            yield text_output

    def similarity_search(
        self, embedding_vector: list[float], k: int = 4
    ) -> List[Document]:
        with sqlite3.connect(self.db_path) as conn:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)

            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, page_content, metadata, distance
                FROM {self.table_name}
                WHERE (embedding MATCH vec_f32(?))
                ORDER BY distance
                LIMIT {k}
                """,
                [str(embedding_vector)],
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    Document(
                        id=row[0],
                        page_content=row[1],
                        metadata=json.loads(row[2]),
                        distance=row[3],
                    )
                )

        return results


def main(
    db_path: Union[Path, str],
    table_name: str,
    embedding_model: OpenAIEmbeddingModel,
    chat_model: LLMChatModel,
    system_prompt: str,
    openai_api_key: Optional[str] = None,
):
    rag = Rag(
        db_path=db_path,
        table_name=table_name,
        embedding_model=embedding_model,
        chat_model=chat_model,
        openai_api_key=openai_api_key,
    )

    default_query = (
        "재화 수출하는 경우 영세율 첨부 서류로 수출실적명세서가 없는 경우 해결 방법"
    )

    # Search interface
    with st.container():
        search_query = st.text_input(
            "🔍 검색 내용을 입력해주세요.",
            placeholder=default_query,
        ).strip()
        search_button = st.button("유사 문서 찾기")

    if search_button:
        if not search_query:
            search_query = default_query

        st.markdown(f"### 유사 문서")

        with st.spinner("찾는 중 ..."):
            embedding_vector, tokens = rag.embed(search_query)
            usd_price, krw_price = get_embedding_price(rag.embedding_model, tokens)
            doc_list = rag.similarity_search(embedding_vector)

        st.markdown(
            f"{len(doc_list)} 개의 문서를 찾았습니다. (토큰: {tokens}, 임베딩 비용: ${usd_price}, ₩{krw_price})"
        )

        if doc_list:
            for doc in doc_list:
                with st.expander(f"{doc.제목} (유사도 거리 : {doc.distance:.4f})"):
                    meta_data = {
                        "문서ID": doc.문서ID,
                        "공개여부": doc.공개여부,
                        "문서분류": doc.문서분류,
                        "법령분류": doc.법령분류,
                        "생성일시": doc.생성일시,
                        "수정일시": doc.수정일시,
                    }

                    st.markdown("#### 문서 메타데이터")
                    st.table(meta_data)

                    st.markdown("#### 파일내용")
                    st.markdown(doc.파일내용)

                    st.markdown("#### 요지")
                    st.markdown(doc.요지)

                    if doc.url:
                        st.markdown(f"**[원문 보기]({doc.url})**")
        else:
            st.info("No documents found matching your search criteria.")

        st.markdown(f"### LLM 응답 (모델: {rag.chat_model})")

        with st.spinner("LLM 응답 생성 중 ..."):
            # 응답을 표시할 빈 markdown 컴포넌트 생성
            response_container = st.empty()

            # 스트리밍 응답 처리
            for current_text in rag.make_reply(
                system_prompt=system_prompt + "\n\n" + f"<context>{doc_list}</context>",
                user_prompt=f"Question: {search_query}",
            ):
                # 누적된 텍스트로 컴포넌트 업데이트
                response_container.markdown(current_text)

    # Footer
    st.markdown("---")
    st.markdown(
        """
<div style="text-align: center; color: gray; font-size: 0.8em;">
만든이 : 파이썬사랑방 <a href="mailto:me@pyhub.kr">me@pyhub.kr</a> , 자문 : 서찬영세무회계사무소
</div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    load_dotenv()

    try:
        with open("system_prompt.txt", "rt", encoding="utf-8") as f:
            loaded_system_prompt = f.read()
    except IOError:
        loaded_system_prompt = None

    main(
        db_path="./sample-taxlaw-1000.sqlite3",
        table_name="taxlaw_documents",
        embedding_model="text-embedding-3-large",
        chat_model="gpt-4o",
        system_prompt=loaded_system_prompt,
        # .env 파일에서 로드된 환경변수 사용
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
