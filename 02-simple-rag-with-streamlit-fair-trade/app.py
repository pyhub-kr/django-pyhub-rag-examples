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
from anthropic import Anthropic, NOT_GIVEN as ANTHROPIC_NOT_GIVEN
from anthropic.types import ModelParam as AnthropicChatModel
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

LLMChatModel: TypeAlias = Union[str, OpenAIChatModel, AnthropicChatModel]


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


def get_chat_price(
    model: LLMChatModel, input_tokens: int, output_tokens: int
) -> tuple[Decimal, Decimal]:
    input_price_per_1m, output_price_per_1m = {
        # https://platform.openai.com/docs/pricing#latest-models
        "gpt-4o": (Decimal("2.5"), Decimal("10.0")),
        "gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
        "o1": (Decimal("15"), Decimal("60.00")),
        "o3-mini": (Decimal("1.10"), Decimal("4.40")),
        "o1-mini": (Decimal("1.10"), Decimal("4.40")),
        # https://www.anthropic.com/pricing#anthropic-api
        "claude-3-7-sonnet-latest": (Decimal("3"), Decimal("15")),
        "claude-3-5-haiku-latest": (Decimal("0.80"), Decimal("4")),
        "claude-3-opus-latest": (Decimal("15"), Decimal("75")),
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

st.title("📚 독점규제 및 공정거래에 관한 법률 (약칭: 공정거래법) 검색")


@dataclass
class Document:
    id: int
    page_content: str
    metadata: dict
    distance: float

    def __post_init__(self):
        """초기화 후 실행되는 메서드로 필요한 속성들을 설정합니다."""
        # page_content에서 필요한 속성들을 파싱하여 할당
        self.제목 = " ".join(self.page_content.splitlines()[:3]).strip()


class Rag:
    def __init__(
        self,
        db_path: str,
        table_name: str,
        embedding_model: OpenAIEmbeddingModel,
        chat_model: LLMChatModel,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        max_tokens: int = 2000,
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.max_tokens = max_tokens

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
        if "claude" in self.chat_model.lower():
            return self.make_reply_using_anthropic(system_prompt, user_prompt)
        return self.make_reply_using_openai(system_prompt, user_prompt)

    def make_reply_using_openai(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> Generator[str, None, None]:
        text_output = ""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        client = openai.Client(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=self.max_tokens,
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

    def make_reply_using_anthropic(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> Generator[str, None, None]:
        text_output = ""

        messages = [
            {"role": "user", "content": user_prompt},
        ]

        client = Anthropic(api_key=self.anthropic_api_key)
        response = client.messages.create(
            model=self.chat_model,
            system=system_prompt or ANTHROPIC_NOT_GIVEN,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=True,
        )

        input_tokens = 0
        output_tokens = 0

        for chunk in response:
            if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                text_output += chunk.delta.text
            elif hasattr(chunk, "type") and chunk.type == "content_block_delta":
                if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                    text_output += chunk.delta.text
                elif hasattr(chunk, "content_block") and hasattr(
                    chunk.content_block, "text"
                ):
                    text_output += chunk.content_block.text
            yield text_output

            if hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                input_tokens += getattr(chunk.message.usage, "input_tokens", 0)
                output_tokens += getattr(chunk.message.usage, "output_tokens", 0)

            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens += getattr(chunk.usage, "input_tokens", 0)
                output_tokens += getattr(chunk.usage, "output_tokens", 0)

        if input_tokens or output_tokens:
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
    anthropic_api_key: Optional[str] = None,
    max_tokens: int = 2000,
):
    rag = Rag(
        db_path=db_path,
        table_name=table_name,
        embedding_model=embedding_model,
        chat_model=chat_model,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        max_tokens=max_tokens,
    )

    default_query = (
        "대기업과 중소기업은 공정거래법을 어떻게 준수해야하나요?"
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
                    st.markdown(doc.page_content)
                    st.markdown(doc.metadata)
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
만든이 : 파이썬사랑방 <a href="mailto:me@pyhub.kr">me@pyhub.kr</a>
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
        # db_path="./sample-taxlaw-1000.sqlite3",
        db_path="./fair-sample.sqlite3",
        # table_name="taxlaw_documents",
        table_name="documents",
        # embedding_model="text-embedding-3-large",
        embedding_model="text-embedding-3-small",
        chat_model="claude-3-7-sonnet-latest",
        # chat_model="gpt-4o",
        system_prompt=loaded_system_prompt,
        # .env 파일에서 로드된 환경변수 사용
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2000,
    )
