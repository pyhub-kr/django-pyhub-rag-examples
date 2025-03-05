import os
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv

from pyhub.llm import LLM
from pyhub.rag.db.sqlite_vec import similarity_search


def main(
    db_path: Union[Path, str],
    table_name: str,
    system_prompt: str,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
):
    query = "대기업과 중소기업은 공정거래법을 어떻게 준수해야하나요?"  # 질의 내용

    # 지식 검색
    doc_list = similarity_search(
        db_path=db_path,
        table_name=table_name,
        query=query,
        embedding_model="text-embedding-3-small",
        api_key=openai_api_key,
    )

    # print(doc_list)
    지식 = str(doc_list)

    # 지식 + 질의를 LLM에게 전달하여 응답 생성
    chat_llm = LLM.create(
        model="claude-3-7-sonnet-latest",
        api_key=anthropic_api_key,
        # model="gpt-4o-mini",
        # api_key=openai_api_key,
        system_prompt=system_prompt + f"\n\n<context>{지식}</context>",
        max_tokens=4000,
    )
    # print(chat_llm.reply(query))  # 응답 한 번에 생성

    for reply in chat_llm.reply(query, stream=True):
        print(reply, end="", flush=True)
    print()


if __name__ == "__main__":
    load_dotenv()

    try:
        with open("system_prompt.txt", "rt", encoding="utf-8") as f:
            loaded_system_prompt = f.read()
    except IOError:
        loaded_system_prompt = None

    main(
        db_path="./fair-sample.sqlite3",
        table_name="documents",
        system_prompt=loaded_system_prompt,
        # .env 파일에서 로드된 환경변수 사용
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
