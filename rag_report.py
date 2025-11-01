import argparse
import os
import sys
from typing import List

from rag_core import query_rag


def format_source_list(contexts: List[dict]) -> str:
    lines: List[str] = []
    for idx, ctx in enumerate(contexts, 1):
        lines.append(
            f"{idx:02d}. {ctx.get('law_name', ctx['file'])} (p.{ctx['page_start']}-{ctx['page_end']}, 점수 {ctx['score']:.3f})"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="물류 관련법 PDF RAG 검색/보고서 생성기")
    parser.add_argument("--root", default=os.getcwd(), help="PDF가 있는 루트 폴더 (기본: 현재 폴더)")
    parser.add_argument("--query", required=True, help="검색 질의")
    parser.add_argument("--top_k", type=int, default=12, help="선정할 상위 컨텍스트 개수")
    parser.add_argument("--embed_model", default="text-embedding-3-small", help="임베딩 모델명")
    parser.add_argument("--chat_model", default="gpt-4o-mini", help="보고서 생성 모델명")
    parser.add_argument("--rebuild_index", action="store_true", help="인덱스 강제 재생성")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.", file=sys.stderr)
        sys.exit(1)

    answer, contexts = query_rag(
        index_root=args.root,
        query=args.query,
        embed_model=args.embed_model,
        chat_model=args.chat_model,
        top_k=args.top_k,
        force_rebuild=args.rebuild_index,
        history_turns=[],
    )

    print("\n===== 보고서 =====\n")
    print(answer)

    if contexts:
        print("\n===== 참고 문서 =====")
        print(format_source_list(contexts))


if __name__ == "__main__":
    main()
