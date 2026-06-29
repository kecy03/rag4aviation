#!/usr/bin/env python3
"""Interactive demo for the Aviation RAG system."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conversation.engine import MultiTurnRAGEngine


def main():
    print("=" * 60)
    print("  Aviation RAG System - Interactive Demo")
    print("  Type 'quit' to exit, 'reset' to clear conversation")
    print("=" * 60)

    engine = MultiTurnRAGEngine()

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() == "quit":
            print("Goodbye!")
            break
        if query.lower() == "reset":
            engine.reset()
            print("Conversation reset.")
            continue

        print("Thinking...")
        result = engine.chat(query)

        print(f"\nAnswer: {result['answer']}")
        print(f"\n(Collection: {result['collection_used']}, Rewritten: {result['rewritten_query']})")
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['score']}] {s['chunk_id']} (page {s['page']})")


if __name__ == "__main__":
    main()
