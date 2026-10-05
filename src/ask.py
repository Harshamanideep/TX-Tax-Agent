"""Interactive question loop.

Run:  python -m src.ask           (search all states)
      python -m src.ask --state NE
"""
import argparse

from src.rag import answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="Limit search to one state, e.g. NE")
    args = parser.parse_args()

    print("Ask about your loaded tax forms (Ctrl+C to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        try:
            result = answer(q, state=args.state)
        except Exception as e:
            print(f"\nError: {e}")
            continue
        print(f"\n{result['answer']}\n\nRetrieved from: {'; '.join(result['sources'])}")


if __name__ == "__main__":
    main()
