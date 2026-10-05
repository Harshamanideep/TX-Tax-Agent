"""Chat with the agent. It remembers the conversation until you quit.

Run:  python -m src.chat
Commands:  /new  start a fresh conversation    /quit  exit
"""
import uuid

from src.agent import run


def main():
    thread_id = str(uuid.uuid4())
    print("Tax form agent ready. Type /new for a fresh conversation, /quit to exit.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/new":
            thread_id = str(uuid.uuid4())
            print("(Started a new conversation.)")
            continue

        try:
            result = run(q, thread_id)
        except Exception as e:
            print(f"\nError: {e}")
            continue

        if result["tools_used"]:
            print("\n[tools] " + " -> ".join(result["tools_used"]))
        print(f"\nAgent: {result['answer']}")


if __name__ == "__main__":
    main()