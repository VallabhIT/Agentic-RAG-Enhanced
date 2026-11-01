from app import config
from app.graph import build_graph
from app.state import new_state
from app.tools import init_vectorstore


def _ask_approval(message: str) -> bool:
    print("\n" + message)
    while True:
        answer = input("Approve? [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'")


def run_turn(graph, user_input: str, history: list) -> dict:
    """Run one user turn, handling the human-approval pause/resume cycle."""
    run_cfg = {"recursion_limit": config.RECURSION_LIMIT}
    result = graph.invoke(new_state(user_input, history), run_cfg)

    # Graph paused on a risky SQL/Python step. Resume WITH explicit approval state so it jumps
    # straight to the execute node instead of re-running rewrite / routing / the danger check.
    if result.get("approval_status") == "pending":
        granted = _ask_approval(result["approval_message"])
        resume = {
            **result,
            "messages": [],  # already in state; don't re-append
            "approval_status": "granted" if granted else "denied",
            "approval_required": False,
        }
        result = graph.invoke(resume, run_cfg)
    return result


def main():
    try:
        init_vectorstore(force_rebuild=False)
    except Exception as exc:
        print(f"Warning: could not initialise the knowledge base ({exc}). RAG will fall back to web search.")

    graph = build_graph()
    history: list = []

    while True:
        user_input = input("\nAsk a question (or 'exit'): ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        try:
            result = run_turn(graph, user_input, history)
        except Exception as exc:
            print(f"\nSomething went wrong: {exc}")
            continue

        print("\nAnswer:")
        print(result.get("final_answer", "No answer generated."))
        history = list(result.get("messages", history))


if __name__ == "__main__":
    main()
