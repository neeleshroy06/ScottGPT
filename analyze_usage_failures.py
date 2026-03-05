import pandas as pd


def has_any(text: str, keywords: list[str]) -> bool:
    """Return True if any keyword appears in the text."""
    return any(k in text for k in keywords)


def classify_ai_messages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict to AI_TUTOR messages and add boolean flags:
    - not_capable_enough: AI says it cannot answer / lacks relevant materials.
    - good_refusal_not_give_away: AI refuses to do homework / provide full solutions.
    """
    ai_df = df[df["user_id"] == "AI_TUTOR"].copy()
    ai_df["message_lower"] = ai_df["message_body"].astype(str).str.lower()

    def is_not_capable_enough(t: str) -> int:
        # Cases where the AI explicitly cannot answer due to missing materials or capability.
        keywords = [
            "cannot find any relevant course materials",
            "i cannot find any relevant course materials",
            "i am sorry, but i cannot find any relevant course materials",
            "i cannot find any relevant course material",
            "i cannot find relevant course materials",
            "i cannot answer your question from the course materials",
        ]
        return int(has_any(t, keywords))

    def is_good_refusal_not_give_away(t: str) -> int:
        # Cases where the AI is correctly refusing to give away too much
        # (e.g., not doing homework / assignments / giving full solutions).
        keywords = [
            "i cannot do your homework",
            "i can't do your homework",
            "cannot complete your homework",
            "cannot complete your assignment",
            "i cannot complete assignments for you",
            "i cannot do your assignment",
            "i can't do your assignment",
            "i cannot give you the answer",
            "i can't give you the answer",
            "i cannot give you the answers",
            "i can't give you the answers",
            "i cannot give you full solutions",
            "i can't give you full solutions",
            "i cannot provide full solutions",
            "i cannot provide the full solution",
            "i cannot directly solve this for you",
            "i cannot directly solve your homework",
            "i cannot directly provide code for your assignment",
        ]
        return int(has_any(t, keywords))

    ai_df["not_capable_enough"] = ai_df["message_lower"].apply(is_not_capable_enough)
    ai_df["good_refusal_not_give_away"] = ai_df["message_lower"].apply(
        is_good_refusal_not_give_away
    )

    return ai_df


def summarize_failures(ai_df: pd.DataFrame) -> None:
    """Print usage statistics for how often the AI did not do what was requested."""
    n_ai = len(ai_df)
    if n_ai == 0:
        print("No AI_TUTOR messages found in usage_data_csv_utf8.csv.")
        return

    n_not_capable = int(ai_df["not_capable_enough"].sum())
    n_good_refusal = int(ai_df["good_refusal_not_give_away"].sum())

    # Any case where the AI did not do what was requested for either reason.
    any_mask = (ai_df["not_capable_enough"] == 1) | (
        ai_df["good_refusal_not_give_away"] == 1
    )
    n_any = int(any_mask.sum())

    def pct(count: int) -> float:
        return round(count / n_ai * 100, 2) if n_ai > 0 else 0.0

    print("Usage-based statistics: AI Tutor not doing what was requested")
    print(f"- Total AI_TUTOR messages: {n_ai}")
    print(
        f"- Any 'did not do what was requested' (either reason): "
        f"{n_any} messages ({pct(n_any)}%)"
    )
    print(
        f"  - Not capable enough / missing materials: "
        f"{n_not_capable} messages ({pct(n_not_capable)}%)"
    )
    print(
        f"  - Good refusal (not giving away too much): "
        f"{n_good_refusal} messages ({pct(n_good_refusal)}%)"
    )


def main() -> None:
    csv_path = "usage_data_csv_utf8.csv"
    df = pd.read_csv(csv_path)
    ai_df = classify_ai_messages(df)
    summarize_failures(ai_df)


if __name__ == "__main__":
    main()

