import pandas as pd


def load_and_filter(csv_path: str) -> pd.DataFrame:
    """
    Load the survey CSV and keep only:
    - Finished == True
    - Status != 'Survey Preview'
    - Students who provided consent (Q1 text)
    - Non-empty free-response answers (Q7)
    """
    df = pd.read_csv(csv_path)

    # `Finished` comes in as strings like "True"/"False" from Qualtrics.
    # Normalize to a simple boolean-like string column for robust filtering.
    finished_normalized = df["Finished"].astype(str).str.strip().str.lower()

    consent_text = (
        "I have read and understood the consent form, and I provide my consent to "
        "participate in this study."
    )

    df = df[
        (finished_normalized == "true")
        & (df["Status"] != "Survey Preview")
        & (df["Q1"] == consent_text)
    ].copy()

    df["Q7"] = df["Q7"].fillna("").astype(str)
    df = df[df["Q7"].str.strip() != ""].copy()

    df["Q7_lower"] = df["Q7"].str.lower()
    return df


def has_any(text: str, keywords: list[str]) -> bool:
    """Return True if any keyword appears in the text."""
    return any(k in text for k in keywords)


def add_theme_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple theme columns to the DataFrame.
    Each theme is 1 if mentioned in Q7, else 0.
    """
    def theme_course_specific(t: str) -> int:
        return int(
            has_any(
                t,
                [
                    "class-specific",
                    "course-specific",
                    "for this class",
                    "for this course",
                    "lecture slides",
                    "lecture slide",
                    "slides",
                    "class materials",
                    "course materials",
                    "notes",
                    "based on class",
                    "based on course",
                ],
            )
        )

    def theme_exam_helpful(t: str) -> int:
        return int(
            has_any(
                t,
                ["midterm", "final", "exam", "quiz", "quizzes", "study for", "studying"],
            )
        )

    def theme_general_positive(t: str) -> int:
        return int(
            has_any(
                t,
                [
                    "good idea",
                    "great idea",
                    "amazing",
                    "nice interface",
                    "easy to use",
                    "i liked",
                    "i love",
                    "very useful",
                    "helpful",
                    "it was helpful",
                ],
            )
        )

    def theme_too_restricted(t: str) -> int:
        return int(
            has_any(
                t,
                [
                    "too restricted",
                    "pretty restrictive",
                    "gimped",
                    "limited in",
                    "limited set of training data",
                    "only notes",
                    "only from notes",
                    "only from slides",
                    "restricted",
                ],
            )
        )

    def theme_points_to_slides_only(t: str) -> int:
        return int(
            has_any(
                t,
                [
                    "just gives you directions",
                    "just gives directions",
                    "place to find sources",
                    "just a place to find sources",
                    "just links",
                    "only gave links",
                    "doesn't really process",
                    "does not really process",
                ],
            )
        )

    def theme_not_capable_enough(t: str) -> int:
        return int(
            has_any(
                t,
                [
                    "did not do what i asked",
                    "didn't do what i asked",
                    "didnt do what i asked",
                    "horrid mock",
                    "bad mock",
                    "super basic questions",
                    "too basic questions",
                    "wasn't really useful",
                    "wasn’t really useful",
                    "hardly useful",
                    "not useful",
                    "not that useful",
                    "did not have enough available information",
                    "could not answer",
                    "couldn't answer",
                    "couldnt answer",
                ],
            )
        )

    def theme_other_chatbots_better(t: str) -> int:
        return int(
            has_any(
                t,
                [
                    "other chatbots",
                    "other bots",
                    "other ai",
                    "chatgpt",
                    "publicly available ai",
                    "other tools are better",
                ],
            )
        )

    def theme_call_limits(t: str) -> int:
        return int(
            has_any(
                t,
                ["limited number of calls", "number of calls", "call limit", "limited calls"],
            )
        )

    def theme_did_not_use(t: str) -> int:
        return int(
            has_any(
                t,
                [
                    "didn't use it",
                    "did not use it",
                    "never used it",
                    "never had to use it",
                    "i have not used it",
                    "i never used the chatbot",
                    "i didnt use it",
                    "i didn't really use it",
                    "forgot about it",
                    "i forgot the ai tutor",
                    "i forgot about it",
                    "unable to have any opinions",
                    "so n/a",
                ],
            )
        )

    def theme_ui_or_access_issues(t: str) -> int:
        return int(
            has_any(
                t,
                [
                    "hard to read",
                    "difficult to read",
                    "format of some responses",
                    "wasn't working",
                    "wasnt working",
                    "didn't really know where i could find it",
                    "did not really know where i could find it",
                    "announcement on canvas",
                    "link to the chatbot",
                    "find the chatbot",
                ],
            )
        )

    df["theme_course_specific"] = df["Q7_lower"].apply(theme_course_specific)
    df["theme_exam_helpful"] = df["Q7_lower"].apply(theme_exam_helpful)
    df["theme_general_positive"] = df["Q7_lower"].apply(theme_general_positive)
    df["theme_too_restricted"] = df["Q7_lower"].apply(theme_too_restricted)
    df["theme_points_to_slides_only"] = df["Q7_lower"].apply(theme_points_to_slides_only)
    df["theme_not_capable_enough"] = df["Q7_lower"].apply(theme_not_capable_enough)
    df["theme_other_chatbots_better"] = df["Q7_lower"].apply(theme_other_chatbots_better)
    df["theme_call_limits"] = df["Q7_lower"].apply(theme_call_limits)
    df["theme_did_not_use"] = df["Q7_lower"].apply(theme_did_not_use)
    df["theme_ui_or_access_issues"] = df["Q7_lower"].apply(theme_ui_or_access_issues)

    return df


def summarize_themes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table with counts and percentages for each theme."""
    theme_cols = [c for c in df.columns if c.startswith("theme_")]
    n = len(df)

    rows = []
    for col in theme_cols:
        count = int(df[col].sum())
        pct = (count / n * 100) if n > 0 else 0.0
        rows.append(
            {
                "theme": col,
                "count_students": count,
                "percent_of_responses": round(pct, 1),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("count_students", ascending=False)
    return summary_df


def example_quotes(df: pd.DataFrame, theme_col: str, k: int = 3) -> list[str]:
    """Return up to k example free-response quotes for a given theme."""
    mask = df[theme_col] == 1
    return df.loc[mask, "Q7"].head(k).tolist()


def main() -> None:
    csv_path = "ucr-chatbot-usage_January 13, 2026_12.42 (1).csv"
    df = load_and_filter(csv_path)
    print(f"Included responses (consented + non-empty Q7): {len(df)}")

    df = add_theme_columns(df)
    summary_df = summarize_themes(df)

    print("\nTheme summary:")
    print(summary_df.to_string(index=False))

    print("\nExample quotes for each theme (up to 3):")
    for theme_col in summary_df["theme"]:
        quotes = example_quotes(df, theme_col, k=3)
        print(f"\n{theme_col}:")
        for q in quotes:
            print(f" - {q}")


if __name__ == "__main__":
    main()

