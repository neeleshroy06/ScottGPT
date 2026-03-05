import pandas as pd


def has_any(text: str, keywords: list[str]) -> bool:
    """Return True if any keyword appears in the text."""
    return any(k in text for k in keywords)


def load_student_messages(csv_path: str) -> pd.DataFrame:
    """Load usage CSV and keep only student-authored messages (not AI_TUTOR)."""
    df = pd.read_csv(csv_path)
    df = df[df["user_id"] != "AI_TUTOR"].copy()
    df["message_body"] = df["message_body"].fillna("").astype(str)
    df["message_lower"] = df["message_body"].str.lower()
    return df


def add_question_type_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple category flags for student questions.
    Categories are not mutually exclusive; a message can belong to multiple types.
    """

    def is_conceptual_course_content(t: str) -> int:
        keywords = [
            "what is",
            "what are",
            "difference between",
            "define",
            "definition of",
            "explain",
            "explanation of",
            "why is",
            "how does",
            # course-specific terminology
            "relational algebra",
            "sql",
            "primary key",
            "foreign key",
            "join",
            "er diagram",
            "erd",
            "normalization",
            "hdfs",
            "mapreduce",
        ]
        return int(has_any(t, keywords))

    def is_worked_example_or_how_to(t: str) -> int:
        keywords = [
            "example",
            "give an example",
            "can you give me an example",
            "show me an example",
            "walk me through",
            "step by step",
            "how do i",
            "how to ",
            "practice quiz",
            "practice question",
        ]
        return int(has_any(t, keywords))

    def is_exam_or_quiz_related(t: str) -> int:
        keywords = [
            "exam",
            "midterm",
            "quiz",
            "test",
            "what should i study",
            "what will be on",
            "what will be covered",
            "how to prepare",
        ]
        return int(has_any(t, keywords))

    def is_homework_or_project_help(t: str) -> int:
        keywords = [
            "homework",
            "assignment",
            "lab",
            "project",
            "final project",
            "grade",
            "instructions :",
            "thee were the insructions",
            "here are the instructions",
        ]
        return int(has_any(t, keywords))

    def is_meta_about_tutor_or_course(t: str) -> int:
        keywords = [
            "what can you do",
            "who created you",
            "how should i use you",
            "what are you able to do",
            "can you tell me a little about who created you",
            "what is this class about",
        ]
        return int(has_any(t, keywords))

    df["qtype_conceptual_course_content"] = df["message_lower"].apply(
        is_conceptual_course_content
    )
    df["qtype_worked_example_or_how_to"] = df["message_lower"].apply(
        is_worked_example_or_how_to
    )
    df["qtype_exam_or_quiz_related"] = df["message_lower"].apply(is_exam_or_quiz_related)
    df["qtype_homework_or_project_help"] = df["message_lower"].apply(
        is_homework_or_project_help
    )
    df["qtype_meta_about_tutor_or_course"] = df["message_lower"].apply(
        is_meta_about_tutor_or_course
    )

    return df


def summarize_question_types(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table with counts and percentages for each question type."""
    qtype_cols = [c for c in df.columns if c.startswith("qtype_")]
    n = len(df)

    rows = []
    for col in qtype_cols:
        count = int(df[col].sum())
        pct = (count / n * 100) if n > 0 else 0.0
        rows.append(
            {
                "question_type": col,
                "count_messages": count,
                "percent_of_messages": round(pct, 1),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("count_messages", ascending=False)
    return summary_df


def example_questions(df: pd.DataFrame, qtype_col: str, k: int = 5) -> list[str]:
    """Return up to k example student questions for a given question type."""
    mask = df[qtype_col] == 1
    return df.loc[mask, "message_body"].head(k).tolist()


def main() -> None:
    csv_path = "usage_data_csv_utf8.csv"
    df = load_student_messages(csv_path)
    print(f"Total student messages: {len(df)}")

    df = add_question_type_columns(df)
    summary_df = summarize_question_types(df)

    print("\nQuestion type summary:")
    print(summary_df.to_string(index=False))

    print("\nExample questions for each type (up to 5):")
    for qtype_col in summary_df["question_type"]:
        print(f"\n{qtype_col}:")
        for q in example_questions(df, qtype_col, k=5):
            print(f" - {q}")


if __name__ == "__main__":
    main()

