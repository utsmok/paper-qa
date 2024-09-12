from paperqa.litqa import LitQAEvaluation, read_litqa_v2_from_hub


def test_creating_litqa_questions() -> None:
    """Test making LitQA eval questions after downloading from Hugging Face Hub."""
    _, eval_split = read_litqa_v2_from_hub()
    assert len(eval_split) > 3
    assert [
        LitQAEvaluation.from_question(
            ideal=row.ideal, distractors=row.distractors, question=row.question  # type: ignore[arg-type]
        )[0]
        for row in eval_split[:3].itertuples()
    ] == [
        (
            "Q: Which of the following mutations in yeast Pbs2 increases its"
            " interaction with SH3?\n\nOptions:\nA) K85W\nB) S83F\nC) Insufficient"
            " information to answer this question\nD) N92H\nE) P97A\nF) I87W\nG) N92S"
        ),
        (
            "Q: What percentage of colorectal cancer-associated fibroblasts typically"
            " survive at 2 weeks if cultured with the platinum-based chemotherapy"
            " oxaliplatin?\n\nOptions:\nA) 0%\nB) Insufficient information to answer"
            " this question\nC) 50-80%\nD) 20-50%\nE) 1-20%\nF) 80-99%"
        ),
        (
            "Q: Which of the following genes shows the greatest difference in gene"
            " expression between homologous cell types in mouse and human"
            " brain?\n\nOptions:\nA) Htr3a\nB) Htr1d\nC) Htr6\nD) Htr5a\nE)"
            " Insufficient information to answer this question"
        ),
    ]
