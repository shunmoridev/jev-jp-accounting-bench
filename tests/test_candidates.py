from bench import build_jfinqa_candidates, numeric_distractors


def test_numeric_distractors_preserve_percent_shape():
    values = numeric_distractors("25.0%")
    assert len(values) == 3
    assert "25.0%" not in values
    assert all(value.endswith("%") for value in values)


def test_numeric_distractors_preserve_currency_shape():
    values = numeric_distractors("1,000円")
    assert "1,000円" not in values
    assert all(value.endswith("円") for value in values)


def test_candidate_generation_is_deterministic_and_contains_gold():
    kwargs = {
        "gold": "25.0%",
        "answer_pool": ["25.0%", "10.0%", "15.0%"],
        "seed": 42,
        "item_id": "nr_test",
    }
    criteria1, gold1 = build_jfinqa_candidates(**kwargs)
    criteria2, gold2 = build_jfinqa_candidates(**kwargs)

    assert criteria1 == criteria2
    assert gold1 == gold2
    assert criteria1[gold1] == "25.0%"
    assert 2 <= len(criteria1) <= 4


def test_text_candidate_generation_uses_same_subtask_pool():
    criteria, gold_label = build_jfinqa_candidates(
        gold="増収",
        answer_pool=["増収", "減収", "横ばい"],
        seed=7,
        item_id="tr_test",
    )
    assert criteria[gold_label] == "増収"
    assert set(criteria.values()) == {"増収", "減収", "横ばい"}
