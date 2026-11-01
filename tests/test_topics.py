from __future__ import annotations

import json
from pathlib import Path

from lens_rag.topics import (
    MAX_TERMS_PER_TOPIC,
    TopicLexicon,
    build_topic_records,
    main,
    normalize_term,
)


def article(
    article_id: int,
    title: str,
    body: str,
    categories: list[str],
    tags: list[str],
) -> dict[str, object]:
    return {
        "id": article_id,
        "title": title,
        "excerpt": body[:100],
        "plain_text": body,
        "category_names": categories,
        "tag_names": tags,
    }


def test_topic_generation_uses_metadata_and_title_signals() -> None:
    records = build_topic_records(
        [
            article(
                1,
                "Jail consent decree monitor report",
                "The monitor discussed compliance director oversight.",
                ["Criminal Justice"],
                ["Orleans Parish jail consent decree"],
            ),
            article(
                2,
                "Jail oversight hearing",
                "The sheriff discussed consent decree progress.",
                ["Criminal Justice"],
                ["Orleans Parish jail consent decree"],
            ),
            article(
                3,
                "School board budget",
                "Education funding hearing.",
                ["Schools"],
                ["Budget"],
            ),
        ],
        max_topics=20,
    )

    consent_topic = next(
        record for record in records if record.label == "Orleans Parish jail consent decree"
    )

    assert consent_topic.count == 2
    assert "parish jail consent decree" in consent_topic.terms
    assert "criminal justice" in consent_topic.terms
    assert len(consent_topic.terms) <= MAX_TERMS_PER_TOPIC


def test_topic_generation_filters_generic_terms() -> None:
    assert normalize_term("articles coverage reporting") == ""


def test_embedding_neighborhoods_contribute_nearby_article_terms() -> None:
    records = build_topic_records(
        [
            article(
                1,
                "Jail consent decree",
                "Monitor oversight.",
                ["Criminal Justice"],
                ["Jail"],
            ),
            article(
                2,
                "Jail monitor report",
                "Sheriff oversight.",
                ["Criminal Justice"],
                ["Jail"],
            ),
            article(
                3,
                "Compliance director plan",
                "Court oversight details.",
                ["Criminal Justice"],
                [],
            ),
            article(
                4,
                "School board budget",
                "Education funding.",
                ["Schools"],
                ["Budget"],
            ),
        ],
        article_vectors={
            "1": [1.0, 0.0],
            "2": [0.98, 0.02],
            "3": [0.99, 0.01],
            "4": [0.0, 1.0],
        },
        max_topics=20,
    )

    jail_topic = next(record for record in records if record.label == "Jail")
    joined_terms = " ".join(jail_topic.terms)

    assert "compliance director" in joined_terms
    assert "school board" not in joined_terms


def test_topic_generation_works_without_embeddings() -> None:
    records = build_topic_records(
        [
            article(1, "Airbnb rules", "Short term rental rules.", ["Land Use"], ["Airbnb"]),
            article(2, "Airbnb permits", "Vacation rental permits.", ["Land Use"], ["Airbnb"]),
        ],
        max_topics=10,
    )

    assert any(record.label == "Airbnb" for record in records)


def test_topic_lexicon_expands_multiword_and_single_word_queries() -> None:
    records = build_topic_records(
        [
            article(1, "Jail oversight", "Consent decree.", ["Criminal Justice"], ["Jail"]),
            article(2, "Jail monitor", "Compliance director.", ["Criminal Justice"], ["Jail"]),
        ],
        max_topics=10,
    )
    lexicon = TopicLexicon(
        [next(record for record in records if record.label == "Jail")]
    )

    multi = lexicon.expand("jail oversight", ["jail", "oversight"])
    single = lexicon.expand("jail", ["jail"])
    loose = lexicon.expand("oversight", ["oversight"])

    assert multi.terms
    assert single.terms
    assert loose.terms == []


def test_topic_builder_cli_is_deterministic(tmp_path: Path) -> None:
    articles_dir = tmp_path / "articles"
    articles_dir.mkdir()
    articles = [
        article(1, "Jail oversight", "Consent decree.", ["Criminal Justice"], ["Jail"]),
        article(2, "Jail monitor", "Compliance director.", ["Criminal Justice"], ["Jail"]),
    ]
    for item in articles:
        (articles_dir / f"{item['id']}.json").write_text(
            json.dumps(item),
            encoding="utf-8",
        )
    output = tmp_path / "topic_lexicon.json"

    assert main(["--data-dir", str(tmp_path), "--output", str(output)]) == 0
    first = output.read_text(encoding="utf-8")
    assert main(["--data-dir", str(tmp_path), "--output", str(output)]) == 0
    second = output.read_text(encoding="utf-8")

    payload = json.loads(second)
    assert first == second
    assert payload["topics"]
