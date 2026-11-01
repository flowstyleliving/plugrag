from __future__ import annotations

import json

from lens_rag.query_engine import (
    build_date_where,
    format_results_json,
    group_query_response,
    metadata_matches_category,
)


def test_group_query_response_groups_by_article() -> None:
    response = {
        "documents": [["Content: school board budget", "Content: more school funding"]],
        "metadatas": [
            [
                {
                    "article_id": "1",
                    "title": "School Board budget",
                    "date": "2024-01-15T10:00:00",
                    "url": "https://example.test/1",
                    "categories": "Education",
                    "tags": "Schools",
                    "chunk_index": 0,
                },
                {
                    "article_id": "1",
                    "title": "School Board budget",
                    "date": "2024-01-15T10:00:00",
                    "url": "https://example.test/1",
                    "categories": "Education",
                    "tags": "Schools",
                    "chunk_index": 1,
                },
            ]
        ],
        "distances": [[0.1, 0.2]],
    }
    results = group_query_response(response, top_k=10)
    assert len(results) == 1
    assert results[0].score == 0.9
    assert len(results[0].matches) == 2


def test_category_filter_matches_names_and_slugs() -> None:
    metadata = {"categories": "Criminal Justice; Education", "category_slugs": "criminal-justice; education"}
    assert metadata_matches_category(metadata, "education")
    assert metadata_matches_category(metadata, "criminal-justice")
    assert not metadata_matches_category(metadata, "environment")


def test_date_filter_builds_chroma_where() -> None:
    assert build_date_where("2024-01-01", None) == {"date_int": {"$gte": 20240101}}
    assert build_date_where(None, "2024-02-01") == {"date_int": {"$lte": 20240201}}
    assert build_date_where("2024-01-01", "2024-02-01") == {
        "$and": [{"date_int": {"$gte": 20240101}}, {"date_int": {"$lte": 20240201}}]
    }


def test_json_output_format() -> None:
    response = {
        "documents": [["Content: jail oversight"]],
        "metadatas": [
            [
                {
                    "article_id": "2",
                    "title": "Jail oversight",
                    "date": "2023-03-01T10:00:00",
                    "url": "https://example.test/2",
                    "categories": "Criminal Justice",
                    "tags": "Jail",
                    "chunk_index": 0,
                }
            ]
        ],
        "distances": [[0.05]],
    }
    payload = json.loads(format_results_json(group_query_response(response, top_k=1)))
    assert payload[0]["article_id"] == "2"
    assert payload[0]["matches"][0]["excerpt"] == "jail oversight"


def test_group_query_response_falls_back_to_document_title() -> None:
    response = {
        "documents": [["Title: Jail oversight hearing\n\nContent: public testimony"]],
        "metadatas": [
            [
                {
                    "article_id": "3",
                    "title": "",
                    "date": "2019-03-01T10:00:00",
                    "url": "https://example.test/3",
                    "categories": "Criminal Justice",
                    "tags": "",
                    "chunk_index": 0,
                }
            ]
        ],
        "distances": [[0.1]],
    }

    results = group_query_response(response, top_k=1)

    assert results[0].title == "Jail oversight hearing"
