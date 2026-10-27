from __future__ import annotations

import csv
from io import StringIO


def results_to_csv(rows: list[dict]) -> bytes:
    output = StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "claim_id", "claim", "verdict", "confidence", "explanation",
            "evidence_sources", "evidence_pages", "evidence_texts",
        ],
    )
    writer.writeheader()
    for row in rows:
        evidence = row.get("evidence", [])
        writer.writerow({
            "claim_id": row.get("claim_id"),
            "claim": row.get("claim"),
            "verdict": row.get("verdict"),
            "confidence": row.get("confidence"),
            "explanation": row.get("explanation"),
            "evidence_sources": " | ".join(dict.fromkeys(e.get("source", "") for e in evidence)),
            "evidence_pages": " | ".join(str(e.get("page", "")) for e in evidence),
            "evidence_texts": " | ".join(e.get("text", "") for e in evidence),
        })
    return output.getvalue().encode("utf-8-sig")
