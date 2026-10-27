
import json
import os
import time
import random

from dotenv import load_dotenv
from google import genai

from src.verification import verify_claim

load_dotenv()


def _source_reference(evidence):
    source = evidence.get("source", "Unknown source")
    page = evidence.get("page", "?")
    return f"{source}, p. {page}"


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key missing. Add GEMINI_API_KEY to your .env file."
        )

    return genai.Client(api_key=api_key)


def _generate_candidate_corrections(rows):
    """Generate candidate corrections using Gemini and retrieved evidence."""

    contradicted = [
        row for row in rows
        if row.get("verdict") == "CONTRADICTED"
    ]

    if not contradicted:
        return {}

    claims_context = []
    evidence_context = []
    evidence_lookup = {}

    for row in contradicted:
        claim_id = row.get("claim_id")

        claims_context.append({
            "claim_id": claim_id,
            "original_claim": row.get("claim", ""),
        })

        for index, ev in enumerate(row.get("evidence", [])[:2], start=1):
            evidence_text = ev.get("text", "")

            if not isinstance(evidence_text, str) or not evidence_text.strip():
                continue

            evidence_id = f"C{claim_id}_E{index}"

            evidence_lookup.setdefault(claim_id, {})[evidence_id] = ev

            evidence_context.append({
                "claim_id": claim_id,
                "evidence_id": evidence_id,
                "source": ev.get("source", "Unknown source"),
                "page": ev.get("page", "?"),
                "text": evidence_text,
            })

    if not evidence_context:
        return {}

    prompt = f"""
You are a conservative factual correction assistant.

Original contradicted claims:
{json.dumps(claims_context, ensure_ascii=False)}

Evidence retrieved from uploaded documents:
{json.dumps(evidence_context, ensure_ascii=False)}

Rules:
- Correct only claims contradicted by the retrieved evidence.
- Use evidence supplied for the same claim only.
- Do not use outside knowledge or invent facts.
- If evidence is insufficient, return null for corrected_claim.
- Cite only evidence IDs supplied for that claim.
- Return valid JSON only.

Required structure:
{{
  "corrections": [
    {{
      "claim_id": 1,
      "corrected_claim": "Corrected statement or null",
      "evidence_ids": ["C1_E1"]
    }}
  ]
}}
"""

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    candidates = {}

    # Keep the client open for the entire API request.
    with _get_client() as client:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    data = json.loads(response.text)

    for item in data.get("corrections", []):
        claim_id = item.get("claim_id")
        corrected = item.get("corrected_claim")
        evidence_ids = item.get("evidence_ids", [])

        valid_lookup = evidence_lookup.get(claim_id, {})

        if not isinstance(corrected, str) or not corrected.strip():
            continue

        if not isinstance(evidence_ids, list):
            continue

        # Only allow evidence IDs belonging to this exact claim.
        cited_evidence = [
            valid_lookup[evidence_id]
            for evidence_id in evidence_ids
            if isinstance(evidence_id, str)
            and evidence_id in valid_lookup
        ]

        if cited_evidence:
            candidates[claim_id] = {
                "corrected_claim": corrected.strip(),
                "evidence": cited_evidence,
            }

    return candidates


def generate_corrected_answer(rows, verifier):
    """
    Preserve supported claims, validate proposed corrections with NLI,
    and mark unsupported claims as unverified.
    """

    candidates = _generate_candidate_corrections(rows)

    output_lines = []
    audit = []

    for row in rows:
        claim_id = row.get("claim_id", "?")
        claim = row.get("claim", "").strip()
        verdict = row.get("verdict", "UNVERIFIED")
        evidence = row.get("evidence", [])

        if verdict == "SUPPORTED":
            line = claim

            supporting = sorted(
                evidence,
                key=lambda ev: ev.get("entailment", 0) or 0,
                reverse=True,
            )

            if supporting:
                line += f" [Source: {_source_reference(supporting[0])}]"

            status = "PRESERVED"

        elif verdict == "CONTRADICTED":
            candidate = candidates.get(row.get("claim_id"))
            verified_correction = None

            if candidate:
                # Validate the proposed correction against its cited evidence.
                check = verify_claim(
                    candidate["corrected_claim"],
                    candidate["evidence"],
                    verifier,
                )

                if check.get("verdict") == "SUPPORTED":
                    verified_correction = candidate

            if verified_correction:
                line = verified_correction["corrected_claim"]

                references = list(dict.fromkeys(
                    _source_reference(ev)
                    for ev in verified_correction["evidence"]
                ))

                line += " [Source: " + "; ".join(references) + "]"
                status = "CORRECTED"

            else:
                line = (
                    "[UNVERIFIED — no evidence-grounded correction could "
                    f"be confirmed] Original claim: {claim}"
                )
                status = "NOT SAFELY CORRECTED"

        else:
            line = f"[UNVERIFIED — insufficient evidence] {claim}"
            status = "UNVERIFIED"

        output_lines.append(line)

        audit.append({
            "claim_id": claim_id,
            "original_claim": claim,
            "original_verdict": verdict,
            "correction_status": status,
            "output": line,
        })

    return {
        "corrected_answer": "\n\n".join(output_lines),
        "audit": audit,
    }
