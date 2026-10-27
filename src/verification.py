
from __future__ import annotations

import re
from typing import Any

from transformers import pipeline


MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
VERDICT_THRESHOLD = 0.75

# If support and contradiction are both strong and close,
# avoid forcing a verdict when the evidence may conflict.
CONFLICT_MARGIN = 0.10


class NliVerifier:
    """Compare individual claims with focused evidence passages."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        debug: bool = False,
    ):
        self.debug = debug

        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            top_k=None,
            truncation=True,
        )

        self.label_map = self._resolve_labels()

        if self.debug:
            print("NLI model:", model_name)
            print("Model labels:", self.classifier.model.config.id2label)
            print("Resolved labels:", self.label_map)

    def _resolve_labels(self) -> dict[str, str]:
        id2label = self.classifier.model.config.id2label
        mapping: dict[str, str] = {}

        for raw_id, raw_label in id2label.items():
            label = str(raw_label).lower().replace("_", " ").strip()

            if "entail" in label:
                canonical = "entailment"
            elif "contrad" in label:
                canonical = "contradiction"
            elif "neutral" in label:
                canonical = "neutral"
            else:
                continue

            mapping[str(raw_id)] = canonical
            mapping[label] = canonical

        if not all(
            name in mapping.values()
            for name in ("entailment", "contradiction", "neutral")
        ):
            raise RuntimeError(f"Cannot resolve NLI labels: {id2label}")

        return mapping

    def _canonical_label(self, raw_label: Any) -> str:
        label = str(raw_label).lower().replace("_", " ").strip()

        if label in self.label_map:
            return self.label_map[label]

        match = re.fullmatch(r"label[\s-]?(\d+)", label)
        if match and match.group(1) in self.label_map:
            return self.label_map[match.group(1)]

        raise RuntimeError(
            f"Unknown NLI label: {raw_label!r}; mapping={self.label_map}"
        )

    def predict(
        self,
        premise: str,
        hypothesis: str,
    ) -> dict[str, float]:
        """Score evidence (premise) against a claim (hypothesis)."""

        if not isinstance(premise, str) or not premise.strip():
            raise ValueError("Evidence premise is empty.")

        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError("Claim hypothesis is empty.")

        output = self.classifier(
            {"text": premise, "text_pair": hypothesis},
            top_k=None,
            truncation=True,
        )

        while (
            isinstance(output, list)
            and len(output) == 1
            and isinstance(output[0], list)
        ):
            output = output[0]

        if isinstance(output, dict):
            output = [output]

        scores = {
            "entailment": 0.0,
            "contradiction": 0.0,
            "neutral": 0.0,
        }

        if not isinstance(output, list):
            raise RuntimeError(f"Unexpected NLI output: {output!r}")

        for item in output:
            if not isinstance(item, dict):
                continue

            canonical = self._canonical_label(item["label"])
            scores[canonical] = float(item["score"])

        if self.debug:
            print("\nNLI premise:", premise[:500])
            print("NLI claim:", hypothesis)
            print("NLI scores:", scores)

        return scores


def split_into_claims(answer: str) -> list[str]:
    """
    Split an answer into sentence-level candidate claims.

    Splits at sentence boundaries and semicolons while preserving
    the original wording. This intentionally avoids splitting every
    occurrence of 'and', which could damage names, lists, and meaning.
    """
    if not isinstance(answer, str):
        return []

    cleaned = re.sub(r"\s+", " ", answer).strip()

    if not cleaned:
        return []

    # Split sentence boundaries and semicolons.
    parts = re.split(
        r"(?<=[.!?])\s+|;\s*",
        cleaned,
    )

    claims = []
    seen = set()

    for part in parts:
        claim = part.strip(" \t\n-•")
        if len(claim) < 12:
            continue

        # Remove duplicates without changing the displayed claim.
        normalized = claim.casefold()
        if normalized not in seen:
            seen.add(normalized)
            claims.append(claim)

    return claims


def _split_evidence(text: str, max_chars: int = 700) -> list[str]:
    """Split evidence into smaller sentence-based passages."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            for start in range(0, len(sentence), max_chars):
                chunks.append(sentence[start:start + max_chars])
            continue

        if not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= max_chars:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def verify_claim(
    claim: str,
    evidence: list[dict[str, Any]],
    verifier: NliVerifier,
) -> dict[str, Any]:
    """Verify one claim against retrieved evidence passages."""

    scored: list[dict[str, Any]] = []

    for item in evidence:
        evidence_text = item.get("text", "")

        if not isinstance(evidence_text, str):
            continue

        for chunk in _split_evidence(evidence_text):
            scores = verifier.predict(
                premise=chunk,
                hypothesis=claim,
            )

            scored.append({
                **item,
                "text": chunk,
                **scores,
                "nli_label": max(scores, key=scores.get),
            })

    best_entailment = max(
        (item["entailment"] for item in scored),
        default=0.0,
    )
    best_contradiction = max(
        (item["contradiction"] for item in scored),
        default=0.0,
    )

    # If strong evidence supports both sides, avoid a forced verdict.
    conflicting_evidence = (
        best_entailment >= VERDICT_THRESHOLD
        and best_contradiction >= VERDICT_THRESHOLD
        and abs(best_entailment - best_contradiction) < CONFLICT_MARGIN
    )

    if conflicting_evidence:
        verdict = "UNVERIFIED"
        confidence = max(best_entailment, best_contradiction)
        explanation = (
            "Retrieved passages provide strong but conflicting signals. "
            "Human review is recommended before deciding the claim."
        )

    elif (
        best_entailment >= VERDICT_THRESHOLD
        and best_entailment > best_contradiction
    ):
        verdict = "SUPPORTED"
        confidence = best_entailment
        explanation = (
            "At least one retrieved passage strongly supports the claim. "
            "Review the cited evidence."
        )

    elif (
        best_contradiction >= VERDICT_THRESHOLD
        and best_contradiction > best_entailment
    ):
        verdict = "CONTRADICTED"
        confidence = best_contradiction
        explanation = (
            "At least one retrieved passage strongly contradicts the claim. "
            "Review the cited evidence."
        )

    else:
        verdict = "UNVERIFIED"
        confidence = max(best_entailment, best_contradiction)
        explanation = (
            "The retrieved evidence did not meet the threshold for "
            "support or contradiction. This does not prove the claim false."
        )

    scored.sort(
        key=lambda item: max(
            item["entailment"],
            item["contradiction"],
        ),
        reverse=True,
    )

    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": float(confidence),
        "explanation": explanation,
        "evidence": scored,
    }
