import html

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.correction import generate_corrected_answer
from src.ingestion import extract_pdf_pages, chunk_pages
from src.retrieval import HybridRetriever
from src.verification import NliVerifier, split_into_claims, verify_claim
from src.reporting import results_to_csv


LABELS = ["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]

# (text colour, background) per verdict: green / red / amber
VERDICT_STYLES = {
    "SUPPORTED": ("#166534", "#dcfce7"),
    "CONTRADICTED": ("#991b1b", "#fee2e2"),
    "UNVERIFIED": ("#92400e", "#fef3c7"),
}

# Session-state keys used by the original app (unchanged) plus one new key.
STATE_KEYS_RESET_ON_RUN = (
    "verification_rows",
    "corrected_answer",
    "evaluation_editor",
    "evaluation_initial",
    "doc_info",  # new: document summary from the last run
)

st.set_page_config(
    page_title="VeriFact AI",
    layout="wide",
)

# --------------------------------------------------
# STYLES (minimal, scoped to custom elements)
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1200px; }
    .vf-badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600;
    }
    .vf-claim, .vf-evidence, .vf-answer {
        padding: 0.55rem 0.9rem; margin: 0.25rem 0 0.75rem;
        border-radius: 0 6px 6px 0; white-space: pre-wrap;
        line-height: 1.5;
    }
    .vf-claim    { border-left: 3px solid #94a3b8; background: rgba(148,163,184,.14); }
    .vf-evidence { border-left: 3px solid #3b82f6; background: rgba(59,130,246,.09); }
    .vf-answer   { border-left: 3px solid #16a34a; background: rgba(22,163,74,.09); }
    .vf-label    { font-size: 0.85rem; opacity: 0.7; margin-bottom: 0.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def badge(verdict):
    fg, bg = VERDICT_STYLES.get(verdict, ("#334155", "#e2e8f0"))
    return (
        f'<span class="vf-badge" style="color:{fg};background:{bg};">'
        f"{html.escape(str(verdict))}</span>"
    )


def text_block(css_class, text):
    return f'<div class="{css_class}">{html.escape(str(text))}</div>'


# --------------------------------------------------
# LOAD MODELS (cached, unchanged)
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    retriever = HybridRetriever()
    verifier = NliVerifier()
    return retriever, verifier


# --------------------------------------------------
# VERIFICATION PIPELINE (same calls and order as before)
# --------------------------------------------------

def run_verification(uploaded_files, answer, top_k):
    """Returns True on success. Clears stale results first."""
    for key in STATE_KEYS_RESET_ON_RUN:
        st.session_state.pop(key, None)

    try:
        with st.spinner("Extracting text from PDFs..."):
            all_pages = []
            per_doc = []

            for file in uploaded_files:
                pages = extract_pdf_pages(file.getvalue(), file.name)
                all_pages.extend(pages)
                per_doc.append(
                    {"Document": file.name, "Pages extracted": len(pages)}
                )

            chunks = chunk_pages(all_pages)

            if not chunks:
                st.error(
                    "No extractable text found. Scanned-only PDFs are not "
                    "supported by this version. Upload text-based PDFs "
                    "(where you can select and copy text)."
                )
                return False

        with st.spinner("Loading models and indexing evidence..."):
            retriever, verifier = load_models()
            retriever.fit(chunks)

        claims = split_into_claims(answer)

        if not claims:
            st.warning(
                "No claims could be extracted. Paste a response with "
                "complete factual sentences."
            )
            return False

        rows = []

        with st.spinner(f"Verifying {len(claims)} claims..."):
            for claim_id, claim in enumerate(claims, start=1):
                evidence = retriever.search(claim, top_k=top_k)
                result = verify_claim(claim, evidence, verifier)
                result["claim_id"] = claim_id
                rows.append(result)

        # Preserve verification results across Streamlit reruns.
        st.session_state["verification_rows"] = rows
        st.session_state["doc_info"] = {
            "documents": per_doc,
            "chunks": len(chunks),
        }

        # Initialize ground-truth labels for a new evaluation.
        st.session_state["evaluation_initial"] = pd.DataFrame({
            "Claim ID": [r["claim_id"] for r in rows],
            "Claim": [r["claim"] for r in rows],
            "Predicted verdict": [r["verdict"] for r in rows],
            "Ground truth": ["UNLABELED"] * len(rows),
        })
        return True

    except Exception as exc:
        st.error(f"Verification failed: {exc}")
        st.exception(exc)
        return False


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("VeriFact AI")
st.caption(
    "Check AI-generated research claims against evidence from your own "
    "papers, then correct and evaluate the results."
)

# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.header("Settings")

    top_k = st.slider(
        "Evidence passages per claim",
        min_value=1,
        max_value=8,
        value=4,
    )

    st.warning(
        "This tool provides evidence-based signals, not a guarantee "
        "of truth. Missing evidence does not prove a claim is false."
    )

# --------------------------------------------------
# MAIN WORKFLOW TABS
# --------------------------------------------------

tab_verify, tab_review, tab_correct, tab_eval = st.tabs([
    "1. Documents and claims",
    "2. Review evidence",
    "3. Correct answer",
    "4. Evaluate",
])

rows = st.session_state.get("verification_rows")

# ==================================================
# TAB 1: DOCUMENTS + VERIFY
# ==================================================

with tab_verify:
    st.subheader("Documents")
    st.write(
        "Upload the papers that should serve as evidence. Use text-based "
        "PDFs; scanned-only PDFs are not supported."
    )

    uploaded_files = st.file_uploader(
        "Upload research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.dataframe(
            pd.DataFrame({
                "Document": [f.name for f in uploaded_files],
                "Size (KB)": [round(f.size / 1024, 1) for f in uploaded_files],
            }),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No documents uploaded yet. Add at least one PDF to begin.")

    doc_info = st.session_state.get("doc_info")
    if doc_info:
        st.caption(
            f'Last run indexed {len(doc_info["documents"])} document(s) '
            f'into {doc_info["chunks"]} evidence chunks.'
        )
        with st.expander("Processed documents"):
            st.dataframe(
                pd.DataFrame(doc_info["documents"]),
                hide_index=True,
                use_container_width=True,
            )

    st.divider()
    st.subheader("AI-generated response to verify")

    answer = st.text_area(
        "Paste the AI-generated research response",
        height=220,
        placeholder="Paste the answer you want to verify.",
    )

    missing = []
    if not uploaded_files:
        missing.append("upload at least one PDF")
    if not answer.strip():
        missing.append("paste a response")
    if missing:
        st.caption("To continue: " + " and ".join(missing) + ".")

    run = st.button(
        "Verify claims",
        type="primary",
        disabled=bool(missing),
    )

    if run:
        if run_verification(uploaded_files, answer, top_k):
            st.success(
                "Verification complete. Open '2. Review evidence' to "
                "see verdicts and sources."
            )
            rows = st.session_state["verification_rows"]

# ==================================================
# TAB 2: REVIEW EVIDENCE
# ==================================================

with tab_review:
    if not rows:
        st.info(
            "No verification results yet. Upload documents, paste a "
            "response, and select 'Verify claims' in step 1."
        )
    else:
        df = pd.DataFrame(rows)
        counts = df["verdict"].value_counts().to_dict()
        total = len(df)
        supported = counts.get("SUPPORTED", 0)

        st.subheader("Verification overview")

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1.container(border=True):
            st.metric("Claims checked", total)
        with k2.container(border=True):
            st.metric("Supported", supported)
        with k3.container(border=True):
            st.metric("Contradicted", counts.get("CONTRADICTED", 0))
        with k4.container(border=True):
            st.metric("Unverified", counts.get("UNVERIFIED", 0))
        with k5.container(border=True):
            st.metric(
                "Support rate",
                f"{supported / total:.0%}" if total else "n/a",
            )

        st.caption(
            "Support rate = supported claims / claims checked. The "
            "confidence signal is derived from NLI model scores. It is not "
            "a calibrated probability that a claim is true."
        )

        st.divider()
        st.subheader("Claims and evidence")

        selected = st.multiselect(
            "Show verdicts",
            options=LABELS,
            default=LABELS,
            key="verdict_filter",
        )

        visible = [r for r in rows if r["verdict"] in selected]
        st.caption(f"Showing {len(visible)} of {total} claims.")

        if not visible:
            st.info("No claims match the selected verdicts.")

        for row in visible:
            verdict = row["verdict"]

            with st.expander(
                f'{row["claim_id"]}. [{verdict}] {row["claim"]}',
                expanded=False,
            ):
                st.markdown(badge(verdict), unsafe_allow_html=True)

                st.markdown(
                    '<div class="vf-label">Claim</div>'
                    + text_block("vf-claim", row["claim"]),
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns(2)
                col1.metric("Predicted verdict", verdict)
                col2.metric(
                    "NLI confidence signal",
                    f'{row["confidence"]:.3f}',
                )

                st.markdown("**Verifier explanation**")
                st.info(row["explanation"])

                st.divider()
                st.markdown("**Evidence and source traceability**")

                evidence_items = row.get("evidence", [])

                if evidence_items:
                    st.caption(
                        f"{len(evidence_items)} evidence passage(s) "
                        "returned by the verifier, ordered by NLI signal."
                    )

                    for i, ev in enumerate(evidence_items, start=1):
                        with st.container(border=True):
                            source = ev.get("source", "Unknown source")
                            page = ev.get("page", "?")

                            st.markdown(
                                f"**Passage {i}** &nbsp; "
                                f"{html.escape(str(source))}, page "
                                f"{html.escape(str(page))}",
                                unsafe_allow_html=True,
                            )

                            evidence_text = ev.get("text", "")

                            if evidence_text:
                                st.markdown(
                                    text_block("vf-evidence", evidence_text),
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.warning(
                                    "No text was available for this passage."
                                )

                            retrieval_score = ev.get("retrieval_score")

                            if retrieval_score is not None:
                                st.caption(
                                    "Normalized retrieval ranking score: "
                                    f"{retrieval_score:.3f}"
                                )

                            s1, s2, s3 = st.columns(3)
                            s1.metric(
                                "Entailment",
                                f'{ev.get("entailment", 0):.3f}',
                            )
                            s2.metric(
                                "Contradiction",
                                f'{ev.get("contradiction", 0):.3f}',
                            )
                            s3.metric(
                                "Neutral",
                                f'{ev.get("neutral", 0):.3f}',
                            )

                            st.caption(
                                "Passage classification: "
                                f'{ev.get("nli_label", "N/A")}'
                            )
                else:
                    st.warning(
                        "No evidence passages were retrieved for this claim. "
                        "The claim cannot be verified from the retrieved "
                        "evidence."
                    )

        st.divider()
        st.subheader("Export verification report")
        st.caption(
            "CSV with one row per claim: verdict, confidence signal and "
            "explanation, as produced by the existing report function."
        )

        st.download_button(
            "Download claim report (CSV)",
            data=results_to_csv(rows),
            file_name="fact_verification_report.csv",
            mime="text/csv",
            key="download_claim_report",
        )

        with st.expander("Raw verification table"):
            st.dataframe(
                df.drop(columns=["evidence"], errors="ignore"),
                use_container_width=True,
                hide_index=True,
            )

# ==================================================
# TAB 3: AUTOMATIC HALLUCINATION CORRECTION
# ==================================================

with tab_correct:
    st.subheader("Automatic hallucination correction")

    if not rows:
        st.info(
            "Run a verification first. Corrections are generated from "
            "the verified claims and their evidence."
        )
    else:
        st.write(
            "Generate a revised answer using evidence from your uploaded "
            "PDFs. Supported claims are preserved, proposed corrections are "
            "checked again by the NLI verifier, and unsupported claims "
            "remain unverified."
        )

        if st.button(
            "Generate corrected answer using Gemini",
            type="primary",
            key="generate_correction",
        ):
            try:
                with st.spinner(
                    "Generating and validating evidence-grounded "
                    "corrections..."
                ):
                    _, correction_verifier = load_models()

                    correction_result = generate_corrected_answer(
                        rows,
                        correction_verifier,
                    )

                    st.session_state["corrected_answer"] = correction_result

            except Exception as exc:
                st.error(f"Correction failed: {exc}")

        correction_result = st.session_state.get("corrected_answer")

        if correction_result:
            st.markdown("### Corrected answer")
            st.markdown(
                text_block("vf-answer", correction_result["corrected_answer"]),
                unsafe_allow_html=True,
            )
            st.caption(
                "Corrections that the NLI verifier did not confirm are not "
                "verified. Review the audit table before using this text."
            )

            st.download_button(
                "Download corrected answer",
                data=correction_result["corrected_answer"],
                file_name="corrected_answer.txt",
                mime="text/plain",
                key="download_corrected_answer",
            )

            st.markdown("### Correction audit")
            st.dataframe(
                pd.DataFrame(correction_result["audit"]),
                use_container_width=True,
                hide_index=True,
            )

# ==================================================
# TAB 4: EVALUATION DASHBOARD
# ==================================================

with tab_eval:
    st.subheader("Evaluation dashboard")

    initial = st.session_state.get("evaluation_initial")

    if not rows or initial is None:
        st.info(
            "No claims to evaluate yet. Run a verification first, then "
            "label each claim here."
        )
    else:
        st.write(
            "Compare predicted verdicts with manually verified ground-truth "
            "labels. Only labeled claims are included in the metrics."
        )

        st.info(
            "Assign ground-truth verdicts independently by checking the "
            "source document or reliable evidence. Do not simply copy model "
            "predictions."
        )

        edited = st.data_editor(
            initial,
            key="evaluation_editor",
            hide_index=True,
            use_container_width=True,
            disabled=[
                "Claim ID",
                "Claim",
                "Predicted verdict",
            ],
            column_config={
                "Claim ID": st.column_config.NumberColumn("Claim ID"),
                "Claim": st.column_config.TextColumn(
                    "Claim",
                    width="large",
                ),
                "Predicted verdict": st.column_config.TextColumn(
                    "Predicted verdict",
                ),
                "Ground truth": st.column_config.SelectboxColumn(
                    "Ground truth",
                    options=["UNLABELED"] + LABELS,
                    required=True,
                    help=(
                        "Select the correct verdict based on the source "
                        "document."
                    ),
                ),
            },
        )

        labeled = edited[
            edited["Ground truth"].isin(LABELS)
            & edited["Predicted verdict"].isin(LABELS)
        ].copy()

        st.caption(
            f"{len(labeled)} of {len(edited)} claims labeled for evaluation."
        )

        if len(labeled) > 0:
            y_true = labeled["Ground truth"]
            y_pred = labeled["Predicted verdict"]

            accuracy = accuracy_score(y_true, y_pred)

            report = classification_report(
                y_true,
                y_pred,
                labels=LABELS,
                output_dict=True,
                zero_division=0,
            )

            macro_precision = report["macro avg"]["precision"]
            macro_recall = report["macro avg"]["recall"]
            macro_f1 = report["macro avg"]["f1-score"]

            st.markdown("### Performance metrics")

            m1, m2, m3, m4 = st.columns(4)
            with m1.container(border=True):
                st.metric("Accuracy", f"{accuracy:.1%}")
            with m2.container(border=True):
                st.metric("Macro precision", f"{macro_precision:.1%}")
            with m3.container(border=True):
                st.metric("Macro recall", f"{macro_recall:.1%}")
            with m4.container(border=True):
                st.metric("Macro F1-score", f"{macro_f1:.1%}")

            st.caption(
                "Macro averages give equal weight to each of the three "
                "verdict classes, including classes not represented in this "
                "sample."
            )

            st.markdown("### Per-class performance")

            per_class = pd.DataFrame([
                {
                    "Verdict": label,
                    "Precision": report[label]["precision"],
                    "Recall": report[label]["recall"],
                    "F1-score": report[label]["f1-score"],
                    "Support": int(report[label]["support"]),
                }
                for label in LABELS
            ])

            st.dataframe(
                per_class.style.format({
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1-score": "{:.3f}",
                }),
                hide_index=True,
                use_container_width=True,
            )

            st.markdown("### Confusion matrix")

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=LABELS,
            )

            cm_df = pd.DataFrame(
                cm,
                index=[f"Actual: {label}" for label in LABELS],
                columns=[f"Predicted: {label}" for label in LABELS],
            )

            st.dataframe(
                cm_df,
                use_container_width=True,
            )

            st.caption(
                "Rows represent the actual verdict; columns represent the "
                "predicted verdict. Values on the diagonal are correct "
                "predictions."
            )

            st.markdown("### Download evaluation results")
            st.caption("CSV of the labeled claims only.")

            evaluation_csv = labeled.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download labeled evaluation CSV",
                data=evaluation_csv,
                file_name="evaluation_results.csv",
                mime="text/csv",
                key="download_evaluation_csv",
            )

        else:
            st.warning(
                "Select a ground-truth verdict for at least one claim to "
                "calculate evaluation metrics. For meaningful results, label "
                "a representative set of claims from all three classes."
            )

# --------------------------------------------------
# SIDEBAR PROGRESS (rendered last so it reflects this run)
# --------------------------------------------------

with st.sidebar:
    st.divider()
    st.markdown("**Progress**")
    st.write(
        f'{"Done" if st.session_state.get("verification_rows") else "Pending"}'
        " - Claims verified"
    )
    st.write(
        f'{"Done" if st.session_state.get("corrected_answer") else "Pending"}'
        " - Answer corrected"
    )