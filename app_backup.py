
import streamlit as st
import pandas as pd
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

st.set_page_config(
    page_title="AI Hallucination Detector",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 AI Hallucination Detector & Fact Verification")
st.caption(
    "Verify AI-generated research claims against evidence "
    "from your uploaded papers."
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
# USER INPUT
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload research papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
)

answer = st.text_area(
    "Paste the AI-generated research response",
    height=220,
    placeholder="Paste the answer you want to verify.",
)

run = st.button(
    "Verify claims",
    type="primary",
    disabled=not uploaded_files or not answer.strip(),
)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    retriever = HybridRetriever()
    verifier = NliVerifier()
    return retriever, verifier


# --------------------------------------------------
# PHASE 1: RUN VERIFICATION
# --------------------------------------------------

if run:
    try:
        with st.spinner("Extracting text from PDFs..."):
            all_pages = []

            for file in uploaded_files:
                pages = extract_pdf_pages(
                    file.getvalue(),
                    file.name,
                )
                all_pages.extend(pages)

            chunks = chunk_pages(all_pages)

            if not chunks:
                st.error(
                    "No extractable text found. "
                    "Scanned-only PDFs are not supported by this version."
                )
                st.stop()

        with st.spinner("Loading models and indexing evidence..."):
            retriever, verifier = load_models()
            retriever.fit(chunks)

        claims = split_into_claims(answer)

        if not claims:
            st.warning("No claims could be extracted.")
            st.stop()

        rows = []

        with st.spinner(f"Verifying {len(claims)} claims..."):
            for claim_id, claim in enumerate(claims, start=1):
                evidence = retriever.search(claim, top_k=top_k)
                result = verify_claim(claim, evidence, verifier)
                result["claim_id"] = claim_id
                rows.append(result)

        # Preserve verification results across Streamlit reruns.
        st.session_state["verification_rows"] = rows

        st.session_state.pop("corrected_answer", None)

        # Initialize ground-truth labels for a new evaluation.
        st.session_state.pop("evaluation_editor", None)

        st.session_state["evaluation_initial"] = pd.DataFrame({
            "Claim ID": [r["claim_id"] for r in rows],
            "Claim": [r["claim"] for r in rows],
            "Predicted verdict": [r["verdict"] for r in rows],
            "Ground truth": ["UNLABELED"] * len(rows),
        })

    except Exception as exc:
        st.error(f"Verification failed: {exc}")
        st.exception(exc)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "verification_rows" in st.session_state:
    rows = st.session_state["verification_rows"]

    # --------------------------------------------------
    # PHASE 4: AUTOMATIC HALLUCINATION CORRECTION
    # --------------------------------------------------

    st.divider()
    st.subheader("Automatic Hallucination Correction")

    st.write(
        "Generate a revised answer using evidence from your uploaded PDFs. "
        "Supported claims are preserved, proposed corrections are checked "
        "again by the NLI verifier, and unsupported claims remain unverified."
    )

    if st.button(
        "Generate corrected answer using Gemini",
        type="primary",
        key="generate_correction",
    ):
        try:
            with st.spinner(
                "Generating and validating evidence-grounded corrections..."
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

        st.code(
            correction_result["corrected_answer"],
            language=None,
        )

        st.download_button(
            "Download corrected answer",
            data=correction_result["corrected_answer"],
            file_name="corrected_answer.txt",
            mime="text/plain",
            key="download_corrected_answer",
        )

        with st.expander("Correction audit"):
            st.dataframe(
                pd.DataFrame(correction_result["audit"]),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    # Your existing code continues below
    df = pd.DataFrame(rows)

    df = pd.DataFrame(rows)
    counts = df["verdict"].value_counts().to_dict()

    st.divider()
    st.header("1. Verification Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Claims checked", len(df))
    c2.metric("Supported", counts.get("SUPPORTED", 0))
    c3.metric("Contradicted", counts.get("CONTRADICTED", 0))
    c4.metric("Unverified", counts.get("UNVERIFIED", 0))

    st.caption(
        "The confidence signal is derived from NLI model scores. "
        "It is not a calibrated probability that a claim is true."
    )

    # --------------------------------------------------
    # CLAIMS AND EVIDENCE
    # --------------------------------------------------

    for row in rows:
        verdict = row["verdict"]

        with st.expander(
            f'{row["claim_id"]}. [{verdict}] {row["claim"]}',
            expanded=False,
        ):
            st.markdown("**Claim assessment**")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted verdict",
                verdict,
            )

            col2.metric(
                "NLI confidence signal",
                f'{row["confidence"]:.3f}',
            )

            st.info(row["explanation"])

            st.divider()
            st.markdown("### Evidence and source traceability")

            evidence_items = row.get("evidence", [])

            if evidence_items:
                st.caption(
                    f"{len(evidence_items)} evidence passage(s) "
                    "returned by the verifier, ordered by NLI signal."
                )

                for i, ev in enumerate(evidence_items, start=1):
                    with st.container(border=True):
                        st.markdown(f"**Evidence passage {i}**")

                        source = ev.get("source", "Unknown source")
                        page = ev.get("page", "?")

                        st.caption(
                            f"Source: {source}  |  Page: {page}"
                        )

                        evidence_text = ev.get("text", "")

                        if evidence_text:
                            st.markdown("**Passage text**")
                            st.write(evidence_text)
                        else:
                            st.warning(
                                "No text was available for this passage."
                            )

                        retrieval_score = ev.get("retrieval_score")

                        if retrieval_score is not None:
                            st.caption(
                                f"Normalized retrieval ranking score: "
                                f"{retrieval_score:.3f}"
                            )

                        st.markdown("**NLI model scores**")

                        score1, score2, score3 = st.columns(3)

                        score1.metric(
                            "Entailment",
                            f'{ev.get("entailment", 0):.3f}',
                        )

                        score2.metric(
                            "Contradiction",
                            f'{ev.get("contradiction", 0):.3f}',
                        )

                        score3.metric(
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
                    "The claim cannot be verified from the retrieved evidence."
                )

    # --------------------------------------------------
    # EXPORT VERIFICATION REPORT
    # --------------------------------------------------

    st.divider()
    st.subheader("Export verification report")

    st.download_button(
        "Download claim report (CSV)",
        data=results_to_csv(rows),
        file_name="fact_verification_report.csv",
        mime="text/csv",
    )

    with st.expander("Raw verification table"):
        st.dataframe(
            df.drop(columns=["evidence"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------
    # PHASE 2: EVALUATION DASHBOARD
    # --------------------------------------------------

    st.divider()
    st.header("2. Evaluation Dashboard")

    st.write(
        "Compare predicted verdicts with manually verified ground-truth "
        "labels. Only labeled claims are included in the metrics."
    )

    st.info(
        "Assign ground-truth verdicts independently by checking the source "
        "document or reliable evidence. Do not simply copy model predictions."
    )

    initial = st.session_state["evaluation_initial"]

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
                    "Select the correct verdict based on the source document."
                ),
            ),
        },
    )

    labeled = edited[
        edited["Ground truth"].isin(LABELS)
        & edited["Predicted verdict"].isin(LABELS)
    ].copy()

    st.caption(
        f'{len(labeled)} of {len(edited)} claims labeled for evaluation.'
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

        st.subheader("Performance Metrics")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Accuracy", f"{accuracy:.1%}")
        m2.metric("Macro Precision", f"{macro_precision:.1%}")
        m3.metric("Macro Recall", f"{macro_recall:.1%}")
        m4.metric("Macro F1-score", f"{macro_f1:.1%}")

        st.caption(
            "Macro averages give equal weight to each of the three verdict "
            "classes, including classes not represented in this sample."
        )

        st.subheader("Per-class Performance")

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

        st.subheader("Confusion Matrix")

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
            "predicted verdict. Values on the diagonal are correct predictions."
        )

        st.subheader("Download Evaluation Results")

        evaluation_csv = labeled.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download labeled evaluation CSV",
            data=evaluation_csv,
            file_name="evaluation_results.csv",
            mime="text/csv",
        )

    else:
        st.warning(
            "Select a ground-truth verdict for at least one claim to "
            "calculate evaluation metrics. For meaningful results, label "
            "a representative set of claims from all three classes."
        )
