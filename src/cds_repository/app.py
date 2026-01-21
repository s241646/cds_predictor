import streamlit as st
import requests
import pandas as pd

# --- Configuration ---
API_URL = "http://localhost:8000"
DISPLAY_LIMIT = 1000

st.set_page_config(page_title="CDS Predictor", page_icon="🧬", layout="wide")

# --- Polished Dark Theme CSS ---
st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    [data-testid="stMetricLabel"] { color: #808495 !important; }
    div[data-testid="metric-container"] {
        background-color: #1E232D;
        border: 1px solid #30363D;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #238636;
        color: white;
        border: none;
    }
    .truncation-warning {
        padding: 10px;
        background-color: #3b2e00;
        border-left: 5px solid #ffcc00;
        color: #ffcc00;
        margin-bottom: 10px;
        border-radius: 4px;
        font-size: 0.9em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.title("🧬 CDS Predictor")
    st.markdown("---")

    try:
        health = requests.get(f"{API_URL}/health").json()
        if health["status"] == "ok":
            st.success(f"System Ready: {health.get('device', 'CPU')}")
    except requests.RequestException:  # Catching specific connection/timeout errors
        st.error("API Connection Failed")
    except Exception as e:  # Catching other unexpected errors safely
        st.error(f"Unexpected Error: {type(e).__name__}")

    st.subheader("Inference Settings")
    batch_size = st.slider("Batch Size", 1, 256, 32)
    return_probs = st.checkbox("Show Probabilities", value=True)
    return_logits = st.checkbox("Show Logits", value=False)

# --- Main Interface ---
st.header("Sequence Inference")

file = st.file_uploader("Drop a FASTA file here", type=["fasta", "fa"])

if file:
    if st.button("Run Prediction"):
        with st.spinner("Analyzing sequences..."):
            try:
                # Prepare payload exactly as expected by your FastAPI
                files = {"fasta": (file.name, file.getvalue(), "text/plain")}
                data = {
                    "batch_size": batch_size,
                    "return_probs": str(return_probs).lower(),
                    "return_logits": str(return_logits).lower(),
                }

                response = requests.post(f"{API_URL}/predict", files=files, data=data)
                response.raise_for_status()

                full_results = response.json()["results"]
                df_full = pd.DataFrame(full_results)
                total_count = len(df_full)

                # --- Metrics Summary ---
                st.markdown("### Summary")
                m1, m2, m3 = st.columns(3)
                positives = df_full["pred"].sum()

                m1.metric("Total Sequences", total_count)
                m2.metric("CDS Detected", positives)
                m3.metric("Detection Rate", f"{(positives / total_count) * 100:.1f}%" if total_count > 0 else "0%")

                # --- Download Section ---
                csv = df_full.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Results as CSV",
                    data=csv,
                    file_name=f"predictions_{file.name}.csv",
                    mime="text/csv",
                    type="primary",
                )

                # --- Results Table Header Logic ---
                if total_count > DISPLAY_LIMIT:
                    st.markdown(f"### Preview (First {DISPLAY_LIMIT:,} Rows)")
                    st.markdown(
                        f'<div class="truncation-warning">⚠️ Showing first {DISPLAY_LIMIT:,} of {total_count:,} sequences for performance. Download the CSV for the full dataset.</div>',
                        unsafe_allow_html=True,
                    )
                    df_display = df_full.head(DISPLAY_LIMIT)
                else:
                    st.markdown("### Results")
                    df_display = df_full

                # Apply styling to display
                def highlight_cds(val):
                    color = "#1b4332" if val == 1 else "#431b1b"
                    return f"background-color: {color}"

                st.dataframe(df_display.style.applymap(highlight_cds, subset=["pred"]), use_container_width=True)

            except Exception as e:
                st.error(f"Prediction Error: {e}")

# --- Footer Metadata ---
with st.expander("Model Metadata"):
    try:
        info = requests.get(f"{API_URL}/info").json()
        st.json(info)
    except requests.RequestException:  # Catching specific connection/timeout errors
        st.write("Metadata unavailable: Could not connect to API.")
    except Exception:  # Catching other issues like JSON decoding errors
        st.write("Metadata unavailable: Invalid response.")
