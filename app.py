import streamlit as st
from pathlib import Path
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="CNN Visual Analysis · Aida Hamzic",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_CSV = PROJECT_ROOT / "outputs" / "curated_manifest.csv"
EVAL_CSV = PROJECT_ROOT / "outputs" / "semantic_evaluated_predictions.csv"
VIS_DIR = PROJECT_ROOT / "outputs" / "visualisations"

CATEGORY_DISPLAY = {
    "all_correct": "All Three Correct",
    "all_wrong_two_agree": "All Three Wrong",
    "two_correct_one_wrong": "Two Correct, One Wrong",
    "two_wrong_agree_one_correct": "One Correct, Two Wrong",
}

CATEGORY_DESCRIPTIONS = {
    "all_correct": "All three architectures correctly identify the scene class. Grad-CAM reveals whether they attend to the same image regions despite reaching identical answers;architectural consensus made visible.",
    "all_wrong_two_agree": "All three models misclassify the image. Two converge on the same wrong class, exposing a shared blind spot across architectures. The heatmaps show what misled them.",
    "two_correct_one_wrong": "Two architectures succeed while one fails. The diverging Grad-CAM shows exactly what the failing model attends to instead a direct window into architectural difference.",
    "two_wrong_agree_one_correct": "Two models converge on the same incorrect prediction while one succeeds. This reveals which architectural design resists the shared failure pattern and why.",
}

MODEL_DISPLAY = {"vgg16": "VGG16", "resnet18": "ResNet-18", "mobilenetv2": "MobileNetV2"}
LAYER_LABELS = {"early": "Early Layer", "middle": "Middle Layer", "late": "Late Layer"}
LAYER_DESC = {"early": "Edges & textures", "middle": "Patterns & shapes", "late": "Semantic features"}
MODELS = ["vgg16", "resnet18", "mobilenetv2"]

C_BG = "#F5F5F3"
C_SIDEBAR = "#F0EEF0"
C_WHITE = "#FFFFFF"
C_TEXT = "#1E2420"
C_SEC = "#99919D"
C_LIGHT = "#B0A8B5"
C_ACCENT = "#9A8FA0"
C_BORDER = "#BAC7BE"
C_CORRECT = "#3D6B47"
C_WRONG = "#8B3A2E"


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400&display=swap');

    html, body, .stApp {{ background-color: {C_BG} !important; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding: 2.5rem 3rem 4rem 3rem; max-width: 1280px; }}
    * {{ font-family: 'Montserrat', sans-serif; }}

    [data-testid="stSidebar"] {{
        border-right: 1px solid {C_BORDER};
    }}
    [data-testid="stSidebar"] label {{
        font-size: 0.6rem !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: {C_ACCENT} !important;
    }}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
        font-size: 0.82rem !important;
        font-weight: 400 !important;
        color: {C_TEXT} !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: {C_BORDER} !important;
        margin: 1rem 0 !important;
    }}

    .sb-eyebrow {{
        font-size: 0.58rem;
        font-weight: 600;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: {C_ACCENT};
        margin-bottom: 8px;
    }}
    .sb-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {C_TEXT};
        line-height: 1.3;
        margin-bottom: 8px;
        letter-spacing: -0.2px;
    }}
    .sb-byline {{
        font-size: 0.74rem;
        font-weight: 300;
        color: {C_SEC};
        line-height: 1.6;
    }}
    .sb-desc {{
        font-size: 0.76rem;
        font-weight: 300;
        color: {C_SEC};
        line-height: 1.65;
        font-style: italic;
    }}

    .pg-eyebrow {{
        font-size: 0.58rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: {C_ACCENT};
        margin-bottom: 12px;
    }}
    .pg-title {{
        font-size: 1.9rem;
        font-weight: 700;
        color: {C_TEXT};
        line-height: 1.2;
        letter-spacing: -0.5px;
        margin-bottom: 12px;
    }}
    .pg-rq {{
        font-size: 0.92rem;
        font-weight: 300;
        font-style: italic;
        color: {C_SEC};
        line-height: 1.65;
        max-width: 800px;
        margin-bottom: 10px;
    }}
    .pg-meta {{
        font-size: 0.6rem;
        font-weight: 500;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: {C_LIGHT};
    }}
    .pg-rule {{
        width: 100%;
        height: 1px;
        background: {C_BORDER};
        border: none;
        margin: 20px 0 24px 0;
    }}

    .info-bar {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 18px;
        background: {C_WHITE};
        border: 1px solid {C_BORDER};
        border-left: 3px solid {C_ACCENT};
        border-radius: 4px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }}
    .info-class {{
        font-size: 0.88rem;
        font-weight: 700;
        color: {C_TEXT};
    }}
    .info-scene {{
        font-size: 0.84rem;
        font-weight: 300;
        font-style: italic;
        color: {C_SEC};
    }}
    .info-chip {{
        background: transparent;
        border: 1px solid {C_BORDER};
        border-radius: 3px;
        padding: 3px 10px;
        font-size: 0.58rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: {C_SEC};
    }}
    .info-id {{
        font-size: 0.6rem;
        color: {C_LIGHT};
        margin-left: auto;
    }}

    .sec-eyebrow {{
        font-size: 0.58rem;
        font-weight: 600;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: {C_ACCENT};
        margin: 36px 0 4px 0;
    }}
    .sec-title {{
        font-size: 1.2rem;
        font-weight: 700;
        color: {C_TEXT};
        margin-bottom: 4px;
        letter-spacing: -0.2px;
    }}
    .sec-rule {{
        height: 1px;
        background: {C_BORDER};
        border: none;
        margin: 8px 0 12px 0;
    }}
    .sec-desc {{
        font-size: 0.82rem;
        font-weight: 300;
        color: {C_SEC};
        line-height: 1.7;
        max-width: 900px;
        margin-bottom: 18px;
    }}

    .model-name {{
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: {C_TEXT};
        text-align: center;
        padding-bottom: 8px;
        border-bottom: 2px solid {C_ACCENT};
        margin-bottom: 10px;
    }}
    .badge-correct {{
        display: inline-block;
        border: 1.5px solid {C_CORRECT};
        color: {C_CORRECT};
        border-radius: 3px;
        padding: 4px 12px;
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        background: transparent;
    }}
    .badge-wrong {{
        display: inline-block;
        border: 1.5px solid {C_WRONG};
        color: {C_WRONG};
        border-radius: 3px;
        padding: 4px 12px;
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        background: transparent;
    }}
    .pred-conf {{
        font-size: 0.68rem;
        font-weight: 500;
        color: {C_SEC};
        text-align: center;
        margin: 5px 0 2px 0;
    }}
    .pred-raw {{
        font-size: 0.7rem;
        font-weight: 300;
        font-style: italic;
        color: {C_LIGHT};
        text-align: center;
        margin-bottom: 8px;
    }}

    .layer-header {{
        font-size: 0.6rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: {C_TEXT};
        text-align: center;
        margin-bottom: 2px;
    }}
    .layer-sub {{
        font-size: 0.7rem;
        font-weight: 300;
        font-style: italic;
        color: {C_LIGHT};
        text-align: center;
        margin-bottom: 10px;
    }}
    .fmap-label {{
        font-size: 0.58rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: {C_SEC};
        text-align: center;
        background: {C_WHITE};
        border: 1px solid {C_BORDER};
        border-radius: 3px;
        padding: 8px 6px;
        width: 100%;
    }}

    .stat-card {{
        background: {C_WHITE};
        border: 1px solid {C_BORDER};
        border-top: 2px solid {C_ACCENT};
        border-radius: 4px;
        padding: 22px 16px;
        text-align: center;
    }}
    .stat-eyebrow {{
        font-size: 0.58rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: {C_ACCENT};
        margin-bottom: 10px;
    }}
    .stat-value {{
        font-size: 2.6rem;
        font-weight: 700;
        color: {C_TEXT};
        line-height: 1;
        margin-bottom: 4px;
        letter-spacing: -1px;
    }}
    .stat-label {{
        font-size: 0.72rem;
        font-weight: 300;
        font-style: italic;
        color: {C_SEC};
    }}
    .stat-top5 {{
        font-size: 0.62rem;
        font-weight: 500;
        color: {C_LIGHT};
        margin-top: 8px;
    }}

    [data-testid="stImage"] img {{
        border-radius: 4px;
        border: 1px solid {C_BORDER};
    }}
    [data-testid="stExpander"] {{
        background: {C_WHITE} !important;
        border: 1px solid {C_BORDER} !important;
        border-radius: 4px !important;
    }}
    [data-testid="stExpander"] summary {{
        font-size: 0.6rem !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: {C_SEC} !important;
    }}
    .disclaimer {{
        font-size: 0.74rem;
        font-weight: 300;
        font-style: italic;
        color: {C_LIGHT};
        line-height: 1.65;
        margin-top: 14px;
        padding-top: 14px;
        border-top: 1px solid {C_BORDER};
    }}
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_CSV)
    df["image_stem"] = df["image_id"].apply(lambda x: Path(str(x)).stem)
    return df


@st.cache_data
def load_accuracy_stats():
    df = pd.read_csv(EVAL_CSV)
    top1 = df.groupby("model_name")["semantic_correct"].mean()
    top5 = df.groupby("model_name")["top5_contains_true_class"].mean()
    by_class = df.groupby(["thesis_class", "model_name"])["semantic_correct"].mean().unstack()
    return top1, top5, by_class


def safe_img(path: Path):
    try:
        return Image.open(path).convert("RGB") if path.exists() else None
    except Exception:
        return None


def safe_val(row, col, default="—"):
    try:
        v = row[col]
        return default if pd.isna(v) else v
    except Exception:
        return default


def render_sidebar(manifest: pd.DataFrame):
    with st.sidebar:
        st.markdown("""
        <div style='padding: 22px 4px 16px 4px;'>
            <div class='sb-eyebrow'>Bachelor Thesis · 2026</div>
            <div class='sb-title'>CNN Visual Analysis</div>
            <div class='sb-byline'>Aida Hamzic<br>Department of Computer Science<br>Lucerne University of Applied Sciences</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        classes = sorted(manifest["thesis_class"].unique())
        sel_class = st.selectbox(
            "Scene Class",
            classes,
            format_func=lambda c: c.capitalize(),
        )

        avail_cats = manifest[
            manifest["thesis_class"] == sel_class
            ]["category_name"].unique()
        cat_map = {CATEGORY_DISPLAY.get(c, c): c for c in sorted(avail_cats)}
        sel_cat_display = st.selectbox("Analysis Category", list(cat_map.keys()))
        sel_cat = cat_map[sel_cat_display]

        subset = manifest[
            (manifest["thesis_class"] == sel_class) &
            (manifest["category_name"] == sel_cat)
            ].reset_index(drop=True)

        sel_idx = st.radio(
            "Image",
            range(len(subset)),
            format_func=lambda i: (
                f"{i + 1}  ·  "
                f"{str(subset.iloc[i]['scene_label']).replace('-', ' ').replace('_', ' ').title()[:28]}"
            ),
        )

        st.divider()

        desc = CATEGORY_DESCRIPTIONS.get(sel_cat, "")
        st.markdown(f"<div class='sb-desc'>{desc}</div>", unsafe_allow_html=True)

    return subset, sel_idx


def render_main(row: pd.Series):
    stem = row["image_stem"]
    vis = VIS_DIR / stem
    scene = str(row["scene_label"]).replace("-", " ").replace("_", " ").title()
    cat_d = CATEGORY_DISPLAY.get(row["category_name"], row["category_name"])
    avg_c = float(safe_val(row, "avg_confidence", 0))

    st.markdown("""
    <div class='pg-eyebrow'>Bachelor Thesis · 2026</div>
    <div class='pg-title'>Visual Analysis of Convolutional Neural Network Decision Processes<br>for Image Recognition and Classification</div>
    <div class='pg-rq'>How can different convolutional neural network architectures be visually analysed and compared in terms of their internal feature-extraction and decision processes?</div>
    <div class='pg-meta'>Department of Computer Science &nbsp;·&nbsp; Lucerne University of Applied Sciences &nbsp;·&nbsp; Aida Hamzic</div>
    <hr class='pg-rule'>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='info-bar'>
        <span class='info-class'>{row['thesis_class'].capitalize()}</span>
        <span style='color:{C_BORDER};font-size:0.7rem;'>·</span>
        <span class='info-scene'>{scene}</span>
        <span class='info-chip'>{cat_d}</span>
        <span class='info-chip'>Avg {avg_c * 100:.1f}% confidence</span>
        <span class='info-id'>{row['image_id']}</span>
    </div>
    """, unsafe_allow_html=True)

    orig = safe_img(vis / "original.jpg")
    _, col_img, _ = st.columns([2, 3, 2])
    with col_img:
        if orig:
            st.image(orig, caption=f"{scene}  ·  Places365 Validation Set",
                     use_column_width=True)
        else:
            st.warning("Original image not found.")

    st.markdown("<div class='sec-eyebrow'>Gradient-Weighted Class Activation Mapping</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Grad-CAM · Spatial Decision Analysis</div>",
                unsafe_allow_html=True)
    st.markdown("<hr class='sec-rule'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sec-desc'>Heatmap overlays produced by computing the gradient of the "
        "predicted class score with respect to the final convolutional layer activations "
        "(Selvaraju et al., 2019). Red regions indicate the highest model attention; "
        "blue the lowest. Each architecture is explained independently under identical "
        "input conditions.</div>",
        unsafe_allow_html=True,
    )

    gcols = st.columns(3)
    for i, model in enumerate(MODELS):
        pred = safe_val(row, f"pred_{model}")
        correct = bool(safe_val(row, f"correct_{model}", False))
        conf = float(safe_val(row, f"conf_{model}", 0))
        raw = safe_val(row, f"top1_label_{model}")
        pred_d = str(pred).capitalize() if pred != "—" else "—"
        badge = "badge-correct" if correct else "badge-wrong"
        label = "Correct" if correct else "Incorrect"

        with gcols[i]:
            st.markdown(f"<div class='model-name'>{MODEL_DISPLAY[model]}</div>",
                        unsafe_allow_html=True)
            st.markdown(
                f"<div style='text-align:center;margin-bottom:4px;'>"
                f"<span class='{badge}'>{label} — {pred_d}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<div class='pred-conf'>{conf * 100:.1f}% confidence</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='pred-raw'>ImageNet: {raw}</div>",
                        unsafe_allow_html=True)
            img = safe_img(vis / f"{model}_gradcam.png")
            if img:
                st.image(img, use_column_width=True)
            else:
                st.markdown(
                    f"<p style='font-size:0.75rem;color:{C_WRONG};text-align:center;'>Not found</p>",
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-eyebrow'>Internal Feature Representations</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Feature Maps</div>", unsafe_allow_html=True)
    st.markdown("<hr class='sec-rule'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sec-desc'>Activations captured at three structural depth stages via "
        "forward hooks. Early layers respond to low-level edges and textures; middle layers "
        "encode partial patterns and recurring shapes; late layers carry the most abstract "
        "semantic representations prior to classification. Each grid shows 16 channels, "
        "individually normalised.</div>",
        unsafe_allow_html=True,
    )

    _, c_e, c_m, c_l = st.columns([0.6, 3, 3, 3])
    for col, stage in [(c_e, "early"), (c_m, "middle"), (c_l, "late")]:
        with col:
            st.markdown(
                f"<div class='layer-header'>{LAYER_LABELS[stage]}</div>"
                f"<div class='layer-sub'>{LAYER_DESC[stage]}</div>",
                unsafe_allow_html=True,
            )

    for model in MODELS:
        c_lbl, c_e, c_m, c_l = st.columns([0.6, 3, 3, 3])
        with c_lbl:
            st.markdown(
                f"<div style='height:100%;display:flex;align-items:center;'>"
                f"<div class='fmap-label'>{MODEL_DISPLAY[model]}</div></div>",
                unsafe_allow_html=True,
            )
        for col, stage in [(c_e, "early"), (c_m, "middle"), (c_l, "late")]:
            with col:
                img = safe_img(vis / f"{model}_features_{stage}.png")
                if img:
                    st.image(img, use_column_width=True)
                else:
                    st.markdown(
                        f"<p style='font-size:0.75rem;color:{C_WRONG};text-align:center;'>Missing</p>",
                        unsafe_allow_html=True,
                    )

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    with st.expander("Model Performance Overview"):
        try:
            top1, top5, by_class = load_accuracy_stats()

            st.markdown(
                f"<div style='font-size:0.58rem;font-weight:600;letter-spacing:2px;"
                f"text-transform:uppercase;color:{C_ACCENT};margin-bottom:16px;'>"
                f"Semantic Top-1 Accuracy</div>",
                unsafe_allow_html=True,
            )

            sc = st.columns(3)
            for i, model in enumerate(MODELS):
                acc = float(top1.get(model, 0))
                acc5 = float(top5.get(model, 0))
                with sc[i]:
                    st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-eyebrow'>{MODEL_DISPLAY[model]}</div>
                        <div class='stat-value'>{acc * 100:.1f}%</div>
                        <div class='stat-label'>Top-1 accuracy</div>
                        <div class='stat-top5'>Top-5 &nbsp;·&nbsp; {acc5 * 100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:0.58rem;font-weight:600;letter-spacing:2px;"
                f"text-transform:uppercase;color:{C_ACCENT};margin-bottom:12px;'>"
                f"Accuracy by Scene Class</div>",
                unsafe_allow_html=True,
            )

            disp = by_class.copy()
            disp.columns = [MODEL_DISPLAY.get(c, c) for c in disp.columns]
            disp.index = [c.capitalize() for c in disp.index]
            disp = (disp * 100).round(1)
            disp.columns = [f"{c} (%)" for c in disp.columns]
            st.dataframe(disp)

            st.markdown("""
            <div class='disclaimer'>
            Accuracy reflects semantic top-1 mapping from ImageNet-1K labels to Places365 scene
            categories. Low values for forest (12%) and glacier (14%) reflect cross-domain vocabulary
            mismatch — ImageNet contains no forest or glacier scene class. This is discussed as a
            methodological finding in the thesis. Mountain (78%) and sea (62%) perform substantially
            better due to stronger ImageNet vocabulary alignment.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not load accuracy stats: {e}")


def main():
    inject_css()

    if not MANIFEST_CSV.exists():
        st.error(f"Manifest not found: {MANIFEST_CSV}")
        st.stop()
    if not VIS_DIR.exists():
        st.error(f"Visualisations folder not found: {VIS_DIR}")
        st.stop()

    manifest = load_manifest()
    subset, sel_idx = render_sidebar(manifest)

    if subset.empty:
        st.warning("No images available for this selection.")
        st.stop()

    render_main(subset.iloc[sel_idx])


if __name__ == "__main__":
    main()