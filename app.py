import base64
import json
from io import BytesIO

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
EVAL_CSV     = PROJECT_ROOT / "outputs" / "semantic_evaluated_predictions.csv"
VIS_DIR      = PROJECT_ROOT / "outputs" / "visualisations"

CATEGORY_DISPLAY = {
    "all_correct":               "All Three Correct",
    "all_wrong_two_agree":       "All Three Wrong",
    "two_correct_one_wrong":     "Two Correct, One Wrong",
    "two_wrong_agree_one_correct": "One Correct, Two Wrong",
}

CATEGORY_DESCRIPTIONS = {
    "all_correct": "All three models predict the correct thesis class. The Grad-CAM heatmaps show whether they attend to the same image regions or converge on the right answer through different spatial evidence.",
    "all_wrong_two_agree": "All three models predict an incorrect thesis class and at least two agree on the same wrong class. Images where all three predict different wrong classes are not included in this category.",
    "two_correct_one_wrong": "Two models predict the correct class; one does not. The Grad-CAM for the failing model shows which regions it attended to instead, making it possible to isolate where its spatial reasoning diverged.",
    "two_wrong_agree_one_correct": "Two models agree on the same wrong prediction while one gets it right. The Grad-CAM comparison shows what the correct model attends to that the other two do not.",
}

MODEL_DISPLAY  = {"vgg16": "VGG16", "resnet18": "ResNet-18", "mobilenetv2": "MobileNetV2"}
LAYER_LABELS   = {"early": "Early Layer", "middle": "Middle Layer", "late": "Late Layer"}
LAYER_DESC     = {"early": "Edges & textures", "middle": "Patterns & shapes", "late": "Semantic features"}
MODELS         = ["vgg16", "resnet18", "mobilenetv2"]

C_BG      = "#F5F5F3"
C_WHITE   = "#FFFFFF"
C_TEXT    = "#1E2420"
C_SEC     = "#99919D"
C_LIGHT   = "#B0A8B5"
C_ACCENT  = "#9A8FA0"
C_BORDER  = "#BAC7BE"
C_CORRECT = "#3D6B47"
C_WRONG   = "#8B3A2E"


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400&display=swap');

    html, body, .stApp {{ background-color: {C_BG} !important; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding: 2rem 3rem 4rem 3rem; max-width: 1280px; }}
    * {{ font-family: 'Montserrat', sans-serif; }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        width: 270px !important;
        min-width: 270px !important;
        max-width: 270px !important;
        border-right: 1px solid {C_BORDER};
    }}
    [data-testid="stSidebar"] label {{
        font-size: 0.6rem !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: {C_ACCENT} !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: {C_BORDER} !important;
        margin: 1rem 0 !important;
    }}
    /* Dot selector buttons */
    [data-testid="stSidebar"] .stButton button {{
        background: transparent !important;
        border: none !important;
        color: {C_ACCENT} !important;
        font-size: 0.85rem !important;
        padding: 2px 0 !important;
        min-height: 0 !important;
        height: 22px !important;
        width: 100% !important;
        line-height: 1 !important;
    }}
    [data-testid="stSidebar"] .stButton button:hover {{
        background: transparent !important;
        color: {C_TEXT} !important;
    }}

    /* ── Sidebar selectbox accent on focus ── */
    [data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {{
        border-color: {C_ACCENT} !important;
        box-shadow: 0 0 0 1px {C_ACCENT} !important;
    }}

    /* ── Sidebar branding ── */
    .sb-eyebrow {{
        font-size: 0.55rem; font-weight: 600; letter-spacing: 2.5px;
        text-transform: uppercase; color: {C_ACCENT}; margin-bottom: 6px;
    }}
    .sb-title {{
        font-size: 1.0rem; font-weight: 700; color: {C_TEXT};
        line-height: 1.3; margin-bottom: 6px; letter-spacing: -0.2px;
    }}
    .sb-byline {{
        font-size: 0.7rem; font-weight: 300; color: {C_SEC}; line-height: 1.6;
    }}

    /* ── Page title — LARGER ── */
    .pg-title {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {C_TEXT};
        line-height: 1.15;
        letter-spacing: -1px;
        margin-bottom: 16px;
    }}
    .pg-rule {{
        width: 100%; height: 1px; background: {C_BORDER};
        border: none; margin: 0 0 24px 0;
    }}

    /* ── Info bar ── */
    .info-bar {{
        display: flex; align-items: center; gap: 12px;
        padding: 12px 18px; background: {C_WHITE};
        border: 1px solid {C_BORDER}; border-left: 3px solid {C_ACCENT};
        border-radius: 4px; margin-bottom: 20px; flex-wrap: wrap;
    }}
    .info-class  {{ font-size: 0.92rem; font-weight: 700; color: {C_TEXT}; }}
    .info-scene  {{ font-size: 0.88rem; font-weight: 300; font-style: italic; color: {C_SEC}; }}
    .info-chip   {{
        background: transparent; border: 1px solid {C_BORDER}; border-radius: 3px;
        padding: 3px 10px; font-size: 0.58rem; font-weight: 600;
        letter-spacing: 1px; text-transform: uppercase; color: {C_SEC};
    }}
    .info-id {{ font-size: 0.6rem; color: {C_LIGHT}; margin-left: auto; }}

    /* ── Category context block ── */
    .cat-context {{
        background: {C_WHITE}; border: 1px solid {C_BORDER};
        border-left: 3px solid {C_ACCENT}; border-radius: 4px;
        padding: 12px 18px; margin-bottom: 24px;
        font-size: 0.84rem; font-weight: 300; font-style: italic;
        color: {C_SEC}; line-height: 1.7;
    }}

    /* ── Section headers — LARGER ── */
    .sec-eyebrow {{
        font-size: 0.6rem; font-weight: 600; letter-spacing: 2.5px;
        text-transform: uppercase; color: {C_ACCENT}; margin: 0 0 4px 0;
    }}
    .sec-title {{
        font-size: 1.55rem; font-weight: 700; color: {C_TEXT};
        margin-bottom: 4px; letter-spacing: -0.3px;
    }}
    .sec-rule {{ height: 1px; background: {C_BORDER}; border: none; margin: 8px 0 14px 0; }}
    .sec-desc {{
        font-size: 0.9rem; font-weight: 400; color: {C_SEC};
        line-height: 1.75; max-width: 900px; margin-bottom: 20px;
    }}

    /* ── Tabs styling ── */
    [data-baseweb="tab-list"] {{
        background: transparent !important;
        border-bottom: 2px solid {C_BORDER} !important;
        gap: 0 !important;
    }}
    [data-baseweb="tab"] {{
        font-size: 0.65rem !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: {C_LIGHT} !important;
        padding: 10px 20px !important;
        border-bottom: 2px solid transparent !important;
        margin-bottom: -2px !important;
        background: transparent !important;
    }}
    [data-baseweb="tab"]:hover {{
        color: {C_SEC} !important;
        background: transparent !important;
    }}
    [aria-selected="true"][data-baseweb="tab"] {{
        color: {C_TEXT} !important;
        border-bottom: 2px solid {C_ACCENT} !important;
        background: transparent !important;
    }}
    [data-baseweb="tab-panel"] {{ padding: 20px 0 0 0 !important; }}

    /* ── Model cards ── */
    .model-card {{
        background: {C_WHITE};
        border: 1px solid {C_BORDER};
        border-top: 3px solid {C_ACCENT};
        border-radius: 4px;
        padding: 16px 14px 0 14px;
        margin-bottom: 8px;
    }}
    .model-card-name {{
        font-size: 0.65rem; font-weight: 700; letter-spacing: 2.5px;
        text-transform: uppercase; color: {C_TEXT};
        text-align: center; margin-bottom: 10px;
    }}
    .badge-correct {{
        display: inline-block; border: 1.5px solid {C_CORRECT}; color: {C_CORRECT};
        border-radius: 3px; padding: 4px 12px; font-size: 0.65rem; font-weight: 700;
        letter-spacing: 1px; text-transform: uppercase; background: transparent;
    }}
    .badge-wrong {{
        display: inline-block; border: 1.5px solid {C_WRONG}; color: {C_WRONG};
        border-radius: 3px; padding: 4px 12px; font-size: 0.65rem; font-weight: 700;
        letter-spacing: 1px; text-transform: uppercase; background: transparent;
    }}
    .pred-conf {{
        font-size: 1.05rem; font-weight: 700; color: {C_TEXT};
        text-align: center; margin: 10px 0 4px 0;
    }}
    .pred-raw {{
        font-size: 0.65rem; font-weight: 400; color: {C_LIGHT};
        text-align: center; margin-bottom: 12px; letter-spacing: 0.2px;
    }}

    /* ── Stage selector (Feature Maps tab) ── */
    .stage-selector {{
        display: flex; gap: 8px; margin-bottom: 20px;
    }}

    /* ── Feature map label ── */
    .fmap-label {{
        font-size: 0.6rem; font-weight: 600; letter-spacing: 1.5px;
        text-transform: uppercase; color: {C_SEC}; text-align: center;
        background: {C_WHITE}; border: 1px solid {C_BORDER};
        border-radius: 3px; padding: 8px 6px; width: 100%;
    }}

    /* ── Stat cards ── */
    .stat-card {{
        background: {C_WHITE}; border: 1px solid {C_BORDER};
        border-top: 2px solid {C_ACCENT}; border-radius: 4px;
        padding: 22px 16px; text-align: center;
    }}
    .stat-eyebrow {{
        font-size: 0.58rem; font-weight: 600; letter-spacing: 2px;
        text-transform: uppercase; color: {C_ACCENT}; margin-bottom: 10px;
    }}
    .stat-value {{
        font-size: 2.6rem; font-weight: 700; color: {C_TEXT};
        line-height: 1; margin-bottom: 4px; letter-spacing: -1px;
    }}
    .stat-label  {{ font-size: 0.75rem; font-weight: 400; font-style: italic; color: {C_SEC}; }}
    .stat-top5   {{ font-size: 0.65rem; font-weight: 500; color: {C_LIGHT}; margin-top: 8px; }}

    /* ── Performance table ── */
    .perf-table {{
        width: 100%; border-collapse: collapse;
        font-family: 'Montserrat', sans-serif; font-size: 0.82rem; margin-top: 4px;
    }}
    .perf-table th {{
        font-size: 0.6rem; font-weight: 600; letter-spacing: 1.5px;
        text-transform: uppercase; color: {C_ACCENT}; padding: 8px 12px;
        text-align: left; border-bottom: 2px solid {C_BORDER}; background: {C_BG};
    }}
    .perf-table td {{
        padding: 8px 12px; color: {C_TEXT}; font-weight: 500;
        border-bottom: 1px solid {C_BORDER}; background: {C_WHITE};
    }}
    .perf-table tr:last-child td {{ border-bottom: none; }}
    .perf-table td:first-child {{ font-weight: 600; color: {C_TEXT}; }}

    /* ── Images ── */
    [data-testid="stImage"] img {{
        border-radius: 4px; border: 1px solid {C_BORDER};
    }}

    /* ── Disclaimer / Footer ── */
    .disclaimer {{
        font-size: 0.74rem; font-weight: 300; font-style: italic;
        color: {C_LIGHT}; line-height: 1.65; margin-top: 14px;
        padding-top: 14px; border-top: 1px solid {C_BORDER};
    }}
    .app-footer {{
        margin-top: 60px; padding-top: 16px; border-top: 1px solid {C_BORDER};
        font-size: 0.6rem; font-weight: 400; letter-spacing: 1.5px;
        text-transform: uppercase; color: {C_LIGHT}; text-align: center;
    }}

    /* ── Cache badge ── */
    .cache-badge {{
        display: inline-block; font-size: 0.55rem; font-weight: 600;
        letter-spacing: 1px; text-transform: uppercase; color: {C_CORRECT};
        background: transparent; border: 1px solid {C_CORRECT};
        border-radius: 3px; padding: 2px 7px;
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
    top1     = df.groupby("model_name")["semantic_correct"].mean()
    top5     = df.groupby("model_name")["top5_contains_true_class"].mean()
    by_class = df.groupby(["thesis_class", "model_name"])["semantic_correct"].mean().unstack()
    return top1, top5, by_class


@st.cache_data
def load_top5_data() -> pd.DataFrame:
    return pd.read_csv(
        EVAL_CSV,
        usecols=["image_id", "model_name", "top5_predictions_json"],
    )


@st.cache_data
def load_confusion_data() -> dict:
    CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    df = pd.read_csv(
        EVAL_CSV,
        usecols=["thesis_class", "model_name", "semantic_predicted_class"],
    )
    out = {}
    for model in MODELS:
        mdf    = df[df["model_name"] == model]
        matrix = {}
        for true_c in CLASSES:
            sub    = mdf[mdf["thesis_class"] == true_c]
            total  = len(sub)
            counts = {c: int((sub["semantic_predicted_class"] == c).sum()) for c in CLASSES}
            counts["unmapped"] = total - sum(counts.values())
            counts["total"]    = total
            matrix[true_c]     = counts
        out[model] = matrix
    return out


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


def img_to_b64(img: Image.Image, size: tuple = (160, 120), fmt: str = "JPEG", quality: int = 75) -> str:
    img_copy = img.copy()
    img_copy.thumbnail(size, Image.BICUBIC)
    buf = BytesIO()
    if fmt == "PNG":
        img_copy.save(buf, format="PNG")
    else:
        img_copy.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()




def render_top5_bars(image_id: str) -> None:
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sec-eyebrow'>Softmax Output: Top-5 Predictions</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='sec-title'>Confidence Distribution</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='sec-rule'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sec-desc'>Softmax probabilities for the five highest-scoring ImageNet "
        "classes. The full distribution runs over all 1,000 classes and sums to 1.0. "
        "These bars show the raw model output with no rescaling. "
        "A dominant first bar indicates high confidence in a single class; "
        "similar heights across the five indicate the model is distributing "
        "probability across competing classes.</div>",
        unsafe_allow_html=True,
    )

    top5_df    = load_top5_data()
    image_rows = top5_df[top5_df["image_id"] == image_id]

    bar_cols = st.columns(3)
    for i, model in enumerate(MODELS):
        model_row = image_rows[image_rows["model_name"] == model]
        with bar_cols[i]:
            st.markdown(
                f"<div class='model-card-name' style='text-align:center;margin-bottom:14px;'>"
                f"{MODEL_DISPLAY[model]}</div>",
                unsafe_allow_html=True,
            )
            if model_row.empty:
                st.markdown(
                    f"<p style='font-size:0.75rem;color:{C_WRONG};text-align:center;'>No data</p>",
                    unsafe_allow_html=True,
                )
                continue
            try:
                top5 = json.loads(model_row.iloc[0]["top5_predictions_json"])
            except Exception:
                st.markdown(
                    f"<p style='font-size:0.75rem;color:{C_WRONG};text-align:center;'>Parse error</p>",
                    unsafe_allow_html=True,
                )
                continue

            bars_html = ""
            for j, pred in enumerate(top5):
                label  = pred["label"]
                conf   = pred["confidence"] * 100
                bar_w  = min(round(conf, 2), 100.0)
                color  = C_TEXT if j == 0 else C_ACCENT
                bars_html += (
                    f"<div style='margin-bottom:10px;'>"
                    f"<div style='display:flex;align-items:center;gap:6px;'>"
                    f"<div style='width:82px;font-size:0.6rem;color:{C_SEC};text-align:right;"
                    f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex-shrink:0;'"
                    f" title='{label}'>{label}</div>"
                    f"<div style='flex:1;background:{C_BORDER};border-radius:2px;height:7px;'>"
                    f"<div style='width:{bar_w:.2f}%;background:{color};height:7px;"
                    f"border-radius:2px;max-width:100%;'></div>"
                    f"</div>"
                    f"<div style='width:40px;font-size:0.62rem;color:{C_TEXT};font-weight:600;"
                    f"text-align:right;flex-shrink:0;'>{conf:.1f}%</div>"
                    f"</div>"
                    f"</div>"
                )
            st.markdown(bars_html, unsafe_allow_html=True)


def render_confusion_matrix() -> None:
    CLASSES   = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    PRED_COLS = CLASSES + ["unmapped"]

    try:
        cm_data = load_confusion_data()
    except Exception as e:
        st.error(f"Could not load confusion data: {e}")
        return

    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sec-eyebrow'>Prediction Confusion Matrix</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='sec-title'>Where Each Model Gets It Wrong</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='sec-rule'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sec-desc'>Each row is a true scene class; each column is the predicted "
        "thesis class after mapping the model's top-1 ImageNet label through the semantic map. "
        "<em>Unmapped</em>: the top-1 prediction had no entry in the semantic map for any of "
        "the six classes, meaning the model predicted an ImageNet object category with no "
        "scene-level equivalent in this taxonomy. "
        "Each cell shows the count and its row percentage. Row totals are 500 images per class.</div>",
        unsafe_allow_html=True,
    )

    cm_model = st.radio(
        "Model",
        options=MODELS,
        format_func=lambda m: MODEL_DISPLAY[m],
        horizontal=True,
        label_visibility="collapsed",
        key="cm_model_sel",
    )

    matrix = cm_data[cm_model]

    def _cell_style(count: int, total: int, is_diag: bool, is_unmapped: bool) -> str:
        pct = count / total if total > 0 else 0
        if is_unmapped:
            alpha = min(pct * 1.5, 0.55)
            r = int(255 * (1 - alpha) + 176 * alpha)
            g = int(255 * (1 - alpha) + 168 * alpha)
            b = int(255 * (1 - alpha) + 181 * alpha)
        elif is_diag:
            alpha = min(pct * 1.2, 0.75)
            r = int(255 * (1 - alpha) + 61 * alpha)
            g = int(255 * (1 - alpha) + 107 * alpha)
            b = int(255 * (1 - alpha) + 71 * alpha)
        else:
            alpha = min(pct * 3.5, 0.65)
            r = int(255 * (1 - alpha) + 139 * alpha)
            g = int(255 * (1 - alpha) + 58 * alpha)
            b = int(255 * (1 - alpha) + 46 * alpha)
        text_col = "#FFFFFF" if alpha > 0.45 else C_TEXT
        return f"background:rgb({r},{g},{b});color:{text_col};"

    col_labels   = [c.capitalize() for c in CLASSES] + ["Unmapped"]
    true_th      = (
        f"<th style='font-size:0.52rem;font-weight:600;letter-spacing:1px;"
        f"text-transform:uppercase;color:{C_ACCENT};padding:8px 12px;text-align:right;"
        f"border-bottom:2px solid {C_BORDER};border-right:2px solid {C_BORDER};"
        f"background:{C_BG};white-space:nowrap;'>True \\ Pred</th>"
    )
    header_cells = "".join(
        f"<th style='font-size:0.52rem;font-weight:600;letter-spacing:1px;"
        f"text-transform:uppercase;color:{C_ACCENT};padding:8px 6px;text-align:center;"
        f"border-bottom:2px solid {C_BORDER};background:{C_BG};'>{lbl}</th>"
        for lbl in col_labels
    )

    rows_html = ""
    for true_c in CLASSES:
        row_data   = matrix[true_c]
        total      = row_data["total"]
        row_label  = (
            f"<td style='font-size:0.62rem;font-weight:600;text-align:right;color:{C_TEXT};"
            f"padding:7px 12px 7px 8px;border:1px solid {C_BORDER};background:{C_BG};"
            f"border-right:2px solid {C_BORDER};white-space:nowrap;'>"
            f"{true_c.capitalize()}</td>"
        )
        data_cells = ""
        for pred_c in PRED_COLS:
            count       = row_data.get(pred_c, 0)
            is_diag     = pred_c == true_c
            is_unmapped = pred_c == "unmapped"
            style       = _cell_style(count, total, is_diag, is_unmapped)
            pct_str     = f"{count / total * 100:.0f}%" if total > 0 else "—"
            data_cells += (
                f"<td style='{style}padding:6px 8px;border:1px solid {C_BORDER};"
                f"text-align:center;'>"
                f"<div style='font-size:0.8rem;font-weight:700;line-height:1;'>{count}</div>"
                f"<div style='font-size:0.55rem;font-weight:400;margin-top:2px;"
                f"opacity:0.85;'>{pct_str}</div>"
                f"</td>"
            )
        rows_html += f"<tr>{row_label}{data_cells}</tr>"

    table_html = (
        f"<div style='overflow-x:auto;margin-top:12px;'>"
        f"<table style='width:100%;border-collapse:collapse;"
        f"font-family:Montserrat,sans-serif;'>"
        f"<thead><tr>{true_th}{header_cells}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)

    unmapped_pcts = {
        m: sum(cm_data[m][c]["unmapped"] for c in CLASSES)
           / sum(cm_data[m][c]["total"] for c in CLASSES) * 100
        for m in MODELS
    }
    unmapped_str = " · ".join(
        f"{MODEL_DISPLAY[m]} {unmapped_pcts[m]:.1f}%" for m in MODELS
    )
    st.markdown(
        f"<div class='disclaimer'>"
        f"500 images per class per model, 3,000 per model total. "
        f"Unmapped rates: {unmapped_str}."
        f"</div>",
        unsafe_allow_html=True,
    )


@st.experimental_dialog(" ", width="large")
def _show_enlarged(img: Image.Image) -> None:
    st.image(img, use_column_width=True)


def render_sidebar(manifest: pd.DataFrame):
    with st.sidebar:
        st.markdown("""
        <div style='padding: 20px 4px 14px 4px;'>
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

        avail_cats = manifest[manifest["thesis_class"] == sel_class]["category_name"].unique()
        cat_map = {CATEGORY_DISPLAY.get(c, c): c for c in sorted(avail_cats)}
        sel_cat_display = st.selectbox("Analysis Category", list(cat_map.keys()))
        sel_cat = cat_map[sel_cat_display]

        subset = manifest[
            (manifest["thesis_class"] == sel_class) &
            (manifest["category_name"] == sel_cat)
        ].reset_index(drop=True)

        st.divider()

        st.markdown(
            f"<div style='font-size:0.6rem;font-weight:600;letter-spacing:2px;"
            f"text-transform:uppercase;color:{C_ACCENT};margin-bottom:10px;'>Select Image</div>",
            unsafe_allow_html=True,
        )

        n = len(subset)
        if n == 0:
            st.warning("No images available for this selection.")
            return pd.DataFrame(), 0

        radio_key  = f"radio_{sel_class}_{sel_cat}"
        current_idx = st.session_state.get(radio_key, 0)
        if current_idx >= n:
            current_idx = 0


        thumb_cols = st.columns(n)
        for i in range(n):
            stem_i = Path(str(subset.iloc[i]["image_id"])).stem
            thumb  = safe_img(VIS_DIR / stem_i / "original.jpg")
            is_sel = current_idx == i
            with thumb_cols[i]:
                if thumb:
                    b64     = img_to_b64(thumb)
                    border  = f"2px solid {C_ACCENT}" if is_sel else "2px solid transparent"
                    opacity = "1.0" if is_sel else "0.45"
                    st.markdown(
                        f"<img src='data:image/jpeg;base64,{b64}' "
                        f"style='width:100%;border-radius:3px;border:{border};"
                        f"opacity:{opacity};display:block;'>",
                        unsafe_allow_html=True,
                    )


        sel_idx = current_idx
        dot_cols = st.columns(n)
        for i in range(n):
            with dot_cols[i]:
                dot = "●" if i == current_idx else "○"
                if st.button(dot, key=f"tb_{sel_class}_{sel_cat}_{i}", use_container_width=True):
                    st.session_state[radio_key] = i
                    sel_idx = i
                    st.rerun()

        st.divider()

        st.markdown(
            f"<div style='font-size:0.72rem;font-weight:300;color:{C_SEC};"
            f"line-height:1.6;padding:2px 0 10px 0;'>"
            f"Grad-CAM and feature maps for VGG16, ResNet-18 and MobileNetV2 "
            f"on Places365 scene images.</div>"
            f"<div><span class='cache-badge'>● Precomputed</span></div>",
            unsafe_allow_html=True,
        )

    return subset, sel_idx


def render_main(row: pd.Series, sel_cat: str):
    stem  = row["image_stem"]
    vis   = VIS_DIR / stem
    scene = str(row["scene_label"]).replace("-", " ").replace("_", " ").title()
    cat_d = CATEGORY_DISPLAY.get(row["category_name"], row["category_name"])
    avg_c = float(safe_val(row, "avg_confidence", 0))


    st.markdown(
        "<div class='pg-title'>Visual Analysis of CNN Decision Processes</div>"
        "<hr class='pg-rule'>",
        unsafe_allow_html=True,
    )


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


    with st.spinner("Loading…"):
        orig = safe_img(vis / "original.jpg")
    _, col_img, _ = st.columns([2, 3, 2])
    with col_img:
        if orig:
            st.image(orig, caption=scene, use_column_width=True)
            if st.button("↗ full size", key=f"exp_orig_{stem}", use_container_width=True):
                _show_enlarged(orig)
        else:
            st.warning("Original image not found.")


    desc = CATEGORY_DESCRIPTIONS.get(sel_cat, "")
    if desc:
        st.markdown(f"<div class='cat-context'>{desc}</div>", unsafe_allow_html=True)


    tab_gc, tab_fm, tab_det = st.tabs(["  Grad-CAM  ", "  Feature Maps  ", "  Details  "])


    with tab_gc:
        st.markdown("<div class='sec-eyebrow'>Gradient-Weighted Class Activation Mapping</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>Spatial Decision Analysis</div>",
                    unsafe_allow_html=True)
        st.markdown("<hr class='sec-rule'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sec-desc'>Grad-CAM computes a weighted sum of the final convolutional "
            "feature maps, where each channel is weighted by the gradient of the predicted class "
            "score with respect to that channel. The result is a coarse spatial map showing which "
            "regions contributed most to the decision. Red marks the highest positive influence, "
            "blue the lowest.</div>",
            unsafe_allow_html=True,
        )

        view_mode = st.radio(
            "View",
            options=["Standard Overlay", "Heatmap Only", "Normalized"],
            horizontal=True,
            label_visibility="collapsed",
            key=f"gc_view_{stem}",
        )

        view_desc = {
            "Standard Overlay": "Each model is normalised to its own activation range.",
            "Heatmap Only": "Jet colormap only, no original image blended.",
            "Normalized": "All three models use the same colour scale, set by the highest activation across the three for this image.",
        }
        st.markdown(
            f"<div style='font-size:0.72rem;font-weight:400;font-style:italic;color:{C_SEC};"
            f"line-height:1.6;margin-bottom:18px;'>{view_desc[view_mode]}</div>",
            unsafe_allow_html=True,
        )

        file_map = {
            "Standard Overlay": "{model}_gradcam.png",
            "Heatmap Only":     "{model}_gradcam_heatmap.png",
            "Normalized":       "{model}_gradcam_normalized.png",
        }

        gcols = st.columns(3)
        for i, model in enumerate(MODELS):
            pred    = safe_val(row, f"pred_{model}")
            _raw_correct = safe_val(row, f"correct_{model}", False)
            correct = str(_raw_correct).strip().lower() == "true"
            conf    = float(safe_val(row, f"conf_{model}", 0))
            raw     = safe_val(row, f"top1_label_{model}")
            pred_d  = str(pred).capitalize() if pred != "—" else "—"
            badge   = "badge-correct" if correct else "badge-wrong"
            label   = "Correct" if correct else "Incorrect"

            with gcols[i]:
                st.markdown(f"""
                <div class='model-card'>
                    <div class='model-card-name'>{MODEL_DISPLAY[model]}</div>
                    <div style='text-align:center;margin-bottom:6px;'>
                        <span class='{badge}'>{label} · {pred_d}</span>
                    </div>
                    <div class='pred-conf'>{conf * 100:.1f}% confidence</div>
                    <div class='pred-raw'>ImageNet: {raw}</div>
                </div>
                """, unsafe_allow_html=True)
                fname = file_map[view_mode].replace("{model}", model)
                with st.spinner(""):
                    img = safe_img(vis / fname)
                if img:
                    st.image(img, use_column_width=True)
                    if st.button("↗", key=f"exp_gc_{stem}_{model}_{view_mode}", use_container_width=True):
                        _show_enlarged(img)
                else:
                    st.markdown(
                        f"<p style='font-size:0.75rem;color:{C_WRONG};text-align:center;'>Not found</p>",
                        unsafe_allow_html=True,
                    )

        render_top5_bars(row["image_id"])

    # ── TAB 2: Feature Maps ─────────────────────────────────────────────────
    with tab_fm:
        st.markdown("<div class='sec-eyebrow'>Internal Feature Representations</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>Feature Maps by Layer Depth</div>",
                    unsafe_allow_html=True)
        st.markdown("<hr class='sec-rule'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sec-desc'>Activations captured via forward hooks at three depth stages. "
            "Each grid shows 16 channels, each normalised independently to its own range. "
            "Early layers respond to low-level edges and texture frequencies. "
            "Middle layers encode spatial patterns and structural shapes. "
            "Late layers represent high-level semantic concepts. "
            "Dark channels produced no activation for this input. "
            "The Late layer is the same target layer used for Grad-CAM. "
            "Grad-CAM reduces those channels into a single spatial map by weighting each "
            "channel by its gradient contribution to the predicted class score.</div>",
            unsafe_allow_html=True,
        )

        stage_choice = st.radio(
            "Layer depth",
            options=["Early", "Middle", "Late"],
            horizontal=True,
            label_visibility="collapsed",
            key=f"stage_{stem}",
        )
        stage_key = stage_choice.lower()

        st.markdown(
            f"<div style='font-size:0.62rem;font-weight:600;letter-spacing:2px;"
            f"text-transform:uppercase;color:{C_ACCENT};margin:12px 0 14px 0;'>"
            f"{LAYER_LABELS[stage_key]} · {LAYER_DESC[stage_key]}</div>",
            unsafe_allow_html=True,
        )

        _CELL   = 96
        _PAD    = 3
        _N_CH   = 16
        _N_COLS = 4

        fm_cols = st.columns(3)
        for i, model in enumerate(MODELS):
            with fm_cols[i]:
                st.markdown(
                    f"<div class='model-card-name' style='text-align:center;margin-bottom:8px;'>"
                    f"{MODEL_DISPLAY[model]}</div>",
                    unsafe_allow_html=True,
                )
                with st.spinner(""):
                    grid_img = safe_img(vis / f"{model}_features_{stage_key}.png")
                if grid_img is None:
                    st.markdown(
                        f"<p style='font-size:0.75rem;color:{C_WRONG};text-align:center;'>Missing</p>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.image(grid_img, use_column_width=True)
                    if st.button("↗", key=f"exp_fm_{stem}_{model}_{stage_key}", use_container_width=True):
                        _show_enlarged(grid_img)

    # ── TAB 3: Details ──────────────────────────────────────────────────────
    with tab_det:
        st.markdown("<div class='sec-eyebrow'>Overall Performance</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>Model Accuracy Overview</div>",
                    unsafe_allow_html=True)
        st.markdown("<hr class='sec-rule'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sec-desc'>Top-1 and top-5 semantic accuracy over 3,000 images per model, "
            "computed by mapping each ImageNet-1K prediction to one of the six thesis scene "
            "classes via the semantic map.</div>",
            unsafe_allow_html=True,
        )

        try:
            top1, top5, by_class = load_accuracy_stats()

            sc = st.columns(3)
            for i, model in enumerate(MODELS):
                acc  = float(top1.get(model, 0))
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

            st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:0.6rem;font-weight:600;letter-spacing:2px;"
                f"text-transform:uppercase;color:{C_ACCENT};margin-bottom:12px;'>"
                f"Accuracy by Scene Class</div>",
                unsafe_allow_html=True,
            )

            disp = by_class.copy()
            disp.columns = [MODEL_DISPLAY.get(c, c) for c in disp.columns]
            disp.index   = [c.capitalize() for c in disp.index]
            disp = (disp * 100).round(1)

            col_headers = "".join(f"<th>{c}</th>" for c in disp.columns)
            rows_html   = ""
            for idx, r in disp.iterrows():
                cells = "".join(f"<td>{v:.1f}%</td>" for v in r)
                rows_html += f"<tr><td>{idx}</td>{cells}</tr>"

            st.markdown(
                f"<table class='perf-table'><thead><tr>"
                f"<th>Class</th>{col_headers}"
                f"</tr></thead><tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True,
            )

            st.markdown("""
            <div class='disclaimer'>
            Forest and glacier accuracy is low because ImageNet contains no scene-level class for
            either category. The semantic map approximates coverage through related object classes
            (skis and snowmobiles for glacier, park benches and stone walls for forest), but the
            cross-domain vocabulary gap remains a structural constraint of this setup.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not load accuracy stats: {e}")

        render_confusion_matrix()

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown(
        f"<div class='app-footer'>"
        f"Aida Hamzic &nbsp;·&nbsp; Department of Computer Science "
        f"&nbsp;·&nbsp; Lucerne University of Applied Sciences &nbsp;·&nbsp; 2026"
        f"</div>",
        unsafe_allow_html=True,
    )


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

    sel_cat = subset.iloc[sel_idx]["category_name"]
    render_main(subset.iloc[sel_idx], sel_cat)


if __name__ == "__main__":
    main()
