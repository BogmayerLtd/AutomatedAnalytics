#update this late
from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
import pandas as pd
import os
import uuid
import io
import base64
import pdfkit
from PyPDF2 import PdfReader, PdfWriter
import re
import logging

import matplotlib.pyplot as plt
import seaborn as sns

from ydata_profiling import ProfileReport
from werkzeug.utils import secure_filename

import shutil
print(shutil.which("wkhtmltopdf"))

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "static/reports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ---------- VISUALIZATIONS TUNED TO YOUR DATA ----------

def plot_inventory_bar(df):
    """
    Grouped bar chart of On Floor Inventory (Cases vs Case Equivs) by Product.
    - Uses 'Product Name' if present; otherwise first non-numeric column.
    - Drops rows where both inventory measures are 0 or NaN.
    - Limits to top 15 products by Cases to keep the x-axis readable.
    """
    # pick product column
    product_col = None
    for c in df.columns:
        if "Product Name" in c or "Product" in c:
            product_col = c
            break
    if product_col is None:
        non_num = df.select_dtypes(exclude=["number"]).columns
        if len(non_num) == 0:
            return ""
        product_col = non_num[0]

    # inventory columns
    cases_col = None
    equiv_col = None
    for c in df.columns:
        if "On Floor Inventory (Cases" in c:
            cases_col = c
        if "On Floor Inventory (Case Equiv" in c:
            equiv_col = c

    if cases_col is None and equiv_col is None:
        return ""

    cols = [product_col]
    if cases_col is not None:
        cols.append(cases_col)
    if equiv_col is not None:
        cols.append(equiv_col)

    plot_df = df[cols].copy()

    # treat NaN as 0 when deciding to drop
    if cases_col is not None:
        plot_df[cases_col] = plot_df[cases_col].fillna(0)
    if equiv_col is not None:
        plot_df[equiv_col] = plot_df[equiv_col].fillna(0)

    if cases_col is not None and equiv_col is not None:
        mask = (plot_df[cases_col] != 0) | (plot_df[equiv_col] != 0)
    elif cases_col is not None:
        mask = plot_df[cases_col] != 0
    else:
        mask = plot_df[equiv_col] != 0

    plot_df = plot_df[mask]
    if plot_df.empty:
        return ""

    # focus on top 15 products by cases (or equiv if no cases)
    sort_col = cases_col if cases_col is not None else equiv_col
    plot_df = plot_df.sort_values(sort_col, ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 4))

    x = range(len(plot_df))
    width = 0.35

    if cases_col is not None and equiv_col is not None:
        ax.bar([i - width / 2 for i in x], plot_df[cases_col], width=width, label="Cases")
        ax.bar([i + width / 2 for i in x], plot_df[equiv_col], width=width, label="Case Equivs")
    elif cases_col is not None:
        ax.bar(x, plot_df[cases_col], width=width, label="Cases")
    else:
        ax.bar(x, plot_df[equiv_col], width=width, label="Case Equivs")

    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df[product_col], rotation=45, ha="right")
    ax.set_ylabel("On Floor Inventory")
    ax.set_title("On Floor Inventory by Product (Cases vs Case Equivs)")
    ax.legend()

    fig.tight_layout()
    return fig_to_base64(fig)

def plot_storecount_lines(df):
    """
    StoreCount over 30/60/90 days for top products by inventory.
    - Uses the same product_col as inventory plot.
    - If StoreCount_30/60/90 columns missing, falls back to Total StoreCount bar plot.
    """
    # product column
    product_col = None
    for c in df.columns:
        if "Product Name" in c or "Product" in c:
            product_col = c
            break
    if product_col is None:
        non_num = df.select_dtypes(exclude=["number"]).columns
        if len(non_num) == 0:
            return ""
        product_col = non_num[0]

    # inventory col to rank products
    cases_col = None
    for c in df.columns:
        if "On Floor Inventory (Cases" in c:
            cases_col = c
            break
    if cases_col is None:
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) == 0:
            return ""
        cases_col = num_cols[0]

    # storecount columns
    sc30 = None
    sc60 = None
    sc90 = None
    for c in df.columns:
        if "StoreCount_30" in c or "StoreCount_3" in c:
            sc30 = c
        if "StoreCount_60" in c or "StoreCount_6" in c:
            sc60 = c
        if "StoreCount_90" in c or "StoreCount_9" in c:
            sc90 = c

    have_all = bool(sc30 and sc60 and sc90)

    # choose top products by cases
    base_df = df[[product_col, cases_col]].copy()
    base_df[cases_col] = base_df[cases_col].fillna(0)
    base_df = base_df.sort_values(cases_col, ascending=False).head(10)
    top_products = base_df[product_col].unique().tolist()

    fig, ax = plt.subplots(figsize=(9, 4))

    if have_all:
        plot_df = df[df[product_col].isin(top_products)][[product_col, sc30, sc60, sc90]].copy()
        plot_df[sc30] = plot_df[sc30].fillna(0)
        plot_df[sc60] = plot_df[sc60].fillna(0)
        plot_df[sc90] = plot_df[sc90].fillna(0)

        # collapse to one row per product (e.g., sum or max)
        plot_df = plot_df.groupby(product_col, as_index=False).max()

        x = range(len(plot_df))
        ax.plot(x, plot_df[sc30], marker="o", label="30 days")
        ax.plot(x, plot_df[sc60], marker="o", label="60 days")
        ax.plot(x, plot_df[sc90], marker="o", label="90 days")

        ax.set_xticks(list(x))
        ax.set_xticklabels(plot_df[product_col], rotation=45, ha="right")
        ax.set_ylabel("Store Count")
        ax.set_title("Store Count by Product Over Time")
        ax.legend()
    else:
        # fallback: sum any StoreCount* columns and show bar
        store_cols = [c for c in df.columns if "StoreCount" in c]
        if not store_cols:
            plt.close(fig)
            return ""

        plot_df = df[df[product_col].isin(top_products)][[product_col] + store_cols].copy()
        plot_df[store_cols] = plot_df[store_cols].fillna(0)
        plot_df["TotalStoreCount"] = plot_df[store_cols].sum(axis=1)
        plot_df = plot_df.groupby(product_col, as_index=False)["TotalStoreCount"].sum()

        sns.barplot(x=product_col, y="TotalStoreCount", data=plot_df, ax=ax, color="orange")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("Total Store Count")
        ax.set_title("Total Store Count by Product")

    fig.tight_layout()
    return fig_to_base64(fig)

# ---------- INSIGHTS / ROUTES (UNCHANGED EXCEPT FOR USING NEW PLOTS) ----------

def generate_local_insights(df):
    try:
        rows, cols = df.shape
        missing_total = df.isnull().sum().sum()
        num_numeric = len(df.select_dtypes(include=["number"]).columns)
        num_categorical = len(df.select_dtypes(include=["object", "category"]).columns)
        paragraph = (
            f"The dataset contains {rows} rows and {cols} columns. "
            f"There are {num_numeric} numeric columns and {num_categorical} categorical columns. "
            f"It has {missing_total} missing values in total.\n"
        )
        numeric_cols = df.select_dtypes(include=["number"]).columns[:3]
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            paragraph += (
                f"Numeric column '{col}': min = {min_val:.2f}, max = {max_val:.2f}, mean = {mean_val:.2f}. "
            )
        missing_per_column = df.isnull().sum().sort_values(ascending=False)
        most_missing = missing_per_column[missing_per_column > 0]
        if not most_missing.empty:
            paragraph += "\nColumns with most missing data:\n"
            for col, cnt in most_missing.head(2).items():
                paragraph += f"'{col}' has {cnt} missing values. "
        if num_numeric >= 2:
            corr_matrix = df.select_dtypes(include=["number"]).corr().abs()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.8:
                        high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            if high_corr:
                paragraph += "\nHighly correlated numeric columns (>0.8):\n"
                for c1, c2, val in high_corr:
                    paragraph += f"'{c1}' and '{c2}' with correlation of {val:.2f}. "
        return paragraph.strip()
    except Exception as e:
        return f"Failed to generate local insights: {e}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    original_filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex
    unique_filename = f"{unique_id}_{original_filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)
    try:
        if original_filename.endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 500
    local_insights = generate_local_insights(df)
    try:
        profile = ProfileReport(df, title="Automated Profiling Report")
        report_filename = f"{unique_id}_profiling_report.html"
        profile.to_file(os.path.join(REPORT_FOLDER, report_filename))
    except Exception as e:
        return jsonify({"error": f"YData Profiling failed: {e}"}), 500

    visualizations = {}
    inv_img = plot_inventory_bar(df)
    if inv_img:
        visualizations["inventory_bar"] = inv_img
    store_img = plot_storecount_lines(df)
    if store_img:
        visualizations["storecount_trend"] = store_img

    return jsonify({
        "message": "Success",
        "report_url": f"/reports/{report_filename}",
        "upload_id": unique_id,
        "visualizations": visualizations,
        "insights": local_insights
    })

@app.route("/reports/<reportfile>")
def serve_report(reportfile):
    return send_from_directory(REPORT_FOLDER, reportfile)

def build_dashboard_html(upload_id):
    upload_files = os.listdir(UPLOAD_FOLDER)
    matching_files = [f for f in upload_files if f.startswith(upload_id)]
    if not matching_files:
        return "<p>Uploaded file not found for full report.</p>"
    filepath = os.path.join(UPLOAD_FOLDER, matching_files[0])
    try:
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        return f"<p>Error loading dataset: {e}</p>"
    preview_html = df.head(20).to_html(classes="table table-striped", border=0)
    insights = generate_local_insights(df)

    inv_b64 = plot_inventory_bar(df)
    store_b64 = plot_storecount_lines(df)

    inv_img = ""
    store_img = ""
    if inv_b64:
        inv_img = f'<img src="data:image/png;base64,{inv_b64}" class="plot-img" alt="Inventory by Product"/>'
    if store_b64:
        store_img = f'<img src="data:image/png;base64,{store_b64}" class="plot-img" alt="Store Count Trends"/>'

    dashboard_html = f"""
    <div class="section">
      <h2>Data Preview (First 20 Rows)</h2>
      {preview_html}
    </div>
    <div class="section">
      <h2>Summary Stats & Insights</h2>
      <p>{insights}</p>
    </div>
    <div class="section">
      <h2>Visualizations</h2>
      {inv_img}
      {store_img}
    </div>
    """
    return dashboard_html

def expand_all_variable_sections(html):
    html = re.sub(r'style="display:\s*none;"', 'style="display:block;"', html)
    html = re.sub(r'(class="[^"]*)collapse([^"]*")', r'\1\2', html)
    html = re.sub(r'data-bs-toggle="collapse"', '', html)
    html = re.sub(r'aria-expanded="false"', 'aria-expanded="true"', html)
    return html

@app.route("/pdf_report/<upload_id>")
def pdf_report(upload_id):
    report_filename = f"{upload_id}_profiling_report.html"
    report_path = os.path.join(REPORT_FOLDER, report_filename)
    if not os.path.exists(report_path):
        return "Report not found", 404
    with open(report_path, "r", encoding="utf-8") as f:
        backend_html = f.read()
    backend_html = expand_all_variable_sections(backend_html)
    dashboard_html = build_dashboard_html(upload_id)
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8"/>
      <title>Full Data Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; color: #111; }}
        h1 {{ color: royalblue; text-align: center; margin-bottom: 40px; }}
        .section {{ margin-bottom: 40px; }}
        .table {{ width: 100%; border-collapse: collapse; }}
        .table-striped tbody tr:nth-child(odd) {{ background-color: #f2f2f2; }}
        .plot-img {{ max-width: 80%; margin-bottom: 20px; border: 1px solid #ccc; display: block; margin-left: auto; margin-right: auto; }}
      </style>
    </head>
    <body>
      <h1>Bogmayer Analytics Dashboard - Full Report</h1>
      {dashboard_html}
      <div class="section"><h2>Automated Profiling Report</h2>{backend_html}</div>
    </body>
    </html>
    """
    wkhtml_path = '/usr/local/bin/wkhtmltopdf'  # adjust to your actual path
    config = pdfkit.configuration(wkhtmltopdf=wkhtml_path)
    temp_pdf_path = "/tmp/raw_report.pdf"
    pdfkit.from_string(full_html, temp_pdf_path, configuration=config, options={'enable-local-file-access': None})

    reader = PdfReader(temp_pdf_path)
    writer = PdfWriter()
    kept_count = 0
    max_pages = 8
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if (text and text.strip()) or idx == len(reader.pages) - 1:
            writer.add_page(page)
            kept_count += 1
        if kept_count >= max_pages:
            break
    outstream = io.BytesIO()
    writer.write(outstream)
    outstream.seek(0)
    response = make_response(outstream.read())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename={upload_id}_full_report.pdf"
    return response

if __name__ == "__main__":
    app.run(debug=True)
