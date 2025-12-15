#update this late
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, make_response
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
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

#changed inventory
def plot_inventory_bar(df):
    """
    If 'Location' contains 'Southern Crown Partners: Charleston, SC',
    plot only that subset, with Product Name on x-axis,
    On Floor Inventory (Cases) on y-axis, skipping the first two rows.
    Otherwise, just plot the second column vs. the third column, skipping two rows.
    Put a ceiling at 1000 on inventory (drop bars > 1000).
    Enforce a fixed order for product names.
    """
    desired_order = [
        "Island Coastal Lager 2/12 pack 12oz Cans",
        "Island Coastal Lager 4/6 pack 12oz Cans",
        "Island Coastal Lager 1/6 Barrel Keg",
        "Island Coastal Lager 1/2 Barrel Keg",
        "Island Active 2/12 pack 12oz Cans",
        "Island Active 1/6 Keg",
        "Island Chill Lemon Lime 6/4 pack 12oz Cans",
        "Island Chill Strawberry Mango 6/4 pack 12oz Cans",
        "Island Chill Pineapple Coconut 6/4 pack 12oz Cans",
    ]

    # Skip first 2 rows for all processing
    df = df.iloc[2:].copy()
    
    if "Distributor Location" in df.columns and (df["Distributor Location"] == "Southern Crown Partners: Charleston, SC").any():
        sub = df[df["Distributor Location"] == "Southern Crown Partners: Charleston, SC"].copy()
        if "Product Name" in sub.columns and "On Floor Inventory (Cases)" in sub.columns:
            # filter to only rows <= 1000
            sub = sub[sub["On Floor Inventory (Cases)"] <= 1000]
            if sub.empty:
                return ""

            # enforce desired product order; keep only products in the list
            sub = sub.set_index("Product Name").reindex(desired_order).dropna(subset=["On Floor Inventory (Cases)"])
            if sub.empty:
                return ""

            x = sub.index
            y = sub["On Floor Inventory (Cases)"].fillna(0)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x, y, width=0.9, edgecolor="white", linewidth=0.7)
            plt.xticks(rotation=90)
            ax.set_xlabel("Product Name")
            ax.set_ylabel("# of Cases")
            ax.set_title("Charleston Inventory (≤ 1000 cases)")
            fig.tight_layout()
            return fig_to_base64(fig)
    
    # Fallback: plot second and third columns (already skipped first two rows)
    if df.shape[1] < 3:
        return ""
    
    # use column 1 for x and column 2 for y
    x = df.iloc[:, 1].astype(str)
    y = pd.to_numeric(df.iloc[:, 2], errors="coerce").fillna(0)

    # filter to inventory <= 1000
    mask = y <= 1000
    x = x[mask]
    y = y[mask]
    if y.empty:
        return ""

    # enforce desired product order for fallback as well
    # build a small dataframe then reindex on the product name
    tmp = pd.DataFrame({"Product Name": x.values, "Value": y.values})
    tmp = tmp.set_index("Product Name").reindex(desired_order).dropna(subset=["Value"])
    if tmp.empty:
        return ""

    x_ordered = tmp.index
    y_ordered = tmp["Value"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_ordered, y_ordered, width=0.9, edgecolor="white", linewidth=0.7)
    plt.xticks(rotation=90)
    ax.set_xlabel(df.columns[1])  # column index 1
    ax.set_ylabel(df.columns[2])  # column index 2
    ax.set_title("Charleston Inventory Plot")
    fig.tight_layout()
    return fig_to_base64(fig)



def plot_storecount_lines(df):
    """
    If a StoreCount_30days column exists, create an overlaid bar chart
    of StoreCount_30days / _60days / _90days by Product Name (or equivalent).
    Otherwise, fall back to a pie chart of column index 2 values,
    with column index 1 as labels.
    """
    # Check for StoreCount_30days (or close variants)
    store_30_cols = [c for c in df.columns if "StoreCount_30" in c or "StoreCount30" in c or "StoreCount_30days" in c]
    
    if store_30_cols:
        # Overlaid 30/60/90-day bar chart
        # Identify the last 3 columns as 30/60/90 as in your notebook pattern
        last_cols = df.columns[-3:]
        col_30, col_60, col_90 = last_cols
        
        plot_df = df.iloc[2:, :].copy()  # skip first 2 rows, consistent with notebook
        plot_df[last_cols] = plot_df[last_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        plot_df = plot_df[(plot_df[last_cols] != 0).any(axis=1)]
        
        # Use Product Name (or fallback) as x labels
        if "Product Name" in plot_df.columns:
            label_col = "Product Name"
        else:
            # fall back to first non-numeric column
            non_num = plot_df.select_dtypes(exclude=["number"]).columns
            if len(non_num) == 0:
                return ""
            label_col = non_num[0]
        
        # optional: limit to fewer products for readability
        plot_df = plot_df.sort_values(col_90, ascending=False).head(12)
        
        x_labels = plot_df[label_col].astype(str)
        x = range(len(x_labels))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # All start at 0, different colors, no transparency
        ax.bar(x, plot_df[col_90], width=0.7, color="tab:green", label="90 days")
        ax.bar(x, plot_df[col_60], width=0.5, color="tab:orange", label="60 days")
        ax.bar(x, plot_df[col_30], width=0.3, color="tab:blue", label="30 days")
        
        ax.set_xticks(list(x))
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_ylabel("# of Retailers")
        ax.set_xlabel(label_col)
        ax.set_title("Store Count by Product 30/60/90 Days")
        ax.legend()
        fig.tight_layout()
        return fig_to_base64(fig)
    
    # ---------- Fallback: pie chart on columns 1 and 2 ----------
    plot_df = df.iloc[2:, :15].copy()
    if plot_df.shape[1] < 3:
        return ""
    
    labels = plot_df.iloc[:, 1].astype(str)
    sizes = pd.to_numeric(plot_df.iloc[:, 2], errors="coerce").fillna(0)
    
    # remove zero-size slices
    mask = sizes != 0
    labels = labels[mask]
    sizes = sizes[mask]
    
    if sizes.sum() == 0:
        return ""
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Pie Chart: {df.columns[2]} by {df.columns[1]}")
    fig.tight_layout()
    return fig_to_base64(fig)

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
        # Use minimal mode for MUCH faster report generation (30-60s instead of 3+ minutes)
        profile = ProfileReport(
            df, 
            title="Automated Profiling Report",
            minimal=True,
            explorative=False,
            # Disable expensive computations that take the most time
            correlations={
                "auto": {"calculate": False},
                "pearson": {"calculate": False},
                "spearman": {"calculate": False},
                "kendall": {"calculate": False},
                "phi_k": {"calculate": False},
                "cramers": {"calculate": False},
            },
            interactions={"continuous": False},
            missing_diagrams={"bar": False, "matrix": False, "heatmap": False},
        )
        report_filename = f"{unique_id}_profiling_report.html"
        profile.to_file(os.path.join(REPORT_FOLDER, report_filename))
    except Exception as e:
        return jsonify({"error": f"YData Profiling failed: {e}"}), 500
    
    # New visualizations
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
    
    # Show only 12 rows in preview
    preview_html = df.head(12).to_html(classes="table table-striped", border=0)
    insights = generate_local_insights(df)
    
    # Generate Basic Data Analysis table (same as frontend)
    columns = df.columns.tolist()
    analysis_html = f'<p><b>Rows:</b> {len(df)}, <b>Columns:</b> {len(columns)}</p>'
    analysis_html += '''<table class="table table-bordered" style="width:100%; border-collapse: collapse;">
    <thead><tr style="background-color: #f2f2f2;">
        <th style="border: 1px solid #ddd; padding: 8px;">Column</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Type</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Missing</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Example</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Stats</th>
    </tr></thead><tbody>'''
    
    for col in columns:
        values = df[col].dropna()
        missing = len(df) - len(values)
        col_type = "string"
        stats = "-"
        example = str(values.iloc[0]) if len(values) > 0 else ""
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = "number"
            if len(values) > 0:
                min_val = values.min()
                max_val = values.max()
                mean_val = values.mean()
                stats = f"min: {min_val:.2f}, max: {max_val:.2f}, mean: {mean_val:.2f}"
        # Check if date
        elif df[col].dtype == 'object':
            try:
                pd.to_datetime(values.head(10), errors='raise')
                col_type = "date"
            except:
                pass
        
        analysis_html += f'''<tr>
            <td style="border: 1px solid #ddd; padding: 8px;">{col}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{col_type}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{missing}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{example}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{stats}</td>
        </tr>'''
    
    analysis_html += '</tbody></table>'
    
    inv_b64 = plot_inventory_bar(df)
    store_b64 = plot_storecount_lines(df)
    
    inv_img = ""
    store_img = ""
    if inv_b64:
        inv_img = f'<img src="data:image/png;base64,{inv_b64}" style="max-width:100%; height:auto; margin:20px 0;" alt="Inventory by Product"/>'
    if store_b64:
        store_img = f'<img src="data:image/png;base64,{store_b64}" style="max-width:100%; height:auto; margin:20px 0;" alt="Store Count Trends"/>'
    
    dashboard_html = f"""
    <h2>Data Preview (First 12 Rows)</h2>
    {preview_html}
    
    <div style="page-break-after: always;"></div>
    
    <h2>Basic Data Analysis</h2>
    {analysis_html}
    
    <div style="margin-bottom: 120px;"></div>
    
    <h2>Summary Stats & Insights</h2>
    <p>{insights}</p>
    
    <div style="margin-bottom: 120px;"></div>
    
    <h2>Visualizations</h2>
    {inv_img}
    {store_img}

    <div style="margin-bottom: 120px;"></div>
    """
    
    
    return dashboard_html

@app.route("/pdf_report/<upload_id>")
def pdf_report(upload_id):
    report_filename = f"{upload_id}_profiling_report.html"
    report_path = os.path.join(REPORT_FOLDER, report_filename)
    if not os.path.exists(report_path):
        return "Report not found", 404
    
    # Read the ydata HTML file as-is
    with open(report_path, "r", encoding="utf-8") as f:
        ydata_html = f.read()
    
    # Build dashboard
    dashboard_html = build_dashboard_html(upload_id)
    
    # Simply prepend dashboard to ydata HTML
    full_html = f"""
    <div style="font-family: Arial, sans-serif; margin: 20px;">
        <h1 style="color: royalblue; text-align: center;">Bogmayer Analytics Dashboard - Full Report</h1>
        {dashboard_html}
        <hr style="margin: 40px 0;"/>
        <h2>Automated Profiling Report</h2>
    </div>
    {ydata_html}
    """
    
    # Find wkhtmltopdf
    wkhtml_path = shutil.which("wkhtmltopdf") or '/usr/local/bin/wkhtmltopdf' or '/usr/bin/wkhtmltopdf'
    config = pdfkit.configuration(wkhtmltopdf=wkhtml_path)
    
    # Simple options
    options = {
        'enable-local-file-access': None,
        'encoding': 'UTF-8',
    }
    
    # Convert to PDF
    pdf_bytes = pdfkit.from_string(full_html, False, configuration=config, options=options)
    
    response = make_response(pdf_bytes)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename={upload_id}_full_report.pdf"
    
    return response

if __name__ == "__main__":
    app.run(debug=True)
