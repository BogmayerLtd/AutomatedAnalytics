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
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

#figure #1
def plot_inventory_projection_lines(df):
    fig, ax = plt.subplots(figsize=(12,6))
    time_cols = ["StoreCount_30days", "StoreCount_60days", "StoreCount_90days"]
    # Check columns exist in df
    existing_cols = [col for col in time_cols if col in df.columns]
    if not existing_cols:
        # FALLBACK: Create a histogram of the first numeric column
        return plot_numeric_distribution_fallback(df)
    
    # Determine which identifier column to use
    id_col = None
    if "Location / Product Name" in df.columns:
        id_col = "Location / Product Name"
    elif "Location" in df.columns and "Product Name" in df.columns:
        id_col = ["Location", "Product Name"]
    
    if id_col is None:
        return plot_numeric_distribution_fallback(df)
    
    # Create subset based on column structure
    if isinstance(id_col, list):
        subset = df[id_col + existing_cols].dropna(subset=existing_cols, how="all")
    else:
        subset = df[[id_col] + existing_cols].dropna(subset=existing_cols, how="all")
    
    if subset.empty:
        return plot_numeric_distribution_fallback(df)
    else:
        for _, row in subset.iterrows():
            if isinstance(id_col, list):
                label = f"{row['Product Name']} @ {row['Location']}"
            else:
                label = row[id_col]
            
            ax.plot([int(col.split('_')[1].replace('days','')) for col in existing_cols], 
                    [row[col] for col in existing_cols],
                    label=label, alpha=0.7)
        ax.set_title("Inventory Projection Over Time")
        ax.set_xlabel("Days")
        ax.set_ylabel("Projected Inventory")
        if subset.shape[0] <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
    return fig_to_base64(fig)


#figure #2
def plot_inventory_bar(df):
    fig, ax = plt.subplots(figsize=(10,6))
    
    # Check for different column name variations
    inventory_col = None
    for col_name in ["On Floor Inventory (Cases)", "On Floor Inventory (Case Equivs)"]:
        if col_name in df.columns:
            inventory_col = col_name
            break
    
    # Determine identifier column structure
    has_combined = "Location / Product Name" in df.columns
    has_separate = "Location" in df.columns and "Product Name" in df.columns
    
    if not inventory_col or (not has_combined and not has_separate):
        return plot_category_comparison_fallback(df)
    
    try:
        subset = df.copy()
        
        # Ensure inventory column is numeric
        subset[inventory_col] = pd.to_numeric(subset[inventory_col], errors='coerce')
        subset = subset.dropna(subset=[inventory_col])
        
        if subset.empty:
            return plot_category_comparison_fallback(df)
        
        if has_combined:
            # Try different split patterns
            if ": " in subset["Location / Product Name"].iloc[0]:
                split_data = subset["Location / Product Name"].str.split(": ", n=1, expand=True)
            else:
                # If no colon separator, use the whole string as product name
                split_data = pd.DataFrame({0: ["Unknown Location"] * len(subset), 
                                         1: subset["Location / Product Name"]})
            
            subset['Location'] = split_data[0]
            subset['Product'] = split_data[1]
            
            # Filter out rows where product is None or empty
            subset = subset[subset['Product'].notna() & (subset['Product'] != '')]
            
            if subset.empty:
                return plot_category_comparison_fallback(df)
            
            # Group by product and location
            pivot = subset.pivot_table(index="Product", columns="Location",
                                       values=inventory_col, aggfunc="sum", fill_value=0)
            
        elif has_separate:
            pivot = subset.pivot_table(index="Product Name", columns="Location",
                                       values=inventory_col, aggfunc="sum", fill_value=0)
        
        # Check if pivot has data
        if pivot.empty or pivot.shape[0] == 0:
            return plot_category_comparison_fallback(df)
        
        # Limit to top 15 products by total inventory
        top_products = pivot.sum(axis=1).sort_values(ascending=False).head(15)
        pivot = pivot.loc[top_products.index]
        
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Inventory by Product and Location")
        ax.set_ylabel(inventory_col)
        ax.set_xlabel("Product")
        ax.legend(title="Location", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        fig.tight_layout()
        
    except Exception as e:
        print(f"Error in plot_inventory_bar: {e}")
        return plot_category_comparison_fallback(df)
    
    return fig_to_base64(fig)


# FALLBACK PLOT #1: Numeric Distribution
def plot_numeric_distribution_fallback(df):
    fig, ax = plt.subplots(figsize=(12,6))
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    if not numeric_cols:
        ax.text(0.5, 0.5, "No numeric data available to visualize.", 
                va="center", ha="center", fontsize=12)
        ax.set_title("Data Distribution (No Numeric Data)")
        return fig_to_base64(fig)
    
    # Plot distribution of the first numeric column
    col = numeric_cols[0]
    data = df[col].dropna()
    
    if len(data) == 0:
        ax.text(0.5, 0.5, f"No valid data in column '{col}'", 
                va="center", ha="center", fontsize=12)
    else:
        ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    return fig_to_base64(fig)


# FALLBACK PLOT #2: Category Comparison
def plot_category_comparison_fallback(df):
    fig, ax = plt.subplots(figsize=(10,6))
    
    # Find categorical and numeric columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    if not categorical_cols or not numeric_cols:
        ax.text(0.5, 0.5, "Insufficient data for category comparison.\nNeeds at least one categorical and one numeric column.", 
                va="center", ha="center", fontsize=12)
        ax.set_title("Category Comparison (Insufficient Data)")
        return fig_to_base64(fig)
    
    # Use first categorical and first numeric column
    cat_col = categorical_cols[0]
    num_col = numeric_cols[0]
    
    # Get top 10 categories by sum of numeric values
    grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(10)
    
    if len(grouped) == 0:
        ax.text(0.5, 0.5, "No data to display", va="center", ha="center", fontsize=12)
    else:
        grouped.plot(kind="bar", ax=ax, color='coral', edgecolor='black')
        ax.set_title(f"Top 10 {cat_col} by {num_col}")
        ax.set_xlabel(cat_col)
        ax.set_ylabel(f"Total {num_col}")
        ax.tick_params(axis='x', rotation=45)
    
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
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i,j] > 0.8:
                        high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i,j]))
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
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        visualizations["inventory_bar"] = plot_inventory_bar(df)
        visualizations["inventory_projection"] = plot_inventory_projection_lines(df)
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
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    hist_img = ""
    corr_img = ""
    if numeric_cols:
        bar_b64 = plot_inventory_bar(df)
        bar_img = f'<img src="data:image/png;base64,{bar_b64}" class="plot-img" alt="Inventory Bar Chart"/>'
        proj_b64 = plot_inventory_projection_lines(df)
        proj_img = f'<img src="data:image/png;base64,{proj_b64}" class="plot-img" alt="Inventory Projection Line Chart"/>'

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
      {bar_img}
      {proj_img}
    </div>
    """
    return dashboard_html

def expand_all_variable_sections(html):
    # Remove 'display:none', 'collapse', aria-expanded, or similar hiding for variable panels
    html = re.sub(r'style="display:\s*none;"', 'style="display:block;"', html)
    html = re.sub(r'(class="[^"]*)collapse([^"]*")', r'\1\2', html)  # removes 'collapse' class
    html = re.sub(r'data-bs-toggle="collapse"', '', html)
    html = re.sub(r'aria-expanded="false"', 'aria-expanded="true"', html)
    # Optionally, expand other known hiding mechanisms if your HTML uses others
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
    wkhtml_path = '/usr/local/bin/wkhtmltopdf'  # Change as needed
    config = pdfkit.configuration(wkhtmltopdf=wkhtml_path)
    temp_pdf_path = "/tmp/raw_report.pdf"
    pdfkit.from_string(full_html, temp_pdf_path, configuration=config, options={'enable-local-file-access': None})

    # Remove blank pages and cap at 8 (always keep last page)
    reader = PdfReader(temp_pdf_path)
    writer = PdfWriter()
    kept_count = 0
    max_pages = 8
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if (text and text.strip()) or idx == len(reader.pages)-1:  # keep if not blank or last page
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
