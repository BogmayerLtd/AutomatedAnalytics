#update this late
import os
import io
import uuid
import base64
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_from_directory, make_response
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "static/reports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

DESIRED_PRODUCTS = [
    "Island Coastal Lager 2/12 pack 12oz Cans", "Island Coastal Lager 4/6 pack 12oz Cans",
    "Island Coastal Lager 1/6 Barrel Keg", "Island Coastal Lager 1/2 Barrel Keg",
    "Island Active 2/12 pack 12oz Cans", "Island Active 1/6 Keg",
    "Island Chill Lemon Lime 6/4 pack 12oz Cans", "Island Chill Strawberry Mango 6/4 pack 12oz Cans",
    "Island Chill Pineapple Coconut 6/4 pack 12oz Cans"
]
CHARLESTON = "Southern Crown Partners: Charleston, SC"

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def plot_inventory_bar(df):
    """Ultra-fast inventory plot - skip profiling entirely"""
    try:
        df = df.iloc[2:].copy()
        if df.shape[1] < 3:
            return ""
        
        # Charleston check first (fast)
        if "Distributor Location" in df.columns and (df["Distributor Location"] == CHARLESTON).any():
            sub = df[df["Distributor Location"] == CHARLESTON]
            if "Product Name" in sub.columns and "On Floor Inventory (Cases)" in sub.columns:
                y = pd.to_numeric(sub["On Floor Inventory (Cases)"], errors="coerce").fillna(0)
                sub = sub[y <= 1000].set_index("Product Name").reindex(DESIRED_PRODUCTS).dropna()
                if sub.empty:
                    return ""
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(sub.index, sub["On Floor Inventory (Cases)"], color="steelblue")
                ax.set_title("Charleston Inventory")
                plt.xticks(rotation=90)
                plt.tight_layout()
                return fig_to_base64(fig)
        
        # Fast fallback: columns 1 vs 2
        x = df.iloc[:, 1].astype(str)
        y = pd.to_numeric(df.iloc[:, 2], errors="coerce").fillna(0)
        mask = y <= 1000
        if not mask.any():
            return ""
        
        tmp = pd.DataFrame({"x": x[mask], "y": y[mask]})
        tmp = tmp.set_index("x").reindex(DESIRED_PRODUCTS).dropna()
        if tmp.empty:
            return ""
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(tmp.index, tmp["y"], color="steelblue")
        ax.set_title("Inventory Plot")
        plt.xticks(rotation=90)
        plt.tight_layout()
        return fig_to_base64(fig)
    except:
        return ""

def plot_storecount_lines(df):
    """Ultra-fast store count - no complex logic"""
    try:
        df = df.iloc[2:].copy()
        if df.shape[1] < 3:
            return ""
        
        # Fast pie chart fallback (most common case)
        labels = df.iloc[:, 1].astype(str)
        sizes = pd.to_numeric(df.iloc[:, 2], errors="coerce").fillna(0)
        mask = sizes > 0
        if not mask.any():
            return ""
        
        labels, sizes = labels[mask], sizes[mask]
        if sizes.sum() == 0:
            return ""
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(sizes, labels=labels[:10], autopct='%1.1f%%')  # Limit to 10 slices
        ax.set_title("Data Distribution")
        plt.tight_layout()
        return fig_to_base64(fig)
    except:
        return ""

def generate_local_insights(df):
    """5-second insights only"""
    rows, cols = df.shape
    missing = df.isnull().sum().sum()
    numeric_cols = df.select_dtypes(include=["number"]).columns[:2]
    
    stats = f"Rows: {rows}, Cols: {cols}, Missing: {missing}"
    for col in numeric_cols:
        try:
            stats += f" | {col}: {df[col].mean():.1f}±{df[col].std():.1f}"
        except:
            pass
    return stats

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400
    
    unique_id = uuid.uuid4().hex[:8]  # Shorter ID
    original_filename = secure_filename(file.filename)
    unique_filename = f"{unique_id}_{original_filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)
    
    try:
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath, nrows=1000)  # LIMIT ROWS
        else:
            df = pd.read_excel(filepath, nrows=1000)  # LIMIT ROWS
    except:
        return jsonify({"error": "Read failed"}), 500
    
    # Generate plots (fast)
    visualizations = {}
    inv_img = plot_inventory_bar(df)
    if inv_img:
        visualizations["inventory"] = inv_img
    
    store_img = plot_storecount_lines(df)
    if store_img:
        visualizations["storecount"] = store_img
    
    # Simple HTML report instead of ydata_profiling
    report_html = f"""
    <html><body style='font-family:Arial; margin:40px;'>
        <h1>Bogmayer Analytics - {unique_id}</h1>
        <h2>Quick Insights</h2>
        <p>{generate_local_insights(df)}</p>
        <h2>Preview</h2>
        {df.head(8).to_html()}
        <h2>Inventory</h2>{'<img src="data:image/png;base64,{inv_img}" style="max-width:100%"/>' if inv_img else ''}
        <h2>Distribution</h2>{'<img src="data:image/png;base64,{store_img}" style="max-width:100%"/>' if store_img else ''}
    </body></html>
    """
    
    report_filename = f"{unique_id}_report.html"
    with open(os.path.join(REPORT_FOLDER, report_filename), "w") as f:
        f.write(report_html)
    
    return jsonify({
        "message": "Success",
        "report_url": f"/reports/{report_filename}",
        "visualizations": visualizations
    })

@app.route("/reports/<reportfile>")
def serve_report(reportfile):
    return send_from_directory(REPORT_FOLDER, reportfile)

if __name__ == "__main__":
    app.run(debug=True)
