import datetime
import json
import os
import shutil
import uuid

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, redirect, render_template, request, send_file, url_for
from fpdf import FPDF
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# --- CẤU HÌNH ---
MODEL_PATH = 'cassava_best.keras'
IMAGE_SIZE = (320, 320)
BASE_DIR = 'static'
IMG_DIR = os.path.join(BASE_DIR, 'images')
PDF_DIR = os.path.join(BASE_DIR, 'pdfs')
HEATMAP_DIR = os.path.join(BASE_DIR, 'heatmaps') # Thêm folder heatmap
HISTORY_FILE = 'history.json'
MAX_HISTORY = 10 # Tăng lên 10 cái lưu cho sướng

# Tạo thư mục
for d in [IMG_DIR, PDF_DIR, HEATMAP_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# Tạo file history
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f: json.dump([], f)

# Load Model
try:
    model = load_model(MODEL_PATH)
    print("✅ STATUS: Model loaded successfully.")
except:
    print("❌ ERROR: Failed to load model.")

INFO = {
    0: {"name": "Cassava Bacterial Blight (CBB)", "advice": "Prune infected leaves. Use resistant varieties."},
    1: {"name": "Cassava Brown Streak Disease (CBSD)", "advice": "Use virus-free planting materials. Control whiteflies."},
    2: {"name": "Cassava Green Mite (CGM)", "advice": "Apply biological or chemical mite control. Ensure irrigation."},
    3: {"name": "Cassava Mosaic Disease (CMD)", "advice": "CRITICAL: Uproot and burn plants immediately."},
    4: {"name": "Healthy", "advice": "The plant is healthy. Maintain regular monitoring."}
}

# --- HELPER: GRAD-CAM HEATMAP (PHẦN KHÓ NHẤT) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    # Load ảnh gốc
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Tạo màu cho heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Chồng hình
    superimposed_img = heatmap * alpha + img
    
    # Lưu hình mới
    filename = "heat_" + os.path.basename(img_path)
    save_path = os.path.join(HEATMAP_DIR, filename)
    cv2.imwrite(save_path, superimposed_img)
    
    return f"static/heatmaps/{filename}", save_path

def find_last_conv_layer(model):
    """Tự động tìm layer Convolution cuối cùng"""
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    return None

# --- HELPER FUNCTIONS ---
def load_history():
    try:
        with open(HISTORY_FILE, 'r') as f: return json.load(f)
    except: return []

def save_history(data):
    with open(HISTORY_FILE, 'w') as f: json.dump(data, f, indent=4)

def delete_files_for_entry(entry):
    try:
        if os.path.exists(entry['img_path']): os.remove(entry['img_path'])
        if os.path.exists(entry['pdf_path']): os.remove(entry['pdf_path'])
        # Xóa luôn heatmap nếu có
        heatmap_sys_path = entry.get('heatmap_path_sys')
        if heatmap_sys_path and os.path.exists(heatmap_sys_path):
            os.remove(heatmap_sys_path)
    except Exception as e:
        print(f"Error deleting files: {e}")

def create_pdf(entry_id, disease, advice, confidence, img_full_path, heatmap_full_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="LABORATORY REPORT - CASSAVA AI", ln=1, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"ID: {entry_id}", ln=1)
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Result: {disease}", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Confidence: {confidence}%", ln=1)
    pdf.ln(5)
    
    # Chèn 2 ảnh: Gốc và Heatmap
    try:
        pdf.cell(95, 10, txt="Original Specimen:", ln=0)
        pdf.cell(95, 10, txt="AI Attention Map (Heatmap):", ln=1)
        
        y_pos = pdf.get_y()
        pdf.image(img_full_path, x=10, y=y_pos, w=85)
        if heatmap_full_path:
            pdf.image(heatmap_full_path, x=105, y=y_pos, w=85)
        pdf.ln(90) # Xuống dòng sau ảnh
    except: pass
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Technical Advice:", ln=1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, txt=advice)
    
    filename = f"Report_{entry_id}.pdf"
    pdf_full_path = os.path.join(PDF_DIR, filename)
    pdf.output(pdf_full_path)
    return pdf_full_path, filename

def predict_process(img_path, entry_id):
    # 1. Xử lý ảnh
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    # 2. Dự đoán
    prediction = model.predict(img_array_expanded)
    idx = int(np.argmax(prediction))
    conf = float(np.max(prediction) * 100)
    conf = round(conf, 2)
    
    # 3. Tạo Heatmap
    heatmap_url = None
    heatmap_sys = None
    try:
        last_layer = find_last_conv_layer(model)
        if last_layer:
            heatmap = make_gradcam_heatmap(img_array_expanded, model, last_layer)
            heatmap_url, heatmap_sys = save_and_display_gradcam(img_path, heatmap)
    except Exception as e:
        print(f"Heatmap Error: {e}")
        heatmap_url = f"static/images/{os.path.basename(img_path)}" # Fallback về ảnh gốc
        heatmap_sys = img_path

    return idx, conf, heatmap_url, heatmap_sys

# --- ROUTES ---

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'my_image' not in request.files: return redirect(request.url)
        file = request.files['my_image']
        if file.filename == '': return redirect(request.url)

        entry_id = str(uuid.uuid4())[:8]
        ext = file.filename.split('.')[-1]
        img_filename = f"{entry_id}.{ext}"
        img_full_path = os.path.join(IMG_DIR, img_filename)
        file.save(img_full_path)

        # Xử lý tất cả trong 1 hàm
        idx, conf, heatmap_url, heatmap_sys = predict_process(img_full_path, entry_id)
        
        # Tạo PDF
        pdf_full_path, pdf_filename = create_pdf(entry_id, INFO[idx]["name"], INFO[idx]["advice"], conf, img_full_path, heatmap_sys)

        history = load_history()
        new_entry = {
            "id": entry_id,
            "date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            "disease": INFO[idx]["name"],
            "advice": INFO[idx]["advice"],
            "confidence": conf,
            "is_healthy": (idx == 4),
            "img_path": img_full_path,
            "pdf_path": pdf_full_path,
            "heatmap_path_sys": heatmap_sys,
            "img_url": f"static/images/{img_filename}",
            "pdf_url": f"static/pdfs/{pdf_filename}",
            "heatmap_url": heatmap_url
        }
        
        history.insert(0, new_entry)
        while len(history) > MAX_HISTORY:
            oldest = history.pop()
            delete_files_for_entry(oldest)
        save_history(history)

        return redirect(url_for('result_page', entry_id=entry_id))

    return render_template("index.html")

@app.route("/result/<entry_id>")
def result_page(entry_id):
    history = load_history()
    entry = next((item for item in history if item["id"] == entry_id), None)
    if not entry: return redirect(url_for('index'))
    return render_template("result.html", entry=entry)

@app.route("/archive")
def archive_page():
    history = load_history()
    
    # Tính toán thống kê cho biểu đồ
    stats = {0:0, 1:0, 2:0, 3:0, 4:0}
    disease_labels = []
    
    # Đếm số lượng bệnh trong lịch sử
    disease_map = {v['name']: k for k, v in INFO.items()}
    
    for item in history:
        d_name = item['disease']
        if d_name in disease_map:
            stats[disease_map[d_name]] += 1
            
    chart_data = list(stats.values())
    
    return render_template("archive.html", history=history, chart_data=chart_data)

@app.route("/delete/<entry_id>")
def delete_entry(entry_id):
    history = load_history()
    entry_to_delete = next((item for item in history if item["id"] == entry_id), None)
    if entry_to_delete:
        delete_files_for_entry(entry_to_delete)
        history = [item for item in history if item["id"] != entry_id]
        save_history(history)
    return redirect(url_for('archive_page'))

@app.route("/download_all")
def download_all():
    zip_filename = "All_Reports.zip"
    zip_path = os.path.join(BASE_DIR, "All_Reports")
    shutil.make_archive(zip_path, 'zip', PDF_DIR)
    return send_file(f"{zip_path}.zip", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)