import streamlit as st
import cv2
import numpy as np
import json
from PIL import Image

# --- 核心影像處理函數 ---

def order_points(pts):
    """ 為四個角點排序：左上、右上、右下、左下 """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, pts):
    """ 將傾斜的紙張拉正 """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def find_card_contour(image):
    """ 自動尋找答案卡的外框 """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

# --- Streamlit 介面 ---

st.set_page_config(page_title="AI 讀卡機系統", layout="wide")
st.title("🖨️ 智能畫卡辨識系統")

with st.sidebar:
    st.header("設定說明")
    st.info("1. 上傳空白範本並定義座標\n2. 上傳學生考卷\n3. 系統自動校正並讀取")
    # 這裡預設載入座標，若無則手動定義
    try:
        with open("coords.json", "r") as f:
            coords = json.load(f)
        st.success("✅ 已載入座標配置檔")
    except:
        st.warning("⚠️ 尚未偵測到 coords.json")

# 上傳區
col1, col2 = st.columns(2)
with col1:
    base_file = st.file_uploader("1. 上傳空白範本 (建立基準)", type=['jpg', 'png'])
with col2:
    student_file = st.file_uploader("2. 上傳學生劃記照片", type=['jpg', 'png'])

if student_file:
    # 讀取影像
    file_bytes = np.asarray(bytearray(student_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 步驟 1: 尋找外框並拉正
    card_pts = find_card_contour(img)
    if card_pts is not None:
        warped = perspective_transform(img, card_pts)
        # 統一縮放到範本大小（假設範本為 600x800）
        warped = cv2.resize(warped, (600, 800))
        
        st.image(warped, caption="系統已自動校正並對齊考卷", use_container_width=True)
        
        if st.button("執行辨識"):
            # 步驟 2: 轉灰階與二值化
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            
            # 步驟 3: 根據座標抓取劃記 (範例邏輯)
            results = []
            # 假設 coords["answers"] 儲存了所有選項的 (x, y)
            # 這裡示範前 5 題
            for i in range(0, min(20, len(coords["answers"])), 4):
                scores = []
                for j in range(4): # A, B, C, D
                    x, y = coords["answers"][i+j]
                    roi = thresh[y-10:y+10, x-10:x+10]
                    scores.append(cv2.countNonZero(roi))
                
                ans = chr(65 + np.argmax(scores))
                results.append(ans)
            
            # 顯示結果
            st.subheader("📝 辨識結果")
            st.write(f"**建議答案串：** {' '.join(results)}")
            st.table({"題號": list(range(1, len(results)+1)), "辨識填答": results})
    else:
        st.error("無法偵測到考卷邊緣，請確保背景乾淨且考卷四角完整入鏡。")
