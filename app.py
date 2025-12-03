# app.py
# Digital Lost & Found — 4 UI screens: Home, Lost Form, Found Form, Lost Report (Search Results)
# Run: streamlit run app.py

import os
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

# ---------- CONFIG ----------
BASE = os.path.abspath(".")
UPLOAD = os.path.join(BASE, "uploads")
FOUND = os.path.join(UPLOAD, "found")
LOST = os.path.join(UPLOAD, "lost")
FOUND_CSV = os.path.join(UPLOAD, "found_reports.csv")
for d in [UPLOAD, FOUND, LOST]:
    os.makedirs(d, exist_ok=True)

DOC_TYPES = ["Aadhaar", "Debit Card", "Credit Card", "Voter ID", "Driving Licence", "College ID", "Other"]

st.set_page_config(page_title="Lost & Found Hub", layout="wide", initial_sidebar_state="collapsed")

# ---------- STYLES (blue + white tablet look) ----------
CSS = """
<style>
:root{--accent-start:#0b63ff;--accent-end:#0b9eff;--muted:#536173}
html,body,.stApp{background:linear-gradient(180deg,#f8fbff 0%,#eef7ff 100%);font-family:Inter,system-ui,Roboto,Arial;}
.header{background:linear-gradient(90deg,var(--accent-start),var(--accent-end));color:#fff;padding:18px;border-radius:12px;margin-bottom:16px}
.h-title{font-size:20px;font-weight:800;margin:0}
.h-sub{font-size:13px;opacity:0.95;margin-top:4px}
.card{background:#fff;border-radius:12px;padding:18px;box-shadow:0 8px 24px rgba(2,6,23,0.04)}
.icon-wrap{width:64px;height:64px;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:800;font-size:28px;margin-bottom:8px}
.icon-found{background:linear-gradient(180deg,#06b6d4,#0b9eff)}
.icon-lost{background:linear-gradient(180deg,#3b82f6,#0b63ff)}
.form-card{background:#fff;padding:16px;border-radius:10px;box-shadow:0 6px 20px rgba(2,6,23,0.04)}
.small-note{color:var(--muted);font-size:13px;text-align:center}
.badge-type{background:linear-gradient(90deg,#0b63ff,#06b6d4);color:#fff;padding:6px 10px;border-radius:999px;font-weight:700}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- UTILITIES ----------
def sanitize_folder(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-","") else "" for c in (name or "").strip()).lower() or "other"

def save_file(upload, folder, prefix):
    ts = int(time.time() * 1000)
    name = f"{prefix}{ts}{upload.name}"
    path = os.path.join(folder, name)
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    return path

def save_found_file(upload, doc_type: str):
    safe = sanitize_folder(doc_type or "other")
    folder = os.path.join(FOUND, safe)
    os.makedirs(folder, exist_ok=True)
    ts = int(time.time() * 1000)
    name = f"found_{safe}{ts}{upload.name}"
    path = os.path.join(folder, name)
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    return path

def append_found_row(row):
    if os.path.exists(FOUND_CSV):
        old = pd.read_csv(FOUND_CSV)
        df = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(FOUND_CSV, index=False)

def auto_crop(pil_img):
    try:
        arr = cv2.cvtColor(np.array(ImageOps.exif_transpose(pil_img)), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(7,7),0)
        _, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return pil_img
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cropped = arr[y:y+h, x:x+w]
        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    except Exception:
        return pil_img

# Basic ORB + hist matching (fast, simple)
def orb_desc(pil_img):
    try:
        g = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=1500)
        _, des = orb.detectAndCompute(g, None)
        return des
    except Exception:
        return None

def orb_score(a,b):
    if a is None or b is None: return 0.0
    try:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(a,b,k=2)
        good = 0
        for m_n in matches:
            if len(m_n) != 2: continue
            m,n = m_n
            if m.distance < 0.75 * n.distance:
                good += 1
        # normalize by 200 (empirical)
        score = min(1.0, good / 200.0)
        return score
    except Exception:
        return 0.0

def hist_score(a,b):
    try:
        a_h = cv2.cvtColor(np.array(a.resize((256,256))), cv2.COLOR_RGB2HSV)
        b_h = cv2.cvtColor(np.array(b.resize((256,256))), cv2.COLOR_RGB2HSV)
        H1 = cv2.calcHist([a_h],[0,1],None,[50,50],[0,180,0,256])
        H2 = cv2.calcHist([b_h],[0,1],None,[50,50],[0,180,0,256])
        cv2.normalize(H1,H1); cv2.normalize(H2,H2)
        val = cv2.compareHist(H1,H2,cv2.HISTCMP_CORREL)
        return float((val+1.0)/2.0)
    except Exception:
        return 0.0

def get_all_found_image_paths():
    paths=[]
    for root, dirs, files in os.walk(FOUND):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png")):
                paths.append(os.path.join(root,f))
    return paths

# ---------- NAV ----------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go(p):
    st.session_state.page = p

# ---------- HEADER ----------
st.markdown(f"""
<div class="header">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <div class="h-title">Lost & Found Hub</div>
      <div class="h-sub">Tablet demo — Blue & White UI</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:12px;opacity:0.95">Local demo — No cloud</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- HOME / STARTING INTERFACE ----------
if st.session_state.page == "home":
    left, right = st.columns([1,1], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="icon-wrap icon-lost">L</div>', unsafe_allow_html=True)
        st.markdown('<div style="display:flex;justify-content:space-between;align-items:center"><div><h3 style="margin:0">I LOST Something</h3><div style="color:#536173">Upload your lost item photo and search found items.</div></div><div><span class="badge-type">Search</span></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("I LOST — Start", key="home_lost", use_container_width=True):
            go("lost_form")
            st.stop()
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="icon-wrap icon-found">F</div>', unsafe_allow_html=True)
        st.markdown('<div style="display:flex;justify-content:space-between;align-items:center"><div><h3 style="margin:0">I FOUND Something</h3><div style="color:#536173">Report a found item so owners can claim it.</div></div><div><span class="badge-type">Report</span></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("I FOUND — Report", key="home_found", use_container_width=True):
            go("found_form")
            st.stop()
    st.markdown('<div class="small-note">Tip: Clear close-up photos (single object, plain background) give better results.</div>', unsafe_allow_html=True)
    st.stop()

# ---------- LOST FORM (user reports lost item & can search) ----------
if st.session_state.page == "lost_form":
    if st.button("← Back to Home", key="back1"):
        go("home")
        st.stop()
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.header("Report a Lost Item")
    with st.form("lost_report_form", clear_on_submit=False):
        col1, col2 = st.columns([2,1])
        with col1:
            item_name = st.text_input("Item Name")
            last_seen = st.text_input("Last Seen Location")
            desc = st.text_area("Detailed Description (color, brand, unique features)", max_chars=400)
            photos = st.file_uploader("Upload Photo(s) (1-3)", type=["jpg","jpeg","png"], accept_multiple_files=True)
        with col2:
            category = st.selectbox("Category", ["-- any --","Electronics","Clothing","Accessories","Documents","Other"])
            date_seen = st.date_input("Date Last Seen")
            contact = st.text_input("Contact (optional)")
        search_btn = st.form_submit_button("Search Found Items")
    st.markdown('</div>', unsafe_allow_html=True)

    if search_btn:
        if not photos:
            st.error("Upload at least one photo to search.")
        else:
            # save lost photos temporarily and process
            lost_imgs = []
            for f in photos[:3]:
                p = save_file(f, LOST, "lost")
                try:
                    pil = Image.open(p).convert("RGB")
                    pil = auto_crop(pil)
                    lost_imgs.append(pil)
                except Exception:
                    st.warning("A photo could not be read and was skipped.")
            # build candidates
            candidates = []
            saved_paths = get_all_found_image_paths()
            st.info(f"Matching against {len(saved_paths)} found images.")
            # compute descriptors for lost (first image used as primary)
            lost_desc = orb_desc(lost_imgs[0]) if lost_imgs else None
            scores = []
            for fp in saved_paths:
                try:
                    fimg = Image.open(fp).convert("RGB")
                    proc = auto_crop(fimg)
                except Exception:
                    continue
                des = orb_desc(proc)
                o = orb_score(lost_desc, des)
                h = hist_score(lost_imgs[0], proc)
                final = round(0.6*o + 0.4*h, 4)
                scores.append({"path":fp, "image":proc, "final":final})
            scores = sorted(scores, key=lambda x: x["final"], reverse=True)
            # go to search results page - store in session
            st.session_state.search_results = scores
            go("lost_report")
            st.stop()
    st.stop()

# ---------- FOUND FORM (user reports found item) ----------
if st.session_state.page == "found_form":
    if st.button("← Back to Home", key="back2"):
        go("home")
        st.stop()
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.header("Report: Found Item")
    with st.form("found_form", clear_on_submit=True):
        finder = st.text_input("Your name")
        location = st.text_input("Found at (location)")
        description = st.text_area("Short description (color, brand, marks)", max_chars=400)
        contact = st.text_input("Contact (optional)")
        doc_type = st.selectbox("Item type", ["-- select --"] + DOC_TYPES)
        imgs = st.file_uploader("Photos (1-4)", type=["jpg","jpeg","png"], accept_multiple_files=True)
        submit = st.form_submit_button("Save Found Report")
    st.markdown('</div>', unsafe_allow_html=True)

    if submit:
        if not imgs:
            st.error("Please upload at least one image.")
        else:
            chosen_type = doc_type if doc_type and doc_type != "-- select --" else "Other"
            saved_paths = []
            for f in imgs[:4]:
                saved_paths.append(save_found_file(f, chosen_type))
            row = {
                "timestamp": int(time.time()),
                "finder": finder,
                "contact": contact,
                "location": location,
                "doc_type": chosen_type,
                "description": description,
                "images": ";".join(saved_paths),
            }
            append_found_row(row)
            st.success("Found report saved and indexed.", icon="✅")
    st.markdown("### Recent found items")
    # show a small gallery
    all_found = get_all_found_image_paths()
    if all_found:
        cols = st.columns(3, gap="small")
        for i, p in enumerate(all_found[:9]):
            try:
                img = Image.open(p).convert("RGB")
                with cols[i % 3]:
                    st.image(img, caption=os.path.basename(p), use_container_width=True)
            except Exception:
                continue
    else:
        st.info("No found items yet.")
    st.stop()

# ---------- LOST REPORT / SEARCH RESULTS ----------
if st.session_state.page == "lost_report":
    if st.button("← Back to Lost Form", key="back3"):
        go("lost_form")
        st.stop()
    st.header("Search Results — Matches")
    results = st.session_state.get("search_results", [])
    if not results:
        st.info("No search performed or no matches available.")
    else:
        # Show top matches and table
        top = results[0]
        top_pct = int(top["final"] * 100)
        st.success(f"Top Match: {os.path.basename(top['path'])} · {top_pct}% confidence")
        st.image(top["image"], use_container_width=True)
        st.markdown("### Top matches")
        cols = st.columns(3, gap="small")
        for i, r in enumerate(results[:9]):
            pct = int(r["final"]*100)
            with cols[i % 3]:
                st.image(r["image"], caption=f"{os.path.basename(r['path'])} · {pct}%", use_container_width=True)
        # Detailed table
        df = pd.DataFrame([{"filename": os.path.basename(r["path"]), "score": r["final"]} for r in results])
        st.markdown("### Detailed results")
        st.dataframe(df, use_container_width=True)
    st.stop()