# web_app.py — AgriSphere with DB saving (full file)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
import time
import json
import re
import urllib.parse as urlparse
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim

# DB imports (make sure db.py is present in project root)
from db import init_db, save_recommendation_record

st.set_page_config(page_title="AgriSphere", layout="wide")

# ---------- Config ----------
MODEL_PATH = "model.joblib"
DATASET_PATH = "Crop_recommendation.csv"
PRICE_CACHE_FILE = "price_cache.json"
PRICE_TTL = 6 * 3600  # seconds

LOCATION_PROFILES = {
    "Punjab (Ludhiana)": {"lat": 30.9010, "lon": 75.8573, "state": "Punjab", "N": 90, "P": 45, "K": 45, "ph": 6.5, "rainfall": 650},
    "Himachal (Shimla)": {"lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh", "N": 50, "P": 25, "K": 35, "ph": 6.3, "rainfall": 1250},
    "Maharashtra (Pune)": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra", "N": 60, "P": 30, "K": 30, "ph": 7.1, "rainfall": 700}
}

TROPICAL_CROPS = {"coffee", "mango", "banana", "coconut", "papaya", "pineapple", "arecanut", "cashew"}

# ---------- Initialize DB ----------
# This will create tables if they don't exist (SQLite by default in db.py)
init_db()

# ---------- Price cache helpers ----------
def load_price_cache():
    if not os.path.exists(PRICE_CACHE_FILE):
        return {}
    try:
        with open(PRICE_CACHE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}

def save_price_cache(cache):
    try:
        with open(PRICE_CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)
    except Exception:
        pass

def clear_price_cache():
    try:
        if os.path.exists(PRICE_CACHE_FILE):
            os.remove(PRICE_CACHE_FILE)
    except Exception:
        pass

def get_cached_price(key):
    cache = load_price_cache()
    rec = cache.get(key)
    if not rec:
        return None
    if time.time() - rec.get("ts", 0) > PRICE_TTL:
        return None
    price = rec.get("price")
    if not is_valid_price_string(price):
        return None
    return price

def set_cached_price_if_valid(key, price):
    if not is_valid_price_string(price):
        return
    cache = load_price_cache()
    cache[key] = {"price": price, "ts": time.time()}
    save_price_cache(cache)

# ---------- Utility helpers ----------
def is_valid_price_string(s):
    if not s or not isinstance(s, str):
        return False
    m = re.search(r"(\d{1,6}[,\.]?\d*)", s)
    if not m:
        return False
    num_s = m.group(1).replace(",", "").replace(".", "")
    if set(num_s) == {"0"}:
        return False
    try:
        num = float(m.group(1).replace(",", ""))
    except Exception:
        return False
    if num < 10:
        return False
    return True

# ---------- Geo helpers ----------
@st.cache_data
def geolocate_ip():
    providers = ["http://ip-api.com/json", "https://ipinfo.io/json"]
    for url in providers:
        try:
            r = requests.get(url, timeout=5)
            j = r.json()
            if "lat" in j and "lon" in j:
                return {"lat": float(j["lat"]), "lon": float(j["lon"]), "city": j.get("city", "")}
            if "loc" in j:
                lat, lon = j["loc"].split(",")
                return {"lat": float(lat), "lon": float(lon), "city": j.get("city", "")}
        except Exception:
            continue
    return None

@st.cache_data
def reverse_geocode(lat, lon):
    try:
        geo = Nominatim(user_agent="agrisphere")
        loc = geo.reverse((lat, lon), language="en", timeout=10)
        adr = loc.raw.get("address", {})
        return {"state": adr.get("state") or adr.get("region") or "", "city": adr.get("city") or adr.get("town") or ""}
    except Exception:
        return {"state": "", "city": ""}

# ---------- Mandi scraper helpers ----------
def extract_price_candidates(text):
    if not text:
        return []
    t = text.replace("\u20B9", "Rs").replace("₹", "Rs")
    candidates = []
    patterns = [
        r"Rs\.?\s*[:\-]?\s*(\d{1,6}[,\.]?\d*)\s*(?:/|per)?\s*(quintal|qtl|kg|ton|tonne|kgms)?",
        r"(\d{1,6}[,\.]?\d*)\s*(quintal|qtl|kg|ton|tonne|kgms)",
        r"(?:Modal Price|Modal)\s*[:\-]?\s*(\d{1,6}[,\.]?\d*)"
    ]
    for pat in patterns:
        for m in re.finditer(pat, t, flags=re.I):
            num = m.group(1)
            unit = None
            if m.lastindex and m.lastindex >= 2:
                unit = m.group(2)
            candidates.append((num, unit))
    for m in re.finditer(r"\b(\d{3,6})\b", t):
        candidates.append((m.group(1), None))
    return candidates

def canonical_price_from_candidate(num_s, unit):
    try:
        num = float(num_s.replace(",", ""))
    except Exception:
        return None
    if num < 10:
        return None
    unit_norm = (unit or "").lower() if unit else ""
    if unit_norm in ("kg",) or num < 50:
        num_q = num * 100.0
    else:
        num_q = num
    num_q = int(round(num_q))
    return f"₹{num_q} / Quintal"

def follow_first_link(soup, base_url):
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(" ").lower()
        if "market" in text or "details" in text or "arrival" in text or "price" in text:
            if href.startswith("http"):
                return href
            else:
                return urlparse.urljoin(base_url, href)
    return None

# Primary agmarknet try
def try_agmarknet(crop, state, debug=False):
    scrop = urlparse.quote_plus(crop)
    sstate = urlparse.quote_plus(state or "")
    url = f"https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity={scrop}&Tx_StateName={sstate}&Tx_District=0&Tx_Market=0&DateFrom="
    headers = {"User-Agent": "Mozilla/5.0 (AgriSphere/1.0)", "Accept-Language": "en-US,en;q=0.9"}
    try:
        r = requests.get(url, headers=headers, timeout=12)
        status = r.status_code
        text = r.text
        if status != 200:
            return None, {"url": url, "status": status, "snippet": (text or "")[:2000]}
        soup = BeautifulSoup(text, "lxml")
        for tbl in soup.find_all("table"):
            txt = tbl.get_text(" ")
            candidates = extract_price_candidates(txt)
            for num_s, unit in candidates:
                c = canonical_price_from_candidate(num_s, unit)
                if c:
                    return c, {"url": url, "status": status}
        keywords = ["Modal Price", "Modal", "Price", "Quintal", "₹", "Rs"]
        for kw in keywords:
            for node in soup.find_all(string=re.compile(re.escape(kw), re.I)):
                snippet = node.parent.get_text(" ")
                for num_s, unit in extract_price_candidates(snippet):
                    c = canonical_price_from_candidate(num_s, unit)
                    if c:
                        return c, {"url": url, "status": status}
        for num_s, unit in extract_price_candidates(text):
            c = canonical_price_from_candidate(num_s, unit)
            if c:
                return c, {"url": url, "status": status}
        return None, {"url": url, "status": status, "snippet": text[:2000]}
    except requests.RequestException as e:
        return None, {"url": url, "status": getattr(e, 'response', None) and e.response.status_code or "ERR", "error": str(e)}
    except Exception as e:
        return None, {"url": url, "status": "ERR", "error": str(e)}

# Fallback scrapers
def try_napanta(crop, state):
    try:
        sc = crop.lower().replace(" ", "-")
        ss = state.lower().replace(" ", "-") if state else ""
        url = f"https://www.napanta.com/agri-commodity-prices/{ss}/{sc}/"
        r = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200:
            return None, {"url": url, "status": r.status_code}
        t = r.text
        for num_s, unit in extract_price_candidates(t):
            c = canonical_price_from_candidate(num_s, unit)
            if c:
                return c, {"url": url, "status": r.status_code}
        return None, {"url": url, "status": r.status_code, "snippet": t[:2000]}
    except Exception as e:
        return None, {"url": url, "status": "ERR", "error": str(e)}

def try_commodityonline(crop, state):
    try:
        sc = crop.lower().replace(" ", "-")
        ss = state.lower().replace(" ", "-") if state else ""
        url = f"https://www.commodityonline.com/mandiprices/{sc}/{ss}"
        r = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200:
            return None, {"url": url, "status": r.status_code}
        t = r.text
        for num_s, unit in extract_price_candidates(t):
            c = canonical_price_from_candidate(num_s, unit)
            if c:
                return c, {"url": url, "status": r.status_code}
        return None, {"url": url, "status": r.status_code, "snippet": t[:2000]}
    except Exception as e:
        return None, {"url": url, "status": "ERR", "error": str(e)}

# Combined get_mandi_price
def get_mandi_price(crop, state, debug=False):
    crop = (crop or "").strip()
    state = (state or "").strip()
    if not crop:
        return "N/A"
    key = f"{crop.lower()}__{state.lower()}"
    cached = get_cached_price(key)
    if cached:
        return cached + " (cached)"
    ag_val, ag_meta = try_agmarknet(crop, state, debug=debug)
    if ag_val:
        set_cached_price_if_valid(key, ag_val)
        return ag_val
    nap_val, nap_meta = try_napanta(crop, state)
    if nap_val:
        set_cached_price_if_valid(key, nap_val)
        return nap_val + " (napanta)"
    com_val, com_meta = try_commodityonline(crop, state)
    if com_val:
        set_cached_price_if_valid(key, com_val)
        return com_val + " (commodityonline)"
    if debug:
        meta = ag_meta or nap_meta or com_meta or {}
        return f"N/A — debug info: {json.dumps(meta)}"
    return "N/A"

# ---------- Weather ----------
def fetch_weather(lat, lon, key):
    try:
        u = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
        r = requests.get(u, timeout=8)
        if r.status_code != 200:
            return None, None
        j = r.json()
        t = float(j['main']['temp']); h = float(j['main']['humidity'])
        if not (-40 <= t <= 60) or not (0 <= h <= 100):
            return None, None
        return t, h
    except Exception:
        return None, None

# ---------- UI ----------
st.title("🌾 AgriSphere — Crop Recommendation & Mandi Prices (DB integrated)")

# cache controls
if st.button("Clear price cache"):
    clear_price_cache()
    st.success("Cleared price cache.")

q = dict(st.query_params or {})
lat_q = float(q.get("lat", 20.0))
lon_q = float(q.get("lon", 78.0))

col1, col2 = st.columns([1, 2])
with col1:
    st.header("Location")
    if st.button("Use IP Location"):
        d = geolocate_ip()
        if d:
            st.session_state["lat"] = d["lat"]
            st.session_state["lon"] = d["lon"]
            st.success(f"IP located: {d.get('city','')}")
    lat = st.number_input("Latitude", value=float(st.session_state.get("lat", lat_q)))
    lon = st.number_input("Longitude", value=float(st.session_state.get("lon", lon_q)))
    st.query_params = {"lat": str(lat), "lon": str(lon)}
    profile = st.selectbox("Nearest profile (optional)", ["(none)"] + list(LOCATION_PROFILES.keys()))
    if profile != "(none)":
        p = LOCATION_PROFILES[profile]
        lat, lon = p["lat"], p["lon"]
        st.session_state["lat"], st.session_state["lon"] = lat, lon
        st.query_params = {"lat": str(lat), "lon": str(lon)}
        st.success(f"Using profile: {profile}")

with col2:
    st.header("Soil & Weather")
    rg = reverse_geocode(lat, lon)
    guessed_state = rg.get("state", "") or ""
    state = st.text_input("State (for mandi lookup)", value=guessed_state)

    API_KEY = st.secrets.get("OPENWEATHERMAP_API_KEY", "") or os.getenv("OPENWEATHERMAP_API_KEY", "")
    temp, hum = (None, None)
    if API_KEY:
        temp, hum = fetch_weather(lat, lon, API_KEY)
    if temp is None:
        temp = st.number_input("Temperature (°C)", value=25.0)
    else:
        st.metric("Temperature (°C)", f"{temp:.1f} °C")
    if hum is None:
        hum = st.number_input("Humidity (%)", value=60.0)
    else:
        st.metric("Humidity (%)", f"{hum}%")

    N = st.number_input("N", value=60)
    P = st.number_input("P", value=30)
    K = st.number_input("K", value=30)
    ph = st.number_input("pH", value=6.5, format="%.2f")
    rainfall = st.number_input("Avg Rainfall (mm/year)", value=700.0)

st.write("---")

# load model safely
@st.cache_data
def load_model_safe(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error("Could not load model.joblib. Retrain locally with your scikit-learn version (see README).")
        st.code(str(e))
        return None

model_art = load_model_safe(MODEL_PATH)
if model_art is None:
    st.stop()

if isinstance(model_art, dict) and "model" in model_art:
    model = model_art["model"]
    saved_features = model_art.get("features")
else:
    model = model_art
    saved_features = None

def build_input_df(saved_features):
    default_order = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    order = saved_features if saved_features else default_order
    mapping = {"N": float(N), "P": float(P), "K": float(K), "temperature": float(temp), "humidity": float(hum), "ph": float(ph), "rainfall": float(rainfall)}
    row = [mapping.get(f, 0.0) for f in order]
    return pd.DataFrame([row], columns=order), order

if st.button("Get Recommendations"):
    df_in, order = build_input_df(saved_features)

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(df_in)[0]
            classes = model.classes_
        except Exception as e:
            st.warning(f"predict_proba failed: {e}")
            preds = model.predict(df_in)
            probs = None
    else:
        preds = model.predict(df_in)
        probs = None

    if probs is None:
        classes = model.classes_
        probs_vec = np.zeros(len(classes))
        try:
            pred_label = preds[0] if isinstance(preds, (list, np.ndarray)) else preds
            idx_pred = int(np.where(classes == pred_label)[0][0])
            probs_vec[idx_pred] = 1.0
        except Exception:
            probs_vec = np.ones(len(classes)) / len(classes)
        probs = probs_vec

    idx_sorted = np.argsort(probs)[::-1][:15]

    # prepare dataset stats for explanations
    stats = None
    if os.path.exists(DATASET_PATH):
        try:
            df_stats = pd.read_csv(DATASET_PATH)
            stats = df_stats.groupby("label").mean().to_dict(orient="index")
        except Exception:
            stats = None

    # compute results list
    Pmin, Pmax = 500.0, 8000.0
    results = []
    debug_infos = {}
    for j in idx_sorted:
        crop = classes[j]
        suitability = float(probs[j])
        price_raw = get_mandi_price(crop, state or "", debug=True)
        low_conf = False
        price_display = price_raw
        debug_info = None
        if isinstance(price_raw, str) and price_raw.startswith("N/A"):
            low_conf = True
            debug_info = price_raw
            price_display = "N/A"
        m = re.search(r"(\d{1,6}[,\.]?\d*)", str(price_raw))
        price_num = None
        if m:
            try:
                price_num = float(m.group(1).replace(",", ""))
            except Exception:
                price_num = None
        p_norm = 0.0
        if price_num is not None:
            if price_num < 50:
                price_q = price_num * 100.0
            else:
                price_q = price_num
            p_norm = (price_q - Pmin) / (Pmax - Pmin)
            p_norm = max(0.0, min(1.0, p_norm))
        # tropical sanity
        try:
            temp_val = float(temp)
        except:
            temp_val = 25.0
        suit_adj = suitability
        if temp_val < 12 and crop.lower() in TROPICAL_CROPS:
            suit_adj = suitability * 0.1
        final_score = 0.6 * suit_adj + 0.4 * p_norm

        results.append({"crop": crop, "suitability": suit_adj, "price_display": price_display, "price_num": price_num, "final_score": final_score, "low_conf": low_conf})
        if debug_info:
            debug_infos[crop] = debug_info

    # ------------------------
    # --- save to DB (optional) ---
    # This is the block you requested — it saves session + top-3 + parsed prices into the DB.
    try:
        session_info = {
            "lat": lat, "lon": lon, "state": state, "profile": profile,
            "temperature": float(temp), "humidity": float(hum),
            "N": float(N), "P": float(P), "K": float(K), "ph": float(ph), "rainfall": float(rainfall)
        }

        top_results_for_db = []
        price_info_list = []
        for rank, r in enumerate(results[:3], start=1):
            top_results_for_db.append({
                "rank": rank,
                "crop": r["crop"],
                "suitability": r.get("suitability"),
                "final_score": r.get("final_score")
            })
            # price parsing: try extract numeric price in ₹/quintal (price_num may be None)
            price_q = None
            if r.get("price_num") is not None:
                pn = r.get("price_num")
                if pn < 50:
                    price_q = float(pn) * 100.0
                else:
                    price_q = float(pn)
            price_info_list.append({
                "crop": r["crop"],
                "source": r.get("price_display", "unknown"),
                "price_q": price_q,
                "raw": r.get("price_display")
            })

        rec_id = save_recommendation_record(session_info, top_results_for_db, price_info_list)
        st.info(f"Saved recommendation record id: {rec_id}")
    except Exception as e:
        st.error(f"Failed saving to DB: {e}")
    # ------------------------

    # display top 3 results to user
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    for rank, r in enumerate(results[:3], start=1):
        st.markdown(f"### #{rank} — {r['crop']}")
        st.write(f"**Suitability:** {r['suitability']*100:.2f}%")
        st.write(f"**Mandi:** {r['price_display']}")
        if r["low_conf"]:
            st.warning("Mandi price uncertain — scraper returned no clear price. Use Test panel below or enter manual price.")
            if r["crop"] in debug_infos:
                if st.checkbox(f"Show debug info for {r['crop']}", key=f"dbg_{r['crop']}"):
                    st.code(debug_infos[r['crop']], language="json")
        st.write("**Why recommended (feature match vs dataset avg):**")
        if stats and r["crop"] in stats:
            crop_stats = stats[r["crop"]]
            for f in ["temperature", "humidity", "ph", "rainfall", "N", "P", "K"]:
                if f in crop_stats:
                    user_val = {'temperature': temp, 'humidity': hum, 'ph': ph, 'rainfall': rainfall, 'N': N, 'P': P, 'K': K}.get(f)
                    st.write(f"• {f}: ideal {crop_stats[f]:.1f}, yours {user_val}")
        else:
            st.write("No dataset stats available for explanation.")

    st.write("---")
    manu = st.text_input("If mandi price missing or wrong, enter approximate price (e.g. 1500 or 18/kg or 1500/quintal) to re-rank", value="")
    if manu:
        m = re.search(r"(\d{1,6}[,\.]?\d*)", manu)
        if m:
            val = float(m.group(1).replace(",", ""))
            if val < 50:
                val_q = val * 100.0
            else:
                val_q = val
            pnorm = max(0.0, min(1.0, (val_q - Pmin) / (Pmax - Pmin)))
            st.success("Manual price applied (display only).")
            for i, r in enumerate(results[:3], start=1):
                updated = 0.6 * r["suitability"] + 0.4 * pnorm
                st.write(f"Updated score for {r['crop']}: {updated:.3f}")

# Test panel
st.write("---")
st.header("Test Mandi Scraper (debug)")
with st.expander("Open test panel"):
    tcrop = st.text_input("Crop to test", value="rice")
    tstate = st.text_input("State to test", value="")
    if st.button("Run Test Scraper"):
        out = get_mandi_price(tcrop, tstate, debug=True)
        st.write("Result:", out)
        if isinstance(out, str) and out.startswith("N/A"):
            st.code(out, language="json")
    manual_price_to_save = st.text_input("Manual price to save (optional; e.g. '1500' or '₹1500 / Quintal')", key="manual_save")
    if st.button("Save manual price to cache"):
        if manual_price_to_save:
            key = f"{tcrop.lower()}__{tstate.lower()}"
            set_cached_price_if_valid(key, manual_price_to_save)
            st.success("Saved manual price to cache for this crop+state.")
