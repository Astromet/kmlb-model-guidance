#!/usr/bin/env python3
"""
Build the KMLB model-guidance data files.

Fetches the latest and the 12-hours-earlier runs of
  * GFS MOS (MAV)                          - weather.gov MDL text bulletins
  * NBM v5 NBS                             - weather.gov MDL text bulletins
  * ECMWF IFS 0.25 (deterministic) + ENS   - Open-Meteo
and writes
  data/models.json    latest runs, one row per model / element / valid time
  data/previous.json  wind, gusts and P12 from each model's run 12 hours earlier
  data/mex.json       GFS extended MOS (MEX), latest run, 12-hourly to day 8, with climatology
  data/ext.json       ECMWF and UKMET extended (raw model), on the MEX valid times
  data/ndfd.json      NWS official forecast (NDFD grid): wind, gusts and P12 at the station
  data/meta.json      when this ran and which runs it used

Usage:  python3 update.py [--station KMLB --lat 28.1028 --lon -80.6453] [--out data]
Only the Python standard library is needed.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

UA = {"User-Agent": "kmlb-model-guidance (github.com/Astromet/kmlb-model-guidance)"}
MDL = "https://www.weather.gov/source/mdl/"
SR = "https://single-runs-api.open-meteo.com/v1/forecast"
ENS = "https://ensemble-api.open-meteo.com/v1/ensemble"
ENS_META = "https://ensemble-api.open-meteo.com/data/ecmwf_ifs025_ensemble/static/meta.json"
UTC = timezone.utc


# ------------------------------------------------------------------ fetching
def get(url, tries=3):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            last = e
        except Exception as e:  # network hiccup: retry
            last = e
        time.sleep(3 * (k + 1))
    print(f"warning: {url} failed: {last}", file=sys.stderr)
    return None


def get_json(url):
    t = get(url)
    return json.loads(t) if t else None


def run_time(text):
    m = re.search(r"(\d\d)/(\d\d)/(\d{4})\s+(\d\d)(\d\d) UTC", text[:400]) if text else None
    return datetime(int(m[3]), int(m[1]), int(m[2]), int(m[4]), tzinfo=UTC) if m else None


def station_block(text, sta, kind):
    if not text:
        return None
    m = re.search(rf"^ {re.escape(sta)}\s+{kind}.*?(?=^\s*$|^ \w{{3,4}}\s+{kind}|\Z)", text, re.M | re.S)
    return m.group(0).rstrip() if m else None


MAX_AGE = timedelta(hours=36)  # a newest run older than this means the product has stopped


def fresh(runs):
    now = datetime.now(UTC)
    return {c: r for c, r in runs.items() if r and now - r <= MAX_AGE}


def latest_and_prior(files, sta, kind):
    """files: {cycle: text}. Returns (latest_block, prior_block) where prior is 12 h earlier."""
    runs = {c: run_time(t) for c, t in files.items() if t}
    if not fresh(runs):
        return None, None
    latest = max(runs, key=runs.get)
    prior = next((c for c, r in runs.items() if r == runs[latest] - timedelta(hours=12)), None)
    return (station_block(files[latest], sta, kind),
            station_block(files[prior], sta, kind) if prior else None)


def fetch_mos(prefix, cycles, sta, kind):
    files = {c: get(f"{MDL}MOS/{prefix}.t{c}z") for c in cycles}
    return latest_and_prior(files, sta, kind)


def fetch_nbs(sta):
    heads = {c: get(f"{MDL}NBM/blend_nbstx.t{c}z_1.txt") for c in ("01", "07", "13", "19")}
    runs = {c: run_time(t) for c, t in heads.items() if t}
    if not fresh(runs):
        return None, None
    latest = max(runs, key=runs.get)
    prior = next((c for c, r in runs.items() if r == runs[latest] - timedelta(hours=12)), None)

    def block(c):
        for p in range(1, 6):
            t = heads[c] if p == 1 else get(f"{MDL}NBM/blend_nbstx.t{c}z_{p}.txt")
            b = station_block(t, sta, r"NBM\s+(?:V[\d.]+\s+)?NBS GUIDANCE")  # header may or may not carry a version
            if b:
                return b
        return None
    return block(latest), (block(prior) if prior else None)


# ------------------------------------------------------------------ conversions (MOS categories)
def cld(p):
    if p is None: return None
    return "CL" if p < 6 else "FW" if p < 32 else "SC" if p < 69 else "BK" if p < 94 else "OV"


def qcat(h):
    """h: hundredths of an inch -> MOS QPF category."""
    if h is None: return None
    for lim, c in ((0, 0), (9, 1), (24, 2), (49, 3), (99, 4), (199, 5)):
        if h <= lim: return c
    return 6


def cigcat(c):
    if c is None: return None
    if c < 0: return 8
    for lim, cat in ((1, 1), (4, 2), (9, 3), (19, 4), (30, 5), (65, 6), (120, 7)):
        if c <= lim: return cat
    return 8


def viscat(v):
    if v is None: return None
    m = v / 10
    for lim, cat in ((0.5, 1), (1, 2), (2, 3), (3, 4), (6, 5)):
        if m < lim: return cat
    return 6 if m == 6 else 7


# ------------------------------------------------------------------ parsing
class Rows(list):
    def add(self, model, run, valid, el, value, text=None):
        if value is None and text is None:
            return
        self.append({"model": model, "run": run.strftime("%Y-%m-%dT%H:%MZ"),
                     "valid": valid.strftime("%Y-%m-%dT%H:%MZ"), "element": el,
                     "value": value, "text": text if text is not None else str(value)})


def fields(line):
    body = line[5:]
    return [body[i:i + 3].strip() for i in range(0, len(body), 3)]


def valid_times(run, hours):
    out, t = [], run
    for h in hours:
        t += timedelta(hours=1)
        while t.hour != h:
            t += timedelta(hours=1)
        out.append(t)
    return out


def num(s):
    return int(s) if re.fullmatch(r"-?\d+", s or "") else None


def parse_mos(block, model, rows, only=None):
    lines = block.split("\n")
    run = run_time(lines[0])
    rowmap = {l[1:4].strip(): l for l in lines[1:]}
    if "N/X" in rowmap and "X/N" not in rowmap:
        rowmap["X/N"] = rowmap["N/X"]  # 12Z/18Z cycles list the min first
    vt = valid_times(run, [int(x) for x in fields(rowmap["HR"]) if x])
    n = len(vt)
    getf = lambda k: (fields(rowmap[k]) + [""] * n)[:n] if k in rowmap else [""] * n
    want = lambda el: only is None or el in only
    for el, key in (("XN", "X/N"), ("TMP", "TMP"), ("DPT", "DPT"), ("WSP", "WSP"),
                    ("P06", "P06"), ("P12", "P12"), ("Q06", "Q06"), ("Q12", "Q12"),
                    ("CIG", "CIG"), ("VIS", "VIS")):
        if not want(el):
            continue
        for t, s in zip(vt, getf(key)):
            v = num(s)
            if v is not None:
                rows.add(model, run, t, el, v, s if el == "WSP" else (s.lstrip("0") or "0"))
    if want("WDR"):
        for t, s in zip(vt, getf("WDR")):
            if num(s) is not None:
                rows.add(model, run, t, "WDR", num(s) * 10, s)
    for el in ("CLD", "TYP"):
        if want(el):
            for t, s in zip(vt, getf(el)):
                if s:
                    rows.add(model, run, t, el, None, s)
    for key in ("T06", "T12"):
        if not want(key):
            continue
        for m in re.finditer(r"(\d+)/\s*(\d+)", rowmap.get(key, "")):
            col = (m.end() - 6) // 3
            if 0 <= col < n:
                rows.add(model, run, vt[col], key, int(m[1]), f"{int(m[1])}/{int(m[2])}")


def parse_nbs(block, rows, only=None):
    lines = block.split("\n")
    run = run_time(lines[0])
    rowmap = {l[1:4].strip(): l for l in lines[1:]}
    vt = valid_times(run, [int(x) for x in fields(rowmap["UTC"]) if x])
    n = len(vt)
    getf = lambda k: [num(x) for x in (fields(rowmap[k]) + [""] * n)[:n]] if k in rowmap else [None] * n
    want = lambda el: only is None or el in only
    M = "NBM"
    for el, key in (("XN", "TXN"), ("TMP", "TMP"), ("DPT", "DPT"), ("WSP", "WSP"), ("GST", "GST"),
                    ("P06", "P06"), ("P12", "P12"), ("T06", "T06"), ("T12", "T12")):
        if want(el):
            for t, v in zip(vt, getf(key)):
                if v is not None:
                    rows.add(M, run, t, el, v, f"{v:02d}" if el in ("WSP", "GST") else str(v))
    if want("WDR"):
        for t, v in zip(vt, getf("WDR")):
            if v is not None:
                v = 36 if v == 0 else v  # NBM writes due north as 0; MOS uses 36
                rows.add(M, run, t, "WDR", v * 10, f"{v:02d}")
    if want("CLD"):
        for t, v in zip(vt, getf("SKY")):
            if v is not None:
                rows.add(M, run, t, "CLD", v, cld(v))
    for el in ("Q06", "Q12"):
        if want(el):
            for t, v in zip(vt, getf(el)):
                if v is not None:
                    rows.add(M, run, t, el, qcat(v), str(qcat(v)))
    if want("CIG"):
        for t, v in zip(vt, getf("CIG")):
            if v is not None:
                rows.add(M, run, t, "CIG", cigcat(v), str(cigcat(v)))
    if want("VIS"):
        for t, v in zip(vt, getf("VIS")):
            if v is not None:
                rows.add(M, run, t, "VIS", viscat(v), str(viscat(v)))
    if want("TYP"):
        pra, psn, pzr, ppl = getf("PRA"), getf("PSN"), getf("PZR"), getf("PPL")
        for i, t in enumerate(vt):
            b = max([(pra[i] if pra[i] is not None else -1, "R"), (psn[i] if psn[i] is not None else -1, "S"),
                     (pzr[i] if pzr[i] is not None else -1, "Z"), (ppl[i] if ppl[i] is not None else -1, "P")])
            if b[0] >= 0:
                rows.add(M, run, t, "TYP", None, b[1])


# ------------------------------------------------------------------ GFS extended MOS (MEX)
def parse_mex(block, rows):
    """GFSX MOS: 12-hourly columns 4 characters wide, then up to two 3-character climatology fields."""
    lines = block.split("\n")
    run = run_time(lines[0])
    fhr_line = next(l for l in lines if l.startswith(" FHR"))
    ncol = (len(fhr_line.rstrip()) - 5 + 1) // 4
    fhrs = [num(fhr_line[5 + 4 * i: 8 + 4 * i].strip()) for i in range(ncol)]
    M = "GFS MEX"
    for l in lines[3:]:
        label = l[1:4].strip()
        if not label:
            continue
        el = "XN" if label in ("X/N", "N/X") else label
        for i, h in enumerate(fhrs):
            f = l[5 + 4 * i: 8 + 4 * i].strip()
            if h is None or not f:
                continue
            v = num(f)
            rows.add(M, run, run + timedelta(hours=h), el, v, f if v is None else (f if el in ("WND",) else str(v)))
        end = 5 + 4 * ncol - 1
        climo = [l[end + 3 * k: end + 3 * k + 3].strip() for k in range(2)]
        climo = [c for c in climo if c and c != "999"]
        if climo:
            rows.append({"model": M, "run": run.strftime("%Y-%m-%dT%H:%MZ"), "valid": "CLIMO",
                         "element": el, "value": None, "text": " ".join(climo)})


def fetch_mex(sta):
    files = {c: get(f"{MDL}MOS/GFSMEX.t{c}z") for c in ("00", "12")}
    cur, _ = latest_and_prior(files, sta, "GFSX MOS GUIDANCE")
    return cur


# ------------------------------------------------------------------ ECMWF and UKMET extended (raw model, MEX layout)
EXT_MODELS = (
    # name, deterministic model, ensemble model, ensemble meta key
    ("ECMWF", "ecmwf_ifs025", "ecmwf_ifs025", "ecmwf_ifs025_ensemble"),
    ("UKMET", "ukmo_global_deterministic_10km", "ukmo_global_ensemble_20km", "ukmo_global_ensemble_20km"),
)


def long_run(model, lat, lon, now):
    """Latest 00Z or 12Z run (the 06Z/18Z runs stop short of day 8)."""
    t = now.replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 12)
    for _ in range(6):
        r = get(f"{SR}?latitude={lat}&longitude={lon}&models={model}&hourly=temperature_2m"
                f"&forecast_days=1&run={iso(t)}", tries=1)
        if r and '"hourly"' in r:
            return t
        t -= timedelta(hours=12)
    return None


def cld3(p):
    """Cloud cover % to the MEX categories: CL (0-2 octas), PC (3-4), OV (5-8)."""
    if p is None: return None
    return "CL" if p < 32 else "PC" if p < 69 else "OV"


def extended(cols, lat, lon, now, rows, meta):
    for name, det_model, ens_model, ens_key in EXT_MODELS:
        run = long_run(det_model, lat, lon, now)
        if not run:
            print(f"warning: no {name} 00Z/12Z run found", file=sys.stderr)
            continue
        det = get_json(f"{SR}?latitude={lat}&longitude={lon}&models={det_model}&timezone=GMT&forecast_days=10"
                       "&wind_speed_unit=kn&temperature_unit=fahrenheit&precipitation_unit=inch"
                       f"&run={iso(run)}&hourly=temperature_2m,dew_point_2m,cloud_cover,wind_speed_10m,"
                       "precipitation,snowfall,cape")
        ens = get_json(f"{ENS}?latitude={lat}&longitude={lon}&models={ens_model}&hourly=precipitation"
                       "&precipitation_unit=inch&timezone=GMT&past_days=1&forecast_days=10")
        em = get_json(f"https://ensemble-api.open-meteo.com/data/{ens_key}/static/meta.json")
        if not det:
            continue
        ens_run = iso(datetime.fromtimestamp(em["last_run_initialisation_time"], UTC)) + "Z" if em else None
        meta[f"{name} extended"] = iso(run) + "Z"
        if ens_run:
            meta[f"{name} extended ENS"] = ens_run
        H = det["hourly"]
        idx = {t: i for i, t in enumerate(H["time"])}
        E = ens["hourly"] if ens else {"time": []}
        eidx = {t: i for i, t in enumerate(E["time"])}
        members = [k for k in E if k.startswith("precipitation")]

        def win(arr, ix, end, h):
            out = []
            for k in range(h):
                i = ix.get(iso(end - timedelta(hours=k)))
                if i is None or arr[i] is None:
                    return None
                out.append(arr[i])
            return out

        def add(c, el, value, text=None):
            rows.add(name, run, c, el, value, text)
            if ens_run:
                rows[-1]["ens_run"] = ens_run
            rows[-1]["members"] = len(members)

        for c in cols:
            i = idx.get(iso(c))
            if i is None or H["temperature_2m"][i] is None:
                continue
            w = win(H["temperature_2m"], idx, c, 13)
            if w:
                add(c, "XN", round(max(w) if c.hour == 0 else min(w)))
            add(c, "TMP", round(H["temperature_2m"][i]))
            if H["dew_point_2m"][i] is not None:
                add(c, "DPT", round(H["dew_point_2m"][i]))
            if H["cloud_cover"][i] is not None:
                add(c, "CLD", H["cloud_cover"][i], cld3(H["cloud_cover"][i]))
            w = win(H["wind_speed_10m"], idx, c, 12)
            if w:
                add(c, "WND", round(max(w)), str(round(max(w))))
            for h in (12, 24):
                if h == 24 and c.hour != 0:
                    continue
                tot = wet = 0
                for m in members:
                    sv = win(E[m], eidx, c, h)
                    if sv is not None:
                        tot += 1
                        wet += sum(sv) >= 0.0495  # share of members with >= 0.05 in
                if tot:
                    add(c, f"P{h}", round(100 * wet / tot))
            for h in (12, 24):
                if h == 24 and c.hour != 0:
                    continue
                w = win(H["precipitation"], idx, c, h)
                if w is not None:
                    cat = qcat(round(sum(w) * 100))
                    add(c, f"Q{h}", cat, str(cat))
            if "cape" in H:
                w = win(H["cape"], idx, c, 12)
                if w:
                    add(c, "CAP", max(w), str(round(max(w) / 100)))
            p12, s12 = win(H["precipitation"], idx, c, 12), win(H["snowfall"], idx, c, 12)
            if p12 is not None:
                add(c, "TYP", None, "-" if sum(p12) < 0.005 else ("S" if s12 and sum(s12) > 0 else "R"))


# ------------------------------------------------------------------ ECMWF
def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M")


def ecmwf_latest_run(lat, lon, now):
    t = now.replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 6)
    for _ in range(8):
        r = get(f"{SR}?latitude={lat}&longitude={lon}&models=ecmwf_ifs025&hourly=temperature_2m"
                f"&forecast_days=1&run={iso(t)}", tries=1)
        if r and '"hourly"' in r:
            return t
        t -= timedelta(hours=6)
    return None


def ecmwf(lat, lon, now, rows, prev_rows, meta):
    run = ecmwf_latest_run(lat, lon, now)
    if not run:
        print("warning: no ECMWF run found", file=sys.stderr)
        return
    q = (f"latitude={lat}&longitude={lon}&models=ecmwf_ifs025&timezone=GMT&forecast_days=5"
         "&wind_speed_unit=kn&temperature_unit=fahrenheit&precipitation_unit=inch")
    det = get_json(f"{SR}?{q}&run={iso(run)}&hourly=temperature_2m,dew_point_2m,cloud_cover,"
                   "wind_direction_10m,wind_speed_10m,wind_gusts_10m,precipitation,snowfall,cape")
    ens = get_json(f"{ENS}?latitude={lat}&longitude={lon}&models=ecmwf_ifs025&hourly=precipitation"
                   "&precipitation_unit=inch&timezone=GMT&past_days=1&forecast_days=5")
    em = get_json(ENS_META)
    meta["ECMWF"] = iso(run) + "Z"
    if em:
        meta["ECMWF ENS"] = iso(datetime.fromtimestamp(em["last_run_initialisation_time"], UTC)) + "Z"
    if not det:
        return
    H = det["hourly"]
    idx = {t: i for i, t in enumerate(H["time"])}
    E = ens["hourly"] if ens else {"time": []}
    eidx = {t: i for i, t in enumerate(E["time"])}
    members = [k for k in E if k.startswith("precipitation")]

    def win(arr, ix, end, h):
        out = []
        for k in range(h):
            i = ix.get(iso(end - timedelta(hours=k)))
            if i is None or arr[i] is None:
                return None
            out.append(arr[i])
        return out

    start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    while start.hour % 3:
        start += timedelta(hours=1)
    M = "ECMWF"
    for j in range(27):
        c = start + timedelta(hours=3 * j)
        i = idx.get(iso(c))
        if i is None:
            continue
        vals = [H[k][i] for k in ("temperature_2m", "dew_point_2m", "cloud_cover",
                                  "wind_direction_10m", "wind_speed_10m", "wind_gusts_10m")]
        if any(v is None for v in vals):
            continue
        tmp, dpt, sky, wdr, wsp, gst = vals
        if c.hour in (0, 12):
            w = win(H["temperature_2m"], idx, c, 13)
            if w:
                rows.add(M, run, c, "XN", round(max(w) if c.hour == 0 else min(w)))
        rows.add(M, run, c, "TMP", round(tmp))
        rows.add(M, run, c, "DPT", round(dpt))
        rows.add(M, run, c, "CLD", sky, cld(sky))
        d = round(wdr / 10) % 36 or 36
        rows.add(M, run, c, "WDR", d * 10, f"{d:02d}")
        rows.add(M, run, c, "WSP", round(wsp), f"{round(wsp):02d}")
        rows.add(M, run, c, "GST", round(gst), f"{round(gst):02d}")
        p3, s3 = win(H["precipitation"], idx, c, 3), win(H["snowfall"], idx, c, 3)
        if p3 is not None:
            rows.add(M, run, c, "TYP", None, "-" if sum(p3) < 0.005 else ("S" if s3 and sum(s3) > 0 else "R"))
        for h, ok in ((6, c.hour % 6 == 0), (12, c.hour % 12 == 0)):
            if not ok:
                continue
            w = win(H["precipitation"], idx, c, h)
            if w is not None:
                cat = qcat(round(sum(w) * 100))
                rows.add(M, run, c, f"Q{h:02d}", cat, str(cat))
            tot = wet = 0
            for m in members:
                s = win(E[m], eidx, c, h)
                if s is not None:
                    tot += 1
                    wet += sum(s) >= 0.0495  # share of members with >= 0.05 in
            if tot:
                rows.add(M, run, c, f"P{h:02d}", round(100 * wet / tot))
            if h == 6:
                cp = win(H["cape"], idx, c, 6)
                if cp:
                    rows.add(M, run, c, "CAP", max(cp), str(round(max(cp) / 100)))
    # the run 12 hours earlier: wind and gusts
    prior = run - timedelta(hours=12)
    pj = get_json(f"{SR}?{q}&run={iso(prior)}&hourly=wind_speed_10m,wind_gusts_10m")
    if pj:
        meta["ECMWF prior"] = iso(prior) + "Z"
        P = pj["hourly"]
        for i, t in enumerate(P["time"]):
            v = datetime.fromisoformat(t).replace(tzinfo=UTC)
            if v.hour % 3 or v < start or P["wind_speed_10m"][i] is None or P["wind_gusts_10m"][i] is None:
                continue
            w, g = round(P["wind_speed_10m"][i]), round(P["wind_gusts_10m"][i])
            prev_rows.add(M, prior, v, "WSP", w, f"{w:02d}")
            prev_rows.add(M, prior, v, "GST", g, f"{g:02d}")


# ------------------------------------------------------------------ NWS official forecast (NDFD grid)
API = "https://api.weather.gov"


def duration_hours(d):
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?", d)
    return int(m[1] or 0) * 24 + int(m[2] or 0) + (1 if m[3] else 0) if m else 1


def hourly(layer):
    """Expand an NWS grid layer ('start/duration' intervals) to {hour: value}."""
    out = {}
    for v in layer.get("values", []):
        start, dur = v["validTime"].split("/")
        t = datetime.fromisoformat(start).astimezone(UTC)
        for h in range(duration_hours(dur)):
            out[t + timedelta(hours=h)] = v["value"]
    return out


def ndfd(lat, lon, rows, meta):
    """NWS gridded forecast at the station: wind and gusts (kt) every 3 h, and P12 at 00Z/12Z
    taken as the highest NWS precipitation chance in that 12-hour period."""
    hdr = {"User-Agent": UA["User-Agent"], "Accept": "application/geo+json"}

    def api(url):
        for k in range(3):
            try:
                req = urllib.request.Request(url, headers=hdr)
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.load(r)
            except Exception as e:
                last = e
                time.sleep(3 * (k + 1))
        print(f"warning: {url} failed: {last}", file=sys.stderr)
        return None
    pt = api(f"{API}/points/{lat:.4f},{lon:.4f}")
    if not pt:
        return
    g = api(pt["properties"]["forecastGridData"])
    if not g:
        return
    G = g["properties"]
    upd = datetime.fromisoformat(G["updateTime"]).astimezone(UTC)
    meta["NWS"] = upd.strftime("%Y-%m-%dT%H:%MZ")
    meta["NWS grid"] = f'{pt["properties"]["gridId"]} {pt["properties"]["gridX"]},{pt["properties"]["gridY"]}'
    kt = 1 / 1.852  # km/h to knots
    spd, gst, pop = hourly(G.get("windSpeed", {})), hourly(G.get("windGust", {})), hourly(G.get("probabilityOfPrecipitation", {}))
    for t in sorted(spd):
        if t.hour % 3 == 0 and spd[t] is not None:
            w = round(spd[t] * kt)
            rows.add("NWS", upd, t, "WSP", w, f"{w:02d}")
    for t in sorted(gst):
        if t.hour % 3 == 0 and gst[t] is not None:
            w = round(gst[t] * kt)
            rows.add("NWS", upd, t, "GST", w, f"{w:02d}")
    ends = sorted({t.replace(hour=h) + timedelta(days=dd) for t in pop for h in (0, 12) for dd in (0, 1)})
    for e in ends:
        win = [pop.get(e - timedelta(hours=k)) for k in range(12)]
        if all(v is not None for v in win):
            rows.add("NWS", upd, e, "P12", max(win))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="KMLB")
    ap.add_argument("--lat", type=float, default=28.1028)
    ap.add_argument("--lon", type=float, default=-80.6453)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    sta = a.station.upper()
    now = datetime.now(UTC)
    rows, prev = Rows(), Rows()
    meta = {}
    prior_only = {"WSP", "GST", "P12"}

    mav, mav_p = fetch_mos("GFSMAV", ("00", "06", "12", "18"), sta, "GFS MOS GUIDANCE")
    nbs, nbs_p = fetch_nbs(sta)
    for name, cur, pri, parse in (("GFS MOS", mav, mav_p, lambda b, r, o: parse_mos(b, "GFS MOS", r, o)),
                                  ("NBM", nbs, nbs_p, parse_nbs)):
        if cur:
            parse(cur, rows, None)
            meta[name] = run_time(cur).strftime("%Y-%m-%dT%H:%MZ")
        else:
            print(f"warning: no {name} bulletin for {sta}", file=sys.stderr)
        if pri:
            parse(pri, prev, prior_only)
            meta[name + " prior"] = run_time(pri).strftime("%Y-%m-%dT%H:%MZ")
    try:
        ecmwf(a.lat, a.lon, now, rows, prev, meta)
    except Exception as e:  # keep the MOS data even if ECMWF fails
        print(f"warning: ECMWF failed: {e}", file=sys.stderr)

    mexr = Rows()
    mex = fetch_mex(sta)
    if mex:
        parse_mex(mex, mexr)
        meta["GFS MEX"] = run_time(mex).strftime("%Y-%m-%dT%H:%MZ")
    else:
        print(f"warning: no GFS MEX bulletin for {sta}", file=sys.stderr)

    extr = Rows()
    cols = sorted({datetime.strptime(r["valid"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
                   for r in mexr if r["valid"] != "CLIMO"})
    if not cols:  # no MEX: 00Z/12Z from tomorrow out to day 8
        t0 = (now + timedelta(hours=12)).replace(minute=0, second=0, microsecond=0)
        while t0.hour % 12:
            t0 += timedelta(hours=1)
        cols = [t0 + timedelta(hours=12 * k) for k in range(15)]
    try:
        extended(cols, a.lat, a.lon, now, extr, meta)
    except Exception as e:
        print(f"warning: extended ECMWF/UKMET failed: {e}", file=sys.stderr)

    nws = Rows()
    try:
        ndfd(a.lat, a.lon, nws, meta)
    except Exception as e:
        print(f"warning: NWS grid failed: {e}", file=sys.stderr)

    if not rows:
        sys.exit("No data fetched; leaving the existing files alone.")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "models.json"), "w") as f:
        json.dump({"rows": rows}, f, separators=(",", ":"))
    with open(os.path.join(a.out, "previous.json"), "w") as f:
        json.dump({"rows": prev}, f, separators=(",", ":"))
    with open(os.path.join(a.out, "mex.json"), "w") as f:
        json.dump({"rows": mexr}, f, separators=(",", ":"))
    with open(os.path.join(a.out, "ext.json"), "w") as f:
        json.dump({"rows": extr}, f, separators=(",", ":"))
    with open(os.path.join(a.out, "ndfd.json"), "w") as f:
        json.dump({"rows": nws}, f, separators=(",", ":"))
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"station": sta, "generated": now.strftime("%Y-%m-%dT%H:%MZ"), "runs": meta}, f, indent=1)
    print(f"{len(rows)} rows, {len(prev)} prior rows, {len(nws)} NWS rows, {len(mexr)} MEX rows, {len(extr)} extended rows; runs: {meta}")


if __name__ == "__main__":
    main()
