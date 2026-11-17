#!/usr/bin/env python3
"""
train_mos.py - train KMLB MOS equations for ECMWF and UKMET.

Predictands (KMLB ASOS, Iowa Environmental Mesonet archive):
  TMP, DPT, WSP every 3 h; XN max 12Z-00Z / min 00Z-12Z from hourly temperatures;
  P06, P12, P24 = measurable precipitation (>= .01 in) in the period.
Predictors (mos.py) come from past forecasts at the KMLB grid point:
  ECMWF IFS 0.25  - Open-Meteo previous-runs archive, lead days 0-7 (Feb 2024 onward)
  UKMET global    - previous-runs archive where complete, then individual 00Z/12Z runs (last ~7 months)
Method, as in NWS MOS:
  continuous elements: multiple linear regression with forward selection (up to 6 terms)
  probabilities: REEP (linear regression on the 0/1 event), clipped to 0-100%
  one equation per model, element, lead day and valid hour; season enters as sin/cos of the day of year.
The last 60 days are held out to score raw vs MOS; the operational equations are then refit on all data.

Writes mos/coefficients.json and mos/verification.json.
Needs numpy.
"""
import csv
import io
import json
import math
import os
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

import mos

UTC = timezone.utc
H = timedelta(hours=1)
LAT, LON = 28.1028, -80.6453
STATION = "MLB"
HOLDOUT_DAYS = 60
MAX_TERMS, MIN_GAIN, MIN_N = 6, 0.003, 80
UA = {"User-Agent": "kmlb-model-guidance MOS trainer (github.com/Astromet/kmlb-model-guidance)"}
PREV = "https://previous-runs-api.open-meteo.com/v1/forecast"
SINGLE = "https://single-runs-api.open-meteo.com/v1/forecast"
UNITS = "&temperature_unit=fahrenheit&wind_speed_unit=kn&precipitation_unit=inch&timezone=GMT"
MODELS = {
    "ECMWF": {"om": "ecmwf_ifs025", "prev_start": datetime(2024, 2, 1, tzinfo=UTC), "single": False},
    "UKMET": {"om": "ukmo_global_deterministic_10km", "prev_start": datetime(2024, 8, 1, tzinfo=UTC), "single": True},
}


def log(*a):
    print(*a, flush=True)


def get(url, tries=4, pause=0.0):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read().decode("utf-8", "replace")
            if pause:
                time.sleep(pause)
            return body
        except urllib.error.HTTPError as e:
            if e.code == 400:
                return None
            last = e
        except Exception as e:
            last = e
        time.sleep(10 * (k + 1))
    log(f"warning: {url[:120]} failed: {last}")
    return None


def parse_hourly(j, suffix=""):
    """Open-Meteo hourly block -> {key: {datetime: value}} for our variables (with an optional name suffix)."""
    times = [datetime.fromisoformat(t).replace(tzinfo=UTC) for t in j["hourly"]["time"]]
    out = {}
    for key, var in mos.OM_VARS.items():
        vals = j["hourly"].get(var + suffix)
        if vals is None:
            continue
        out[key] = {t: v for t, v in zip(times, vals) if v is not None}
    return out


# ------------------------------------------------------------------ observations
def fetch_obs(start, end):
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
           f"station={STATION}&data=tmpf&data=dwpf&data=sknt&data=p01i"
           f"&year1={start.year}&month1={start.month}&day1={start.day}"
           f"&year2={end.year}&month2={end.month}&day2={end.day}"
           "&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T&report_type=3")
    body = get(url, tries=6)
    if not body:
        sys.exit("Could not download KMLB observations.")
    obs = {k: {} for k in ("tmpf", "dwpf", "sknt", "p01i")}
    for row in csv.DictReader(io.StringIO(body)):
        try:
            t = datetime.strptime(row["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        except (KeyError, ValueError):
            continue
        t = (t + timedelta(minutes=30)).replace(minute=0)  # :53 routine ob -> top of the hour
        for k in obs:
            v = row.get(k, "M")
            if v in ("M", ""):
                continue
            obs[k][t] = 0.0 if v == "T" else float(v)
    log(f"observations: {len(obs['tmpf'])} hourly temperatures, {len(obs['p01i'])} precip reports")
    return obs


def predictand(obs, element, t):
    if element == "TMP":
        return obs["tmpf"].get(t)
    if element == "DPT":
        return obs["dwpf"].get(t)
    if element == "WSP":
        return obs["sknt"].get(t)
    if element == "XN":
        v = [obs["tmpf"].get(t - k * H) for k in range(13)]
        v = [x for x in v if x is not None]
        if len(v) < 11:
            return None
        return max(v) if t.hour == 0 else min(v)
    h = mos.PROB[element]
    v = [obs["p01i"].get(t - k * H) for k in range(h)]
    v = [x for x in v if x is not None]
    if len(v) < h - 1:
        return None
    return 1.0 if sum(v) >= 0.01 else 0.0


# ------------------------------------------------------------------ model archives
def fetch_previous_runs(om, start, end):
    """{lead_day: series} from the previous-runs archive (day 0 = the latest run's first hours)."""
    names = []
    for key, var in mos.OM_VARS.items():
        names.append(var)
        names += [f"{var}_previous_day{n}" for n in range(1, mos.MAX_LEAD_DAY + 1)]
    out = {n: defaultdict(dict) for n in range(mos.MAX_LEAD_DAY + 1)}
    t = start
    while t < end:
        t2 = min(t + timedelta(days=90), end)
        url = (f"{PREV}?latitude={LAT}&longitude={LON}&models={om}&hourly={','.join(names)}"
               f"&start_date={t:%Y-%m-%d}&end_date={t2:%Y-%m-%d}{UNITS}")
        body = get(url, pause=2)
        if body:
            j = json.loads(body)
            for n in range(mos.MAX_LEAD_DAY + 1):
                part = parse_hourly(j, "" if n == 0 else f"_previous_day{n}")
                for k, d in part.items():
                    out[n][k].update(d)
        log(f"  {om} previous runs {t:%Y-%m-%d}..{t2:%Y-%m-%d}: {'ok' if body else 'failed'}")
        t = t2 + timedelta(days=1)
    return out


def fetch_single_runs(om, start, end):
    """[(run time, series)] for each available 00Z/12Z run."""
    runs = []
    t = start
    names = ",".join(mos.OM_VARS.values())
    while t <= end:
        body = get(f"{SINGLE}?latitude={LAT}&longitude={LON}&models={om}&run={t:%Y-%m-%dT%H:%M}"
                   f"&forecast_days=9&hourly={names}{UNITS}", tries=2, pause=0.3)
        if body:
            j = json.loads(body)
            if "hourly" in j:
                runs.append((t, parse_hourly(j)))
        t += timedelta(hours=12)
    log(f"  {om} single runs {start:%Y-%m-%d}..{end:%Y-%m-%d}: {len(runs)} runs")
    return runs


def samples(model, cfg, obs, now):
    """{(element, day, hour): [(valid time, predictors, observed)]}"""
    out = defaultdict(list)
    single_start = None
    if cfg["single"]:
        # earliest individual run still kept (step back until runs stop)
        t = (now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
        while get(f"{SINGLE}?latitude={LAT}&longitude={LON}&models={cfg['om']}&run={t - timedelta(days=15):%Y-%m-%dT%H:%M}"
                  f"&forecast_days=1&hourly=temperature_2m", tries=1, pause=0.3):
            t -= timedelta(days=15)
        single_start = t
        log(f"  {model}: individual runs kept from about {single_start:%Y-%m-%d}")
    prev_end = (single_start or now) - timedelta(days=1)
    prev = fetch_previous_runs(cfg["om"], cfg["prev_start"], prev_end)
    for n, series in prev.items():
        times = sorted(series.get("t2m", {}))
        for t in times:
            if t >= prev_end:
                continue
            for el, hours in mos.ELEMENTS.items():
                if t.hour not in hours:
                    continue
                x = mos.predictors(series, t, el)
                y = predictand(obs, el, t) if x else None
                if y is not None:
                    out[(el, n, t.hour)].append((t, x, y))
    if single_start:
        for run, series in fetch_single_runs(cfg["om"], single_start, now - timedelta(days=1)):
            for t in sorted(series.get("t2m", {})):
                lead = (t - run).total_seconds() / 3600
                if lead < 6 or t > now:
                    continue
                n = mos.lead_day(lead)
                for el, hours in mos.ELEMENTS.items():
                    if t.hour not in hours:
                        continue
                    x = mos.predictors(series, t, el)
                    y = predictand(obs, el, t) if x else None
                    if y is not None:
                        out[(el, n, t.hour)].append((t, x, y))
    return out


# ------------------------------------------------------------------ regression
def forward_select(X, y, names):
    """MOS-style forward selection: add the predictor that most raises R^2, up to MAX_TERMS."""
    n = len(y)
    chosen, best_r2 = [], 0.0
    sst = float(((y - y.mean()) ** 2).sum()) or 1e-9
    while len(chosen) < MAX_TERMS:
        best = None
        for j, name in enumerate(names):
            if j in chosen:
                continue
            cols = chosen + [j]
            A = np.column_stack([X[:, cols], np.ones(n)])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = 1 - float(((A @ coef - y) ** 2).sum()) / sst
            if best is None or r2 > best[0]:
                best = (r2, j, coef)
        if best is None or best[0] - best_r2 < MIN_GAIN:
            break
        best_r2 = best[0]
        chosen.append(best[1])
    A = np.column_stack([X[:, chosen], np.ones(n)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"t": [names[j] for j in chosen], "c": [round(float(c), 6) for c in coef[:-1]],
            "b": round(float(coef[-1]), 6), "n": n, "r2": round(best_r2, 4),
            "r": [[round(float(X[:, j].min()), 4), round(float(X[:, j].max()), 4)] for j in chosen]}


def fit(rows):
    names = [k for k in rows[0][1] if k not in mos.NOT_OFFERED]
    X = np.array([[x[k] for k in names] for _, x, _ in rows], dtype=float)
    y = np.array([o for _, _, o in rows], dtype=float)
    return forward_select(X, y, names)


def train(data, cutoff=None):
    eqs = defaultdict(dict)
    for (el, n, hour), rows in data.items():
        use = [r for r in rows if cutoff is None or r[0] < cutoff]
        if len(use) >= MIN_N:
            eqs[el][f"{n}|{hour:02d}"] = fit(use)
    return eqs


def score(data, eqs, cutoff):
    """Holdout scores: MAE for continuous elements, Brier score for probabilities, raw vs MOS."""
    res = {}
    for el in mos.ELEMENTS:
        raw_e, mos_e, clim, byday = [], [], [], defaultdict(lambda: [[], []])
        train_events = [o for (e, _, _), rows in data.items() if e == el for t, _, o in rows if t < cutoff]
        base = sum(train_events) / len(train_events) if train_events and el in mos.PROB else None
        for (e, n, hour), rows in data.items():
            if e != el:
                continue
            eq = eqs.get(el, {}).get(f"{n}|{hour:02d}")
            if not eq:
                continue
            for t, x, o in rows:
                if t < cutoff:
                    continue
                m = mos.apply(eq, x)
                if el in mos.PROB:
                    m = min(mos.PROB_MAX, max(0.0, m))
                    r_err, m_err = (x["RAW"] - o) ** 2, (m - o) ** 2
                    clim.append((base - o) ** 2)
                else:
                    if abs(m - x["RAW"]) > mos.GUARD[el]:
                        m = x["RAW"]  # same fallback as the live forecasts
                    r_err, m_err = abs(x["RAW"] - o), abs(m - o)
                raw_e.append(r_err)
                mos_e.append(m_err)
                byday[n][0].append(r_err)
                byday[n][1].append(m_err)
        if not raw_e:
            continue
        mean = lambda v: round(sum(v) / len(v), 3)
        res[el] = {"n": len(raw_e), "raw": mean(raw_e), "mos": mean(mos_e),
                   "by_day": {str(n): {"n": len(v[0]), "raw": mean(v[0]), "mos": mean(v[1])} for n, v in sorted(byday.items())}}
        if clim:
            res[el]["climo"] = mean(clim)
    return res


def main():
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    cutoff = now - timedelta(days=HOLDOUT_DAYS)
    obs = fetch_obs(min(c["prev_start"] for c in MODELS.values()) - timedelta(days=2), now)
    coefs = {"trained": now.strftime("%Y-%m-%dT%H:%MZ"), "station": "KMLB",
             "lead_day": "round((lead hours - 4) / 24), 0-7", "models": {}}
    verif = {"trained": coefs["trained"], "holdout": [cutoff.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")],
             "units": {"MAE": "F for TMP, DPT, XN; kt for WSP", "Brier": "P06, P12, P24 (0 = perfect)"}, "models": {}}
    for model, cfg in MODELS.items():
        log(f"{model}: building samples")
        data = samples(model, cfg, obs, now)
        if not data:
            log(f"warning: no {model} samples")
            continue
        span = sorted(t for rows in data.values() for t, _, _ in rows)
        log(f"{model}: {sum(len(v) for v in data.values())} samples, {span[0]:%Y-%m-%d} to {span[-1]:%Y-%m-%d}")
        verif["models"][model] = {"samples": sum(len(v) for v in data.values()),
                                  "from": span[0].strftime("%Y-%m-%d"),
                                  "scores": score(data, train(data, cutoff), cutoff)}
        coefs["models"][model] = train(data)
        for el, s in verif["models"][model]["scores"].items():
            log(f"  {el}: n={s['n']} raw={s['raw']} mos={s['mos']}" + (f" climo={s['climo']}" if "climo" in s else ""))
        for el, tab in coefs["models"][model].items():
            terms = defaultdict(int)
            for eq in tab.values():
                for t in eq["t"]:
                    terms[t] += 1
            log(f"  {el}: {len(tab)} equations; most used: {sorted(terms.items(), key=lambda kv: -kv[1])[:5]}")
    os.makedirs("mos", exist_ok=True)
    with open("mos/coefficients.json", "w") as f:
        json.dump(coefs, f, separators=(",", ":"))
    with open("mos/verification.json", "w") as f:
        json.dump(verif, f, indent=1)
    log("wrote mos/coefficients.json and mos/verification.json")


if __name__ == "__main__":
    main()
