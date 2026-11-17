"""
mos.py - shared pieces of the KMLB-trained MOS for ECMWF and UKMET.

The same predictor code is used to train the equations (train_mos.py) and to apply
them to live runs (update.py), so the two always agree.

Model series are dicts of {variable: {datetime: value}} with variables
  t2m  2 m temperature, F        td   2 m dewpoint, F       rh  2 m relative humidity, %
  ws   10 m wind speed, kt       wd   10 m wind direction   cc  total cloud cover, %
  msl  mean sea-level pressure   pr   precipitation in the hour ending at the time, in
"""
import math
from datetime import timedelta

H = timedelta(hours=1)

# Open-Meteo variable for each series key
OM_VARS = {"t2m": "temperature_2m", "td": "dew_point_2m", "rh": "relative_humidity_2m",
           "ws": "wind_speed_10m", "wd": "wind_direction_10m", "cc": "cloud_cover",
           "msl": "pressure_msl", "pr": "precipitation"}

ELEMENTS = {
    # element: valid hours (UTC)
    "TMP": (0, 3, 6, 9, 12, 15, 18, 21),
    "DPT": (0, 3, 6, 9, 12, 15, 18, 21),
    "WSP": (0, 3, 6, 9, 12, 15, 18, 21),
    "XN": (0, 12),            # 00Z: max over 12Z-00Z, 12Z: min over 00Z-12Z
    "P06": (0, 6, 12, 18),    # chance of >= .01 in over the 6 h ending then
    "P12": (0, 12),
    "P24": (0,),
}
PROB = {"P06": 6, "P12": 12, "P24": 24}
MAX_LEAD_DAY = 7


def lead_day(lead_hours):
    """Lead time in hours to the archive's lead-day bin (day N holds leads of about 24N to 24N+6 h)."""
    return max(0, min(MAX_LEAD_DAY, round((lead_hours - 4) / 24)))


def season(t):
    a = 2 * math.pi * (t.timetuple().tm_yday - 1) / 365.25
    return math.sin(a), math.cos(a)


def _vals(series, var, t, hours):
    out = []
    s = series.get(var, {})
    for k in range(hours):
        v = s.get(t - k * H)
        if v is None:
            return None
        out.append(v)
    return out


def _at(series, var, t):
    return series.get(var, {}).get(t)


def _rh(series, t):
    v = _at(series, "rh", t)
    if v is None:
        T, D = _at(series, "t2m", t), _at(series, "td", t)
        if T is None or D is None:
            return None
        tc, dc = (T - 32) / 1.8, (D - 32) / 1.8
        v = 100 * math.exp(17.625 * dc / (243.04 + dc)) / math.exp(17.625 * tc / (243.04 + tc))
    return v


def _uv(ws, wd):
    r = math.radians(wd)
    return -ws * math.sin(r), -ws * math.cos(r)


def predictors(series, t, element):
    """Predictor values for one valid time, or None if anything needed is missing.
    The first entry is the raw model value of the element itself (used as the 'raw' forecast)."""
    s1, c1 = season(t)
    if element in ("TMP", "DPT", "WSP"):
        T, D, ws, wd = (_at(series, k, t) for k in ("t2m", "td", "ws", "wd"))
        cc, msl, rh = _at(series, "cc", t), _at(series, "msl", t), _rh(series, t)
        pr6 = _vals(series, "pr", t, 6)
        Tm3, Tp3 = _at(series, "t2m", t - 3 * H), _at(series, "t2m", t + 3 * H)
        if None in (T, D, ws, wd, cc, msl, rh, Tm3, Tp3) or pr6 is None:
            return None
        u, v = _uv(ws, wd)
        raw = {"TMP": T, "DPT": D, "WSP": ws}[element]
        p6 = max(sum(pr6), 0.0)
        return {"RAW": raw, "T": T, "TD": D, "DEP": T - D, "RH": rh, "WS": ws, "U": u, "V": v,
                "CC": cc, "MSL": msl, "P6": p6, "SP6": math.sqrt(min(p6, 2.0)), "T-3": Tm3, "T+3": Tp3,
                "S1": s1, "C1": c1}
    if element == "XN":
        T = _vals(series, "t2m", t, 13)
        D, ws, wd, cc = (_vals(series, k, t, 13) for k in ("td", "ws", "wd", "cc"))
        pr = _vals(series, "pr", t, 12)
        if None in (T, D, ws, wd, cc, pr):
            return None
        ext = max(T) if t.hour == 0 else min(T)
        us, vs = zip(*(_uv(a, b) for a, b in zip(ws, wd)))
        return {"RAW": ext, "TX": ext, "TMEAN": sum(T) / 13, "T": T[0], "TD": sum(D) / 13,
                "WS": sum(ws) / 13, "U": sum(us) / 13, "V": sum(vs) / 13, "CC": sum(cc) / 13,
                "P12": sum(pr), "S1": s1, "C1": c1}
    if element in PROB:
        h = PROB[element]
        pr = _vals(series, "pr", t, h)
        T, D, ws, wd, cc, msl = (_vals(series, k, t, h) for k in ("t2m", "td", "ws", "wd", "cc", "msl"))
        if None in (pr, T, D, ws, wd, cc, msl):
            return None
        ps = sum(pr)
        us, vs = zip(*(_uv(a, b) for a, b in zip(ws, wd)))
        dep = sum(a - b for a, b in zip(T, D)) / h
        return {"RAW": 1.0 if ps >= 0.01 else 0.0, "PS": ps, "SQRTP": math.sqrt(max(ps, 0)), "PMAX": max(pr),
                "B01": float(ps >= 0.01), "B05": float(ps >= 0.05), "B10": float(ps >= 0.10),
                "B25": float(ps >= 0.25), "B50": float(ps >= 0.50), "DEP": dep, "CC": sum(cc) / h,
                "U": sum(us) / h, "V": sum(vs) / h, "MSL": sum(msl) / h, "S1": s1, "C1": c1}
    raise ValueError(element)


# Predictors offered to forward selection (P6 stays computable for older equations, but its
# leverage on heavy-rain outliers is too large, so new equations use SP6 = sqrt of 6-h rain).
NOT_OFFERED = {"RAW", "P6"}
GUARD = {"TMP": 15, "DPT": 15, "XN": 15, "WSP": 12}  # keep raw if MOS strays further than this
PROB_MAX = 0.98


def apply(eq, x):
    """Evaluate one equation {'t': terms, 'c': coefs, 'b': intercept, 'r': [[lo, hi], ...]} on predictor
    dict x. Each predictor is held to the range it had in training, as MOS does."""
    rng = eq.get("r") or [None] * len(eq["t"])
    total = eq["b"]
    for t, c, r in zip(eq["t"], eq["c"], rng):
        v = x[t]
        if r:
            v = min(max(v, r[0]), r[1])
        total += c * v
    return total


def find_eq(coefs, model, element, day, hour):
    """Equation for model/element at lead day and valid hour, falling back to the nearest lead day."""
    table = coefs.get("models", {}).get(model, {}).get(element, {})
    for d in sorted(range(MAX_LEAD_DAY + 1), key=lambda k: (abs(k - day), k)):
        eq = table.get(f"{d}|{hour:02d}")
        if eq:
            return eq
    return None


def forecast(coefs, model, element, series, t, lead_hours):
    """MOS forecast for one element at valid time t, or None (no equation or missing predictors)."""
    if t.hour not in ELEMENTS[element]:
        return None
    eq = find_eq(coefs, model, element, lead_day(lead_hours), t.hour)
    x = predictors(series, t, element) if eq else None
    if x is None:
        return None
    v = apply(eq, x)
    if element in PROB:
        return max(0, min(round(100 * PROB_MAX), round(100 * v)))
    if abs(v - x["RAW"]) > GUARD[element]:
        return None  # implausible; the caller keeps the raw value
    if element == "WSP":
        return max(0, round(v))
    return round(v)
