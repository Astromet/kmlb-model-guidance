# KMLB model guidance

GFS MOS, the National Blend of Models and ECMWF for Melbourne, FL (KMLB), side by side in MOS layout, with charts of wind speed, 12-hour rain chances and how each model changed from its run 12 hours earlier.

The site is rebuilt every 6 hours by GitHub Actions (`.github/workflows/update.yml`): `update.py` fetches the latest bulletins and model data (skipping any model whose newest run is over 36 hours old), writes `data/*.json`, and the page is deployed to GitHub Pages. Each run also saves its data files to the `data` branch. Run it any time from the **Actions** tab with **Run workflow**.

## Files

| File | What it does |
| --- | --- |
| `index.html` | The dashboard page. Reads `data/models.json`, `data/previous.json` and `data/meta.json`. |
| `update.py` | Fetches the data and writes the JSON files. Standard library only. |
| `data/` | A starting snapshot; the deployed site always uses freshly fetched data. |

Run locally:

```sh
python3 update.py            # writes data/*.json
python3 -m http.server       # then open http://localhost:8000
```

For another station: `python3 update.py --station KTPA --lat 27.96 --lon -82.54`, and change the same arguments in the workflow.

## Sources and conversions

- **GFS MOS (MAV)** and **GFS extended MOS (MEX)**, shown as its own 12-hourly table out to day 8 with climatology: [NWS MDL MOS bulletins](https://www.weather.gov/mdl/mos_getbull). NAM MOS was retired in November 2026 and has been removed.
- **NBM v5 NBS**: [NBM text bulletins](https://www.weather.gov/mdl/nbm_text). Sky cover, QPF, ceiling and visibility are converted to MOS categories.
- **ECMWF**: IFS 0.25° deterministic run and 51-member ENS via [Open-Meteo](https://open-meteo.com/) (CC BY 4.0). This is raw model output, not MOS. P06/P12 are the share of ensemble members with at least 0.05 in; CAP is max CAPE in hundreds of J/kg.
- **NWS forecast vs NBM** subtracts the latest NBM from the NWS official forecast grid (NDFD, via [api.weather.gov](https://www.weather.gov/documentation/services-web-api)) at KMLB for wind, gusts and 12-hour rain chances. NWS publishes hourly chances, so its 12-hour value is the highest hourly chance in the period.
- **Change charts** compare each model's latest run with its run 12 hours earlier. ECMWF rain chances are left out because earlier ensemble runs aren't kept.
