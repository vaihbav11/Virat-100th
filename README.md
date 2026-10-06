Live Link - https://virat-100th-avyqkgjcujtl3mvavjf8fz.streamlit.app/
# Data Directory

## Purpose

This directory holds the innings data used to estimate Virat Kohli's historical
century frequency per innings.  The application works **without** any data file —
it falls back to manually configurable probability assumptions displayed clearly
in the UI.

---

## Adding Real Historical Data

1. Copy `innings_template.csv` and rename it **`innings_data.csv`**.
2. Populate each row with one international innings.
3. Restart the Streamlit app — it will automatically detect and load the file.

### Column definitions

| Column    | Type    | Description                                                  |
|-----------|---------|--------------------------------------------------------------|
| `date`    | string  | Match date in `YYYY-MM-DD` format                            |
| `format`  | string  | One of: `Test`, `ODI`, `T20I` — **IPL and domestic are excluded** |
| `opponent`| string  | Opposing team name                                           |
| `runs`    | integer | Runs scored in the innings                                   |
| `not_out` | integer | `1` if not out, `0` if out                                   |
| `century` | integer | `1` if Kohli scored ≥100 runs, `0` otherwise                 |

### Important notes

- **Do not include IPL, Ranji Trophy, or other domestic innings.** The predictor
  counts only international centuries (Test + ODI + T20I).
- Only rows with `format` set to `Test`, `ODI`, or `T20I` are used; other values
  are silently ignored.
- The `century` column must be set correctly — the model does not infer centuries
  from the `runs` column automatically.
- This repository does **not** ship pre-populated innings data.  You must source
  verified data from a legitimate provider such as ESPNcricinfo or Cricsheet.

### Sources of verified historical data

- [ESPNcricinfo Statsguru](https://stats.espncricinfo.com) — official career stats
- [Cricsheet](https://cricsheet.org) — ball-by-ball CSV files under open licence

---

## Template

`innings_template.csv` contains the header row and two example rows.  Delete the
example rows and replace with real data before renaming the file.
