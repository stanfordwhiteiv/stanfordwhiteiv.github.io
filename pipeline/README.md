# Travel page pipeline

Rebuilds the public data behind `travel/index.html` from a private Google Timeline export.

```
python pipeline/build_travel.py
```

Needs Python 3.9+ with `numpy` and `matplotlib`. It takes about 10 seconds.

## Files

| Path | In git? | What it is |
| --- | --- | --- |
| `.private/Timeline.json` | **No** (ignored) | The raw export. Exact coordinates and timestamps; never publish it. |
| `pipeline/geo_cache/` | No (ignored) | Boundary and gazetteer downloads (Natural Earth, NPS, GeoNames). |
| `pipeline/*.txt` | Yes | Hand-kept lists of countries, states/provinces and national parks visited. |
| `travel/data/travel.js`, `geo.js` | Yes | The scrubbed, public output the page reads. |

## Updating

1. Export a fresh timeline (Google Maps app → Settings → Location → Timeline → Export) and save it as
   `.private/Timeline.json`, replacing the old one.
2. Add any new places to `countries.txt`, `states_provinces.txt` or `national parks.txt`. Keep the
   `1. Name` numbered format. Spelling is matched loosely; the build prints how each fuzzy match resolved.
3. Run the script, check the page, then commit `travel/data/`.

The build also reports any state, country or park where you logged a stop that isn't in your lists, so you can
decide whether to add it.

## What it does

- Uses only the timeline's visits: the places it decided you stopped. Path pings and raw GPS fixes are
  skipped because they include in-flight fixes and drive-throughs. A multi-day stay counts for each day.
- Works out home month by month from where you slept, so trips don't count as moves.
- Delete a file in `pipeline/geo_cache/` to re-download it. To count another unit as a national park, add its
  NPS unit name to `EXTRA_PARKS` at the top of the script.

## Privacy

- Every published location (map dots and the compass extremes) is snapped to `PRIVACY_GRID`
  (default 0.01°, about 1 km).
- Map cells carry only the year they were first visited. How many days a cell was visited is never
  written out, because frequency is what singles out a home, workplace or other regular stop.
- Home bases are exported as a city name and a position rounded to 0.1° (about 10 km).
- The day-by-day calendar stores only the distance from home and the state or country, not coordinates.
- The page itself also caps map zoom at roughly state level.
