"""
Builds the data files behind travel/index.html from your Google Timeline export
and the hand-kept lists of places visited.

    python pipeline/build_travel.py

Inputs:   .private/Timeline.json (git-ignored raw export)
          pipeline/countries.txt, pipeline/states_provinces.txt, pipeline/national parks.txt
Outputs:  travel/data/travel.js, travel/data/geo.js (public: ~1 km cells, no visit frequencies)
Boundary + gazetteer data is downloaded once into pipeline/geo_cache/ (delete a file to refresh it).

Needs Python 3.9+ with numpy and matplotlib.
"""
import collections
import difflib
import io
import json
import os
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone

import numpy as np
from matplotlib.path import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, 'pipeline')
CACHE = os.path.join(HERE, 'geo_cache')
OUT = os.path.join(ROOT, 'travel', 'data')
TIMELINE = os.path.join(ROOT, '.private', 'Timeline.json')   # raw export: never published

# ------------------------------------------------------------------ settings
# Every published location is snapped to this grid (degrees; 0.01 is ~1 km) so the public
# maps and data never pinpoint a home, workplace or other regular stop.
PRIVACY_GRID = 0.01
# Park units that aren't typed "National Parks" in the NPS boundary service but count here.
EXTRA_PARKS = ['New River Gorge National Park and Preserve', 'Chiricahua National Monument']
# Spellings in the lists that fuzzy matching can't be trusted with.
COUNTRY_ALIASES = {
    'usa': 'United States of America', 'us': 'United States of America', 'america': 'United States of America',
    'unitedstates': 'United States of America', 'unitesstates': 'United States of America',
    'uk': 'United Kingdom', 'england': 'United Kingdom', 'scotland': 'United Kingdom', 'wales': 'United Kingdom',
    'greatbritain': 'United Kingdom', 'northernireland': 'United Kingdom', 'czechrepublic': 'Czechia',
    'holland': 'Netherlands', 'southkorea': 'South Korea', 'korea': 'South Korea', 'vaticancity': 'Vatican',
}

os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ cached downloads
NE = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/'
NPS = ('https://services1.arcgis.com/fBc8EJBxQRMcHlei/arcgis/rest/services/'
       'NPS_Land_Resources_Division_Boundary_and_Tract_Data_Service/FeatureServer/2/query?')


def cached(name, url):
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        log('downloading', name)
        req = urllib.request.Request(url, headers={'User-Agent': 'travel-map-builder'})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        with open(path, 'wb') as f:
            f.write(data)
    return path


def nps_url():
    names = ','.join("'" + n.replace("'", "''") + "'" for n in EXTRA_PARKS)
    return NPS + urllib.parse.urlencode({
        'where': f"UNIT_TYPE='National Parks' OR UNIT_NAME IN ({names})",
        'outFields': 'UNIT_NAME,UNIT_CODE,STATE,UNIT_TYPE', 'outSR': 4326,
        'geometryPrecision': 4, 'maxAllowableOffset': 0.003, 'f': 'geojson'})


ADMIN0 = cached('admin0_50.geojson', NE + 'ne_50m_admin_0_countries.geojson')
ADMIN1 = cached('admin1_50.geojson', NE + 'ne_50m_admin_1_states_provinces.geojson')
PARKS = cached('nps_parks.geojson', nps_url())
CITIES = cached('cities5000.zip', 'https://download.geonames.org/export/dump/cities5000.zip')
ADMCODES = cached('admin1CodesASCII.txt', 'https://download.geonames.org/export/dump/admin1CodesASCII.txt')


# ------------------------------------------------------------------ geometry helpers
def haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def polygons(geom):
    if geom['type'] == 'Polygon':
        return [geom['coordinates']]
    return geom['coordinates'] if geom['type'] == 'MultiPolygon' else []


class Region:
    """Polygon(s) with holes, tested even-odd per polygon."""

    def __init__(self, name, geom, props=None):
        self.name, self.props, self.polys = name, props or {}, []
        for poly in polygons(geom):
            verts, codes = [], []
            for ring in poly:
                r = np.asarray(ring, float)[:, :2]
                c = np.full(len(r), Path.LINETO)
                c[0] = Path.MOVETO
                verts.append(r)
                codes.append(c)
            v = np.concatenate(verts)
            self.polys.append((Path(v, np.concatenate(codes)), v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()))
        v = np.concatenate([p[0].vertices for p in self.polys])
        self.bbox = (v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max())

    def contains(self, lon, lat, buffer=0.0):
        lon, lat = np.atleast_1d(lon), np.atleast_1d(lat)
        out = np.zeros(len(lon), bool)
        x0, x1, y0, y1 = self.bbox
        m = (lon >= x0 - buffer) & (lon <= x1 + buffer) & (lat >= y0 - buffer) & (lat <= y1 + buffer)
        if not m.any():
            return out
        idx = np.nonzero(m)[0]
        xy = np.column_stack([lon[idx], lat[idx]])
        hit = np.zeros(len(idx), bool)
        for path, a, b, c, d in self.polys:
            mm = (xy[:, 0] >= a - buffer) & (xy[:, 0] <= b + buffer) & (xy[:, 1] >= c - buffer) & (xy[:, 1] <= d + buffer)
            if mm.any():
                sub = np.nonzero(mm)[0]
                inside = path.contains_points(xy[sub])
                if buffer:  # radius sign depends on ring orientation, so try both
                    inside |= path.contains_points(xy[sub], radius=buffer) | path.contains_points(xy[sub], radius=-buffer)
                hit[sub] |= inside
        out[idx] = hit
        return out

    def label_point(self):
        """A point well inside the largest polygon (coarse pole of inaccessibility)."""
        pth, x0, x1, y0, y1 = max(self.polys, key=lambda p: (p[2] - p[1]) * (p[4] - p[3]))
        xs, ys = np.meshgrid(np.linspace(x0, x1, 40), np.linspace(y0, y1, 40))
        cand = np.column_stack([xs.ravel(), ys.ravel()])
        inside = cand[pth.contains_points(cand)]
        if not len(inside):
            return float(pth.vertices[:, 1].mean()), float(pth.vertices[:, 0].mean())
        v = pth.vertices
        best = max(inside, key=lambda p: np.min((v[:, 0] - p[0]) ** 2 + (v[:, 1] - p[1]) ** 2))
        return float(best[1]), float(best[0])


def douglas_peucker(pts, tol):
    if len(pts) < 4:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        a, b = pts[s], pts[e]
        seg, ab = pts[s + 1:e], pts[e] - pts[s]
        L = np.hypot(*ab)
        d = np.hypot(*(seg - a).T) if L == 0 else np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L
        i = int(np.argmax(d))
        if d[i] > tol:
            keep[s + 1 + i] = True
            stack += [(s, s + 1 + i), (s + 1 + i, e)]
    return pts[keep]


def simplify(geom, tol, nd):
    """Simplify and wind rings the way d3 expects (outer clockwise, holes counter-clockwise)."""
    out = []
    for poly in polygons(geom):
        rings = []
        for i, ring in enumerate(poly):
            r = np.round(douglas_peucker(np.asarray(ring, float)[:, :2], tol), nd)
            if len(r) < 4:
                if i == 0:
                    break
                continue
            area = np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1])
            if (area > 0) == (i == 0):   # positive shoelace area = counter-clockwise
                r = r[::-1]
            rings.append(r.tolist())
        if rings:
            out.append(rings)
    return {'type': 'MultiPolygon', 'coordinates': out}


def norm(s):
    s = unicodedata.normalize('NFKD', s or '').encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]', '', s.lower())


def match(name, table, what):
    """table: normalized key -> canonical value. Exact, then close fuzzy match, else None."""
    k = norm(name)
    if k in table:
        return table[k]
    close = difflib.get_close_matches(k, list(table), n=1, cutoff=0.8)
    if close:
        log(f'  note: {what} "{name}" matched to "{table[close[0]]}"')
        return table[close[0]]
    log(f'  WARNING: {what} "{name}" not recognised, skipped')
    return None


def iso(dnum):
    return None if dnum is None else (date(1970, 1, 1) + timedelta(days=int(dnum))).isoformat()


# ------------------------------------------------------------------ gazetteer
adm1_names = {}
for line in open(ADMCODES, encoding='utf-8'):
    p = line.rstrip('\n').split('\t')
    if len(p) >= 2:
        adm1_names[p[0]] = p[1]
rows = []
with zipfile.ZipFile(CITIES) as z:
    for line in io.TextIOWrapper(z.open('cities5000.txt'), encoding='utf-8'):
        p = line.split('\t')
        if p[7] != 'PPLX':   # skip "sections of a populated place" (neighbourhoods)
            rows.append((float(p[4]), float(p[5]), int(p[14] or 0), p[1], p[8], p[10], p[7]))
GZ = np.array([(r[0], r[1], r[2]) for r in rows])
# capitals and admin seats win ties against big unincorporated neighbourhoods ("New South Memphis")
SEAT_BONUS = np.array([{'PPLC': 0.5, 'PPLA': 0.3, 'PPLA2': 0.2}.get(r[6], 0.0) for r in rows])
GZN = [(r[3], r[4], r[5], r[6]) for r in rows]

admin0_feats = json.load(open(ADMIN0, encoding='utf-8'))['features']
COUNTRY_NAME = {}   # ISO2 -> display name
for f in admin0_feats:
    p = f['properties']
    COUNTRY_NAME[p['ISO_A2_EH']] = 'United States' if p['ADMIN'] == 'United States of America' else p['NAME_LONG']


def place_name(lat, lon, big=False):
    """big=True prefers a major city nearby (airports); otherwise the nearest town."""
    d = haversine(lat, lon, GZ[:, 0], GZ[:, 1])
    m = (d < 80) & (GZ[:, 2] >= 150000) if big else d < 40
    if not m.any():
        m = d < (60 if big else 250)
        if not m.any():
            m = d < 250
            if not m.any():
                return 'Unknown'
    score = np.log10(np.maximum(GZ[:, 2], 1)) - d / 50 + SEAT_BONUS if big else -d
    i = int(np.argmax(np.where(m, score, -1e9)))
    n, cc, a1, _ = GZN[i]
    if cc in ('US', 'CA'):
        a1n = adm1_names.get(f'{cc}.{a1}', '')
        return n if a1n in ('', 'Washington, D.C.', 'District of Columbia') or a1n == n else f'{n}, {a1n}'
    return f'{n}, {COUNTRY_NAME.get(cc, cc)}' if COUNTRY_NAME.get(cc, cc) != n else n


# ------------------------------------------------------------------ read the timeline
log('reading Timeline.json')
TL = json.load(open(TIMELINE, encoding='utf-8'))


def latlng(s):
    a, b = s.replace('°', '').split(',')
    return float(a), float(b)


def ts(s):
    return datetime.fromisoformat(s).timestamp()


# Only "visits" (places Google decided I stopped) are used. Path pings and raw GPS fixes are
# far denser but include in-flight fixes and drive-throughs, which aren't places I've been.
visits = []
for s in TL.get('semanticSegments', []):
    if 'visit' not in s:
        continue
    v, c = s['visit'], s['visit'].get('topCandidate', {})
    if 'latLng' in c.get('placeLocation', {}):
        la_, lo_ = latlng(c['placeLocation']['latLng'])
        visits.append((ts(s['startTime']), ts(s['endTime']), la_, lo_, c.get('semanticType'), v.get('hierarchyLevel', 0)))
del TL
visits.sort()
log(f'{len(visits):,} visits')


def local_day(t_, lon_):
    return int((t_ + round(lon_ / 15) * 3600) // 86400)   # local date, approximated by longitude


# one row per (stop, day it covers): a three-night stay counts as three days there
rows = []
for t0, t1, vla, vlo, st, lvl in visits:
    if lvl != 0:
        continue
    d0, d1 = local_day(t0, vlo), local_day(max(t0, t1 - 1), vlo)
    for dnum in range(d0, min(d1, d0 + 400) + 1):
        rows.append((dnum, vla, vlo))
R_ = np.array(rows)
gday, gla, glo = R_[:, 0].astype(int), R_[:, 1], R_[:, 2]
# parent places (hierarchy level 1, e.g. a whole national park) help only with "was I inside this park"
parent = np.array([(local_day(t0, vlo), vla, vlo) for t0, t1, vla, vlo, st, lvl in visits if lvl != 0] or [(0, 0.0, 0.0)])

# ------------------------------------------------------------------ homes over time
home_hours = collections.defaultdict(collections.Counter)
night_hours = collections.defaultdict(collections.Counter)
exact = {}
for t0, t1, vla, vlo, st, lvl in visits:
    if lvl != 0 or t1 <= t0:
        continue
    key = (round(vla, 2), round(vlo, 2))
    exact.setdefault(key, (vla, vlo))
    m = datetime.fromtimestamp(t0, timezone.utc).strftime('%Y-%m')
    if st in ('HOME', 'INFERRED_HOME'):
        home_hours[m][key] += (t1 - t0) / 3600
    off, tt = round(vlo / 15) * 3600, t0
    while tt < t1:
        nxt = min(t1, (tt // 3600 + 1) * 3600)
        if ((tt + off) % 86400) / 3600 < 6:
            night_hours[m][key] += (nxt - tt) / 3600
        tt = nxt
mhome = {}
for m in sorted(set(home_hours) | set(night_hours)):
    c = night_hours[m] if night_hours[m] else home_hours[m]   # where I actually slept beats Google's HOME tag
    if c:
        mhome[m] = c.most_common(1)[0][0]


def near(a, b, km=3):
    return haversine(a[0], a[1], b[0], b[1]) < km


ms, vals = sorted(mhome), [mhome[m] for m in sorted(mhome)]
smooth = []   # a one-month blip is a trip, not a move
for i, v in enumerate(vals):
    stable = (i + 1 < len(vals) and near(vals[i + 1], v)) or (i > 0 and near(vals[i - 1], v))
    smooth.append(v if stable or not smooth else smooth[-1])
periods = []
for m, v in zip(ms, smooth):
    if periods and near(periods[-1]['pos'], v):
        periods[-1]['end'] = m
    else:
        periods.append(dict(pos=exact[v], start=m, end=m))
for p in periods:
    p['name'] = place_name(*p['pos'])
    log(f"  home {p['start']} – {p['end']}: {p['name']}")
anchors = []
for p in periods:
    if not any(near(p['pos'], a, 0.5) for a in anchors):
        anchors.append(p['pos'])


def home_for(dnum):
    ym = iso(dnum)[:7]
    cur = periods[0]
    for p in periods:
        if p['start'] <= ym:
            cur = p
    return cur['pos']


# ------------------------------------------------------------------ regions + the user's lists
countries = [Region(f['properties']['ADMIN'], f['geometry'], f['properties']) for f in admin0_feats]
admin1_feats = [f for f in json.load(open(ADMIN1, encoding='utf-8'))['features']
                if f['properties']['admin'] in ('United States of America', 'Canada')]
for f in admin1_feats:
    f['properties']['name'] = unicodedata.normalize('NFKD', f['properties']['name']).encode('ascii', 'ignore').decode()
admin1 = [Region(f['properties']['name'], f['geometry'], f['properties']) for f in admin1_feats]

# first stop per ~1 km cell keeps the region tests fast
ck = np.round(gla / 0.01).astype(np.int64) * 100000 + np.round(glo / 0.01).astype(np.int64)
order = np.lexsort((gday, ck))
first = order[np.r_[True, ck[order][1:] != ck[order][:-1]]]
cla, clo, cday = gla[first], glo[first], gday[first]


def first_seen(regs, buffer):
    out = {}
    for r in regs:
        m = r.contains(clo, cla, buffer)
        if m.any():
            out[r.name] = int(cday[m].min())
    return out


tl_country = first_seen(countries, 0.03)
tl_admin1 = first_seen(admin1, 0.01)


def read_list(fname):
    items = []
    for line in open(os.path.join(HERE, fname), encoding='utf-8'):
        m = re.match(r'^\s*\d+[.)]\s*(.+?)\s*$', line)
        if m:
            items.append(m.group(1))
    return items


ctable = {}
for f in admin0_feats:
    p = f['properties']
    for k in ('ADMIN', 'NAME', 'NAME_LONG', 'FORMAL_EN', 'NAME_ALT', 'BRK_NAME', 'ISO_A3'):
        if p.get(k):
            ctable.setdefault(norm(p[k]), p['ADMIN'])
ctable.update({k: v for k, v in COUNTRY_ALIASES.items()})
cprops = {f['properties']['ADMIN']: f['properties'] for f in admin0_feats}
visited_countries = list(dict.fromkeys(filter(None, (match(c, ctable, 'country') for c in read_list('countries.txt')))))

atable = {}
for f in admin1_feats:
    p = f['properties']
    atable[norm(p['name'])] = p['name']
    if p.get('postal'):
        atable.setdefault(norm(p['postal']), p['name'])
atable.update({'washingtondc': 'District of Columbia', 'dc': 'District of Columbia'})
visited_admin1 = list(dict.fromkeys(filter(None, (match(s, atable, 'state/province') for s in read_list('states_provinces.txt')))))
admin1_country = {f['properties']['name']: 'US' if f['properties']['admin'].startswith('United') else 'CA' for f in admin1_feats}


def short_park(n):
    if 'American Samoa' in n:
        return 'American Samoa'
    for suffix in ('National Park and Preserve', 'National Park', 'National Monument and Preserve', 'National Monument'):
        n = n.replace(suffix, '')
    return n.strip()


park_feats = {}
for f in json.load(open(PARKS, encoding='utf-8'))['features']:
    park_feats.setdefault(short_park(f['properties']['UNIT_NAME']), f)
ptable = {norm(k): k for k in park_feats}
visited_parks = list(dict.fromkeys(filter(None, (match(p, ptable, 'park') for p in read_list('national parks.txt')))))
log(f'lists: {len(visited_countries)} countries, {len(visited_admin1)} states/provinces, '
    f'{len(visited_parks)} of {len(park_feats)} national parks')

parks = []
for name in sorted(park_feats):
    f = park_feats[name]
    r = Region(name, f['geometry'])
    m, mp = r.contains(glo, gla, 0.01), r.contains(parent[:, 2], parent[:, 1], 0.01)
    days = list(gday[m]) + list(parent[mp, 0].astype(int))
    lat_, lon_ = r.label_point()
    parks.append(dict(name=name, state=re.split(r'[ ]', f['properties']['STATE'] or '')[0],
                      lat=round(lat_, 4), lon=round(lon_, 4), visited=name in visited_parks,
                      first=iso(min(days)) if days else None))

# ------------------------------------------------------------------ dots / heat cells, privacy snapped
dla = np.round(gla / PRIVACY_GRID) * PRIVACY_GRID
dlo = np.round(glo / PRIVACY_GRID) * PRIVACY_GRID
RES = 0.001   # encoding unit for the published integers; cells themselves are PRIVACY_GRID wide
key = np.round(dla / RES).astype(np.int64) * 1000000 + (np.round(dlo / RES).astype(np.int64) + 500000)
# Each cell is published only as "somewhere I've been, first in year Y". How often a cell was
# visited is deliberately dropped: frequency is what would single out a home or workplace.
order = np.lexsort((gday, key))
key_s, day_s = key[order], gday[order]
starts = np.r_[True, key_s[1:] != key_s[:-1]]
cells, first_day = key_s[starts], day_s[starts]
year0 = date.fromisoformat(iso(gday.min())).year
dots = []
for k, fd in zip(cells, first_day):
    dots += [int(k // 1000000), int(k % 1000000 - 500000), date.fromisoformat(iso(fd)).year - year0]
log(f'{len(cells):,} map cells (no visit counts published)')

# ------------------------------------------------------------------ daily distance from home
regions_for_label = admin1 + [c for c in countries if c.name not in ('United States of America', 'Canada')]
label_cache = {}


def where(lat_, lon_):
    k = (round(lat_, 1), round(lon_, 1))
    if k not in label_cache:
        label_cache[k] = ''
        for buf in (0.0, 0.15):
            hit = next((r for r in regions_for_label if r.contains(lon_, lat_, buf)[0]), None)
            if hit:
                n = hit.name
                label_cache[k] = COUNTRY_NAME.get(hit.props.get('ISO_A2_EH'), n) if 'ISO_A2_EH' in hit.props else n
                break
    return label_cache[k]


daily = []
order = np.argsort(gday, kind='stable')
bounds = np.r_[0, np.nonzero(np.diff(gday[order]))[0] + 1, len(order)]
for s, e in zip(bounds[:-1], bounds[1:]):
    ii = order[s:e]
    dnum = int(gday[ii[0]])
    h = home_for(dnum)
    dist = haversine(gla[ii], glo[ii], h[0], h[1])
    j = int(np.argmax(dist))
    daily.append([dnum, round(float(dist[j]), 1), where(gla[ii][j], glo[ii][j]) if dist[j] >= 50 else ''])

# ------------------------------------------------------------------ extremes
ext = {}
for k, i in (('north', np.argmax(gla)), ('south', np.argmin(gla)), ('east', np.argmax(glo)), ('west', np.argmin(glo))):
    ext[k] = dict(lat=round(float(gla[i]), 2), lon=round(float(glo[i]), 2), day=iso(gday[i]), name=place_name(gla[i], glo[i], big=True))
far = haversine(gla, glo, *periods[-1]['pos'])
i = int(np.argmax(far))
ext['farthest'] = dict(lat=round(float(gla[i]), 2), lon=round(float(glo[i]), 2), day=iso(gday[i]),
                       km=round(float(far[i])), name=place_name(gla[i], glo[i], big=True))

# ------------------------------------------------------------------ report list vs timeline differences
in_tl_not_list = sorted(set(tl_admin1) - set(visited_admin1)) + sorted(set(tl_country) - set(visited_countries))
in_tl_not_list += [p['name'] + ' National Park' for p in parks if p['first'] and not p['visited']]
if in_tl_not_list:
    log('in the timeline but not in your lists (not shown as visited):',
        ', '.join(n if n.endswith('Park') else f'{n} ({iso(tl_admin1.get(n, tl_country.get(n)))})' for n in in_tl_not_list))

# ------------------------------------------------------------------ write
def cname(admin):
    return COUNTRY_NAME.get(cprops[admin]['ISO_A2_EH'], admin)


world = []
for f in admin0_feats:
    p = f['properties']
    if p['ADMIN'] == 'Antarctica':
        continue
    world.append({'type': 'Feature', 'properties': {'name': cname(p['ADMIN']), 'visited': p['ADMIN'] in visited_countries,
                                                    'sub': p['ADMIN'] in ('United States of America', 'Canada')},
                  'geometry': simplify(f['geometry'], 0.01, 3)})
states = [{'type': 'Feature', 'properties': {'name': f['properties']['name'], 'country': admin1_country[f['properties']['name']],
                                              'visited': f['properties']['name'] in visited_admin1},
           'geometry': simplify(f['geometry'], 0.01, 3)} for f in admin1_feats]
park_shapes = [{'type': 'Feature', 'properties': {'name': p['name']},
                'geometry': simplify(park_feats[p['name']]['geometry'], 0.002, 4)} for p in parks]

geo = dict(world={'type': 'FeatureCollection', 'features': world},
           admin1={'type': 'FeatureCollection', 'features': states},
           parks={'type': 'FeatureCollection', 'features': park_shapes})
travel = dict(
    generated=date.today().isoformat(), span=[iso(gday.min()), iso(gday.max())], year0=year0,
    dots=dots,
    stops=sum(1 for v in visits if v[5] == 0),
    # home positions rounded to ~10 km: enough to frame a "Home" view, not to find a house
    homes=[dict(start=p['start'], end=p['end'], name=p['name'], lat=round(p['pos'][0], 1), lon=round(p['pos'][1], 1)) for p in periods],
    daily=daily, extremes=ext, parks=parks,
    countries=[dict(name=cname(a), first=iso(tl_country.get(a))) for a in visited_countries],
    admin1=[dict(name=n, country=admin1_country[n], first=iso(tl_admin1.get(n))) for n in visited_admin1],
)
for name, obj, var in (('geo.js', geo, 'GEO'), ('travel.js', travel, 'TRAVEL')):
    with open(os.path.join(OUT, name), 'w', encoding='utf-8') as fh:
        fh.write(f'window.{var}=' + json.dumps(obj, separators=(',', ':'), ensure_ascii=False) + ';\n')
    log(f'wrote travel/data/{name} ({os.path.getsize(os.path.join(OUT, name)) // 1024:,} KB)')
