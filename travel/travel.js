// Travel page: maps and charts drawn from data/travel.js + data/geo.js (built by pipeline/build_travel.py).
(function () {
    'use strict';
    var T = window.TRAVEL, G = window.GEO;
    if (!T || !G || !window.d3) return;

    var root = document.documentElement;
    var still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    var coarse = matchMedia('(pointer: coarse)').matches;
    var fmt = d3.format(',');
    var DAY = 864e5;

    // ---------- Theme colors (re-read when the theme flips) ----------
    var C = {};
    var NAMES = ['viz-1', 'viz-2', 'viz-3', 'viz-4', 'map-ocean', 'map-land', 'map-line', 'map-visited', 'map-visited-line',
        'seq-1', 'seq-2', 'seq-3', 'seq-4', 'seq-5', 'surface', 'bg', 'ink', 'ink-soft', 'ink-faint', 'rule'];
    var redraws = [];
    function readColors() {
        var s = getComputedStyle(root);
        NAMES.forEach(function (n) { C[n] = s.getPropertyValue('--' + n).trim(); });
    }
    function themeChanged() { readColors(); redraws.forEach(function (f) { f(); }); }
    readColors();
    new MutationObserver(themeChanged).observe(root, { attributes: true, attributeFilter: ['data-theme'] });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', themeChanged);

    // ---------- Small helpers ----------
    function el(tag, cls, text) {
        var e = document.createElement(tag);
        if (cls) e.className = cls;
        if (text != null) e.textContent = text;
        return e;
    }
    function dayDate(d) { return new Date(d * DAY); }
    function isoDate(s) { return new Date(s + 'T00:00:00Z'); }
    var fmtDay = d3.utcFormat('%b %-d, %Y');
    var fmtMonth = d3.utcFormat('%b %Y');
    function monthLabel(ym) { return fmtMonth(isoDate(ym + '-01')); }
    function debounce(fn, ms) { var t; return function () { clearTimeout(t); t = setTimeout(fn, ms); }; }

    // ---------- Tooltip ----------
    var tip = document.getElementById('tip');
    function showTip(x, y, title, lines) {
        tip.replaceChildren();
        tip.appendChild(el('strong', null, title));
        (lines || []).forEach(function (l) {
            if (typeof l === 'string') tip.appendChild(el('div', 'sub', l));
            else tip.appendChild(l);
        });
        tip.classList.add('on');
        var r = tip.getBoundingClientRect();
        var px = x + 16, py = y + 16;
        if (px + r.width > innerWidth - 8) px = x - r.width - 16;
        if (py + r.height > innerHeight - 8) py = y - r.height - 16;
        tip.style.left = Math.max(8, px) + 'px';
        tip.style.top = Math.max(8, py) + 'px';
    }
    function hideTip() { tip.classList.remove('on'); }
    function tipRow(color, value, label) {
        var r = el('div', 'row');
        var k = el('span', 'key');
        k.style.background = color;
        r.appendChild(k);
        r.appendChild(el('b', null, value));
        r.appendChild(el('span', 'sub', label));
        return r;
    }
    addEventListener('scroll', hideTip, { passive: true });

    // ---------- Decode the dot cells ----------
    // Each cell is [lat, lon, first year]; visit counts are not published.
    var N = T.dots.length / 3;
    var dLat = new Float64Array(N), dLon = new Float64Array(N), dYr = new Uint8Array(N);
    for (var i = 0; i < N; i++) {
        dLat[i] = T.dots[3 * i] / 1000;
        dLon[i] = T.dots[3 * i + 1] / 1000;
        dYr[i] = T.dots[3 * i + 2];
    }
    var lastYearIdx = d3.max(dYr);

    var visitedCountry = new Map(T.countries.map(function (c) { return [c.name, c]; }));
    var visitedAdmin1 = new Map(T.admin1.map(function (c) { return [c.name, c]; }));
    var worldFC = G.world, admin1FC = G.admin1;
    var subFeatures = worldFC.features.filter(function (f) { return !f.properties.sub; });

    // Mercator is infinite at the poles, so flat maps fit to the inhabited band instead of the sphere.
    var FLAT_FIT = { type: 'MultiPoint', coordinates: [[-168, -56], [179, 76]] };

    // Flat maps stop at about a 9°-wide view (state level), a little short of the globe's limit,
    // so nobody can zoom down to individual neighborhoods.
    var MAX_FLAT_ZOOM = 40;
    var PRESETS = [
        { label: 'World', bounds: null },
        { label: 'North America', bounds: [[-128, 15], [-62, 58]] },
        { label: 'The West', bounds: [[-125, 31], [-102, 50]] },
        { label: 'Europe', bounds: [[-10, 36], [25, 58]] },
        { label: 'East Asia', bounds: [[98, -2], [145, 42]] }
    ].filter(Boolean);

    // ---------- Canvas surface ----------
    function Surface(container, aspect) {
        var canvas = document.createElement('canvas');
        container.appendChild(canvas);
        var s = { canvas: canvas, ctx: canvas.getContext('2d'), w: 0, h: 0, onResize: null };
        function fit() {
            var w = container.clientWidth;
            if (!w || w === s.w) return;
            var h = Math.round(aspect(w));
            var r = window.devicePixelRatio || 1;
            s.w = w; s.h = h;
            canvas.width = Math.round(w * r);
            canvas.height = Math.round(h * r);
            canvas.style.height = h + 'px';
            s.ctx.setTransform(r, 0, 0, r, 0, 0);
            if (s.onResize) s.onResize();
        }
        s.fit = fit;
        new ResizeObserver(debounce(fit, 60)).observe(container);
        fit();
        return s;
    }

    function mapAspect(w) { return w < 600 ? w * 0.85 : Math.min(w * 0.56, 620); }
    function globeAspect(w) { return w < 600 ? w : Math.min(w * 0.62, 640); }

    // ---------- Map view: flat (zoom + pan) or globe (drag to spin) ----------
    var views = [];
    function MapView(container, o) {
        var holder = el('div');
        container.appendChild(holder);
        var surf = Surface(holder, o.aspect || (o.globe ? globeAspect : mapAspect));
        var canvas = surf.canvas, ctx = surf.ctx;
        var v = { holder: holder, canvas: canvas, ctx: ctx, globe: !!o.globe, k: 1, engaged: false, dragging: false, shown: true };
        var tr = d3.zoomIdentity, raf = 0, base, s0, t0, spin = null, lastT = d3.zoomIdentity;
        var proj = o.globe ? d3.geoOrthographic().clipAngle(90).precision(0.7).rotate(o.rotate || [96, -34])
            : (o.projection || d3.geoMercator)().precision(0.7);
        v.proj = proj;
        v.path = d3.geoPath(proj, ctx);
        if (!o.globe) base = (o.projection || d3.geoMercator)();

        function layout() {
            v.w = surf.w; v.h = surf.h;
            if (o.globe) {
                v.r0 = Math.min(v.w, v.h) / 2 - 8;
                proj.translate([v.w / 2, v.h / 2]);
            } else {
                base.fitExtent([[6, 6], [v.w - 6, v.h - 6]], o.fit);
                s0 = base.scale(); t0 = base.translate();
                proj.clipExtent([[0, 0], [v.w, v.h]]);
                zoom.translateExtent([[0, 0], [v.w, v.h]]).extent([[0, 0], [v.w, v.h]]);
            }
            apply();
            v.render();
        }
        function apply() {
            if (o.globe) proj.scale(v.r0 * tr.k);
            else proj.scale(s0 * tr.k).translate([t0[0] * tr.k + tr.x, t0[1] * tr.k + tr.y]);
            v.k = tr.k;
        }
        v.render = function () {
            if (!raf && v.shown) raf = requestAnimationFrame(function () { raf = 0; paint(); });
        };
        function paint() {
            v.dirty = true;
            ctx.clearRect(0, 0, v.w, v.h);
            if (o.globe) {
                ctx.fillStyle = C.surface; ctx.fillRect(0, 0, v.w, v.h);
                ctx.beginPath(); v.path({ type: 'Sphere' });
                ctx.fillStyle = C['map-ocean']; ctx.fill();
                ctx.strokeStyle = C['map-line']; ctx.lineWidth = 1; ctx.stroke();
                ctx.beginPath(); v.path(d3.geoGraticule10());
                ctx.strokeStyle = C['map-line']; ctx.globalAlpha = 0.45; ctx.lineWidth = 0.5; ctx.stroke(); ctx.globalAlpha = 1;
            } else {
                ctx.fillStyle = C['map-ocean']; ctx.fillRect(0, 0, v.w, v.h);
            }
            o.draw(v);
        }
        v.visible = function (lon, lat) {
            if (!o.globe) return true;
            var r = proj.rotate();
            return d3.geoDistance([lon, lat], [-r[0], -r[1]]) < 1.55;
        };
        v.project = function (lon, lat) {
            if (o.globe && !v.visible(lon, lat)) return null;
            return proj([lon, lat]);
        };

        var zoom = d3.zoom()
            .scaleExtent(o.globe ? [0.75, 40] : [1, o.maxZoom || MAX_FLAT_ZOOM])
            .filter(function (e) {
                if (e.type === 'wheel') return v.engaged || e.ctrlKey || e.metaKey;
                if (e.type === 'touchstart') return e.touches.length >= 2;
                return !e.button;
            })
            .on('start', function (e) {
                stopSpin();
                if (e.sourceEvent && /down|start/.test(e.sourceEvent.type)) v.dragging = true;
                hideTip();
            })
            .on('zoom', function (e) {
                var t = e.transform;
                if (o.globe && e.sourceEvent && t.k === lastT.k) {
                    var r = proj.rotate(), deg = 180 / Math.PI / proj.scale();
                    proj.rotate([r[0] + (t.x - lastT.x) * deg, Math.max(-85, Math.min(85, r[1] - (t.y - lastT.y) * deg)), r[2]]);
                }
                lastT = t;
                tr = t;
                apply();
                v.render();
            })
            .on('end', function () { v.dragging = false; v.render(); });
        d3.select(canvas).call(zoom).on('dblclick.zoom', null);
        v.zoom = zoom;

        canvas.addEventListener('pointerdown', function () { v.engaged = true; });
        canvas.addEventListener('pointerleave', function () { v.engaged = false; hideTip(); if (o.leave) o.leave(v); });
        canvas.addEventListener('pointermove', function (e) {
            if (v.dragging || !o.hover) return;
            var b = canvas.getBoundingClientRect();
            o.hover(v, e.clientX - b.left, e.clientY - b.top, e);
        });

        var btns = el('div', 'zoom-btns');
        [['+', 2], ['−', 0.5]].forEach(function (b) {
            var bt = el('button', null, b[0]);
            bt.type = 'button';
            bt.setAttribute('aria-label', b[1] > 1 ? 'Zoom in' : 'Zoom out');
            bt.onclick = function () { stopSpin(); d3.select(canvas).transition().duration(still ? 0 : 400).call(zoom.scaleBy, b[1]); };
            btns.appendChild(bt);
        });
        holder.style.position = 'relative';
        holder.appendChild(btns);
        var hint = el('div', 'hint', coarse ? 'Two fingers to move the map' : (o.globe ? 'Drag to spin · click, then scroll to zoom' : 'Drag to pan · click, then scroll to zoom'));
        holder.appendChild(hint);

        v.flyTo = function (bounds) {
            stopSpin();
            var sel = d3.select(canvas).transition().duration(still ? 0 : 1100);
            if (o.globe) {
                var c = bounds ? [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2] : [-96, 34];
                var span = bounds ? Math.max(bounds[1][0] - bounds[0][0], (bounds[1][1] - bounds[0][1]) * 1.4) : 180;
                var k = Math.max(0.9, Math.min(40, 120 / span));
                var r0 = proj.rotate(), r1 = [-c[0], -c[1], 0];
                if (r1[0] - r0[0] > 180) r1[0] -= 360;
                if (r0[0] - r1[0] > 180) r1[0] += 360;
                var ir = d3.interpolate(r0, r1);
                sel.tween('spin', function () { return function (t) { proj.rotate(ir(t)); v.render(); }; })
                    .call(zoom.transform, d3.zoomIdentity.scale(k));
                return;
            }
            if (!bounds) { sel.call(zoom.transform, d3.zoomIdentity); return; }
            var p0 = base([bounds[0][0], bounds[1][1]]), p1 = base([bounds[1][0], bounds[0][1]]);
            var kk = Math.min(o.maxZoom || MAX_FLAT_ZOOM, 0.92 / Math.max((p1[0] - p0[0]) / v.w, (p1[1] - p0[1]) / v.h));
            sel.call(zoom.transform, d3.zoomIdentity.translate(v.w / 2 - kk * (p0[0] + p1[0]) / 2, v.h / 2 - kk * (p0[1] + p1[1]) / 2).scale(kk));
        };

        // gentle spin while the globe is on screen, until someone touches it
        var onScreen = false, touched = false;
        function stopSpin() { touched = true; if (spin) { spin.stop(); spin = null; } }
        function startSpin() {
            if (!o.globe || still || touched || spin || !onScreen || !v.shown) return;
            var last = 0;
            spin = d3.timer(function (el) {
                var r = proj.rotate();
                proj.rotate([r[0] + (el - last) * 0.004, r[1], r[2]]);
                last = el;
                v.render();
            });
        }
        if (o.globe) {
            new IntersectionObserver(function (es) {
                onScreen = es[0].isIntersecting;
                if (onScreen) startSpin(); else if (spin) { spin.stop(); spin = null; }
            }).observe(canvas);
        }
        v.show = function (on) {
            v.shown = on;
            holder.style.display = on ? '' : 'none';
            if (on) { surf.fit(); layout(); startSpin(); } else if (spin) { spin.stop(); spin = null; }
        };
        surf.onResize = layout;
        layout();
        views.push(v);
        return v;
    }
    redraws.push(function () { views.forEach(function (v) { v.render(); }); });

    // Nearest-point lookup over whatever is currently drawn (rebuilt lazily after each paint)
    function Picker() {
        var qt = null, pts = [];
        return {
            reset: function () { pts = []; qt = null; },
            add: function (x, y, d) { pts.push([x, y, d]); },
            find: function (x, y, r) {
                if (!qt) qt = d3.quadtree(pts);
                var p = qt.find(x, y, r);
                return p ? p[2] : null;
            }
        };
    }

    function wireViewSwitch(section, map, globe, presetsFor) {
        var seg = section.querySelector('.seg');
        var cur = globe || map;   // globe is the default view where there is one
        if (seg) {
            seg.addEventListener('click', function (e) {
                var b = e.target.closest('button');
                if (!b) return;
                seg.querySelectorAll('button').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
                cur = b.dataset.view === 'globe' ? globe : map;
                map.show(cur === map);
                globe.show(cur === globe);
            });
            map.show(false);
        }
        var chips = section.querySelector('[data-presets]');
        if (chips) {
            presetsFor.forEach(function (p) {
                var c = el('button', 'chip', p.label);
                c.type = 'button';
                c.setAttribute('aria-pressed', 'false');
                c.onclick = function () { cur.flyTo(p.bounds); };
                chips.appendChild(c);
            });
        }
    }

    function drawLand(v, fc, fill, line, width) {
        var ctx = v.ctx;
        ctx.beginPath(); v.path(fc);
        ctx.fillStyle = fill; ctx.fill();
        if (line) { ctx.strokeStyle = line; ctx.lineWidth = width || 0.5; ctx.stroke(); }
    }

    // ---------- Header stats ----------
    (function stats() {
        var span = T.span.map(isoDate);
        document.querySelector('[data-span]').textContent = span[0].getUTCFullYear() + '–' + span[1].getUTCFullYear();
        var us = T.admin1.filter(function (a) { return a.country === 'US' && a.name !== 'District of Columbia'; }).length;
        var dc = T.admin1.some(function (a) { return a.name === 'District of Columbia'; });
        var TERR = ['Yukon', 'Northwest Territories', 'Nunavut'];
        var ca = T.admin1.filter(function (a) { return a.country === 'CA' && TERR.indexOf(a.name) < 0; }).length;
        var caT = T.admin1.filter(function (a) { return TERR.indexOf(a.name) >= 0; }).length;
        var parksOn = T.parks.filter(function (p) { return p.visited; }).length;
        var rows = [
            ['Countries', T.countries.length, continentsNote()],
            ['U.S. states', us, 'of 50' + (dc ? ', plus D.C.' : '')],
            ['Provinces', ca, 'of Canada’s 10' + (caT ? ', + ' + caT + ' terr.' : '')],
            ['National parks', parksOn, 'of ' + T.parks.length],
            ['Stops logged', fmt(T.stops), 'at ' + fmt(N) + ' places'],
            ['Days logged', fmt(T.daily.length), 'since ' + fmtMonth(span[0])]
        ];
        var dl = document.getElementById('stats');
        rows.forEach(function (r) {
            var d = el('div');
            d.appendChild(el('dt', null, r[0]));
            var dd = el('dd', null, String(r[1]));
            dd.appendChild(el('small', null, r[2]));
            d.appendChild(dd);
            dl.appendChild(d);
        });
        var u = document.querySelector('[data-updated]');
        if (u) u.textContent = 'Updated ' + fmtMonth(isoDate(T.generated));
    })();

    function continentsNote() {
        var CONT = { 'United States': 'NA', Canada: 'NA', Mexico: 'NA', Guatemala: 'NA', Honduras: 'NA', Belize: 'NA', Jamaica: 'NA', Bahamas: 'NA', 'Dominican Republic': 'NA',
            Brazil: 'SA', 'United Kingdom': 'EU', Germany: 'EU', Austria: 'EU', 'Czech Republic': 'EU', Czechia: 'EU', Singapore: 'AS', Malaysia: 'AS', Japan: 'AS' };
        var n = new Set(T.countries.map(function (c) { return CONT[c.name]; }).filter(Boolean)).size;
        return n ? 'on ' + n + ' continents' : '';
    }

    // ---------- 1. Every place ----------
    (function everyPlace() {
        var section = document.getElementById('everywhere');
        var box = document.getElementById('dots-viz');
        var slider = section.querySelector('input[type=range]');
        var out = section.querySelector('output');
        var y0 = T.year0;
        slider.min = y0; slider.max = y0 + lastYearIdx; slider.value = slider.max;
        var maxIdx = lastYearIdx;
        var counter = section.querySelector('[data-dots-count]');
        function setYear(y) {
            maxIdx = y - y0;
            out.textContent = y;
            var n = 0;
            for (var i = 0; i < N; i++) if (dYr[i] <= maxIdx) n++;
            counter.textContent = fmt(n) + ' places on the timeline';
            map.render(); globe.render();
        }

        function draw(v) {
            var ctx = v.ctx, pick = v.pick || (v.pick = Picker());
            pick.reset();
            drawLand(v, worldFC, C['map-land'], C['map-line'], 0.5);
            if (v.k > 2.5 || v.globe) {
                ctx.beginPath(); v.path(admin1FC);
                ctx.strokeStyle = C['map-line']; ctx.lineWidth = 0.5; ctx.stroke();
            }
            var s = Math.max(2.6, Math.min(6, 2.4 + Math.log2(v.k) * 0.45));
            ctx.fillStyle = C['viz-1'];
            for (var i = 0; i < N; i++) {
                if (dYr[i] > maxIdx) continue;
                var p = v.project(dLon[i], dLat[i]);
                if (!p || p[0] < -4 || p[1] < -4 || p[0] > v.w + 4 || p[1] > v.h + 4) continue;
                ctx.fillRect(p[0] - s / 2, p[1] - s / 2, s, s);
                pick.add(p[0], p[1], i);
            }
        }
        function hover(v, x, y, e) {
            var d = v.pick && v.pick.find(x, y, 10);
            if (d == null) return hideTip();
            showTip(e.clientX, e.clientY, 'First seen in ' + (y0 + dYr[d]));
        }
        var map = MapView(box, { draw: draw, hover: hover, fit: FLAT_FIT, projection: d3.geoMercator });
        var globe = MapView(box, { globe: true, draw: draw, hover: hover });
        wireViewSwitch(section, map, globe, PRESETS);
        slider.addEventListener('input', function () { stopPlay(); setYear(+slider.value); });

        var timer = null;
        var play = section.querySelector('.play');
        function stopPlay() { if (timer) { clearInterval(timer); timer = null; } }
        play.onclick = function () {
            if (timer) return stopPlay();
            var y = y0;
            setYear(y); slider.value = y;
            timer = setInterval(function () {
                y++;
                if (y > y0 + lastYearIdx) return stopPlay();
                slider.value = y; setYear(y);
            }, still ? 300 : 900);
        };
        setYear(+slider.max);
    })();

    // ---------- 2. Scratch-off: countries, states, provinces ----------
    (function scratch() {
        var section = document.getElementById('scratch');
        var box = document.getElementById('scratch-viz');
        var feats = subFeatures.concat(admin1FC.features);
        var bounds = feats.map(function (f) { return d3.geoBounds(f); });
        function isOn(f) {
            return f.properties.country ? visitedAdmin1.has(f.properties.name) : visitedCountry.has(f.properties.name);
        }
        var on = { type: 'FeatureCollection', features: feats.filter(isOn) };
        var off = { type: 'FeatureCollection', features: feats.filter(function (f) { return !isOn(f); }) };

        function draw(v) {
            var ctx = v.ctx;
            ctx.beginPath(); v.path(off);
            ctx.fillStyle = C['map-land']; ctx.fill();
            ctx.beginPath(); v.path(on);
            ctx.fillStyle = C['map-visited']; ctx.fill();
            ctx.beginPath(); v.path(off);
            ctx.strokeStyle = C['map-line']; ctx.lineWidth = 0.5; ctx.stroke();
            ctx.beginPath(); v.path(on);
            ctx.strokeStyle = C['map-visited-line']; ctx.lineWidth = 0.6; ctx.stroke();
            if (v.hl) {
                ctx.beginPath(); v.path(v.hl);
                ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.stroke();
            }
        }
        function find(ll) {
            for (var i = 0; i < feats.length; i++) {
                var b = bounds[i];
                var inLon = b[0][0] <= b[1][0] ? ll[0] >= b[0][0] && ll[0] <= b[1][0] : ll[0] >= b[0][0] || ll[0] <= b[1][0];
                if (inLon && ll[1] >= b[0][1] && ll[1] <= b[1][1] && d3.geoContains(feats[i], ll)) return feats[i];
            }
            return null;
        }
        function hover(v, x, y, e) {
            var ll = v.proj.invert([x, y]);
            var f = ll && (!v.globe || v.visible(ll[0], ll[1])) ? find(ll) : null;
            if (f !== v.hl) { v.hl = f; v.render(); }
            if (!f) return hideTip();
            var rec = f.properties.country ? visitedAdmin1.get(f.properties.name) : visitedCountry.get(f.properties.name);
            var lines = !rec ? ['Not yet'] : [rec.first ? 'First logged stop ' + fmtMonth(isoDate(rec.first)) : 'Visited'];
            showTip(e.clientX, e.clientY, f.properties.name, lines);
        }
        function leave(v) { if (v.hl) { v.hl = null; v.render(); } }
        var map = MapView(box, { draw: draw, hover: hover, leave: leave, fit: { type: 'Sphere' }, projection: d3.geoNaturalEarth1, maxZoom: 60 });
        var globe = MapView(box, { globe: true, draw: draw, hover: hover, leave: leave, rotate: [80, -35] });
        wireViewSwitch(section, map, globe, [
            { label: 'World', bounds: null },
            { label: 'North America', bounds: [[-135, 12], [-55, 62]] },
            { label: 'Europe', bounds: [[-11, 40], [22, 58]] },
            { label: 'Asia', bounds: [[95, -5], [145, 45]] }
        ]);

        // tallies (the table view of this map)
        var tally = document.getElementById('tally');
        var allUS = admin1FC.features.filter(function (f) { return f.properties.country === 'US'; }).map(function (f) { return f.properties.name; }).sort();
        var allCA = admin1FC.features.filter(function (f) { return f.properties.country === 'CA'; }).map(function (f) { return f.properties.name; }).sort();
        [['Countries', T.countries.map(function (c) { return c.name; }), null],
         ['U.S. states', allUS, visitedAdmin1],
         ['Canada', allCA, visitedAdmin1]].forEach(function (g) {
            var d = el('div');
            var names = g[2] ? g[1] : g[1].slice().sort();
            var states = names.filter(function (n) { return n !== 'District of Columbia'; });
            var count = g[2] ? states.filter(function (n) { return g[2].has(n); }).length : names.length;
            var h = el('h3', null, g[0]);
            h.appendChild(el('b', null, g[2] ? count + ' / ' + states.length : String(count)));
            d.appendChild(h);
            var wrap = el('div', 'names');
            names.forEach(function (n) { wrap.appendChild(el('span', !g[2] || g[2].has(n) ? '' : 'off', n)); });
            d.appendChild(wrap);
            tally.appendChild(d);
        });
    })();

    // ---------- 3. National parks ----------
    (function parks() {
        var on = T.parks.filter(function (p) { return p.visited; });
        document.querySelector('[data-parks-count]').textContent = on.length;
        document.querySelector('[data-parks-total]').textContent = T.parks.length;

        var box = document.getElementById('parks-viz');
        var states = { type: 'FeatureCollection', features: admin1FC.features.filter(function (f) { return f.properties.country === 'US'; }) };
        var svg = d3.select(box).append('svg').attr('role', 'img').attr('aria-label', 'Map of U.S. national parks, visited ones filled');
        var gS = svg.append('g'), gP = svg.append('g');

        function render() {
            var w = box.clientWidth, h = Math.round(w < 600 ? w * 0.7 : Math.min(w * 0.6, 600));
            svg.attr('viewBox', [0, 0, w, h]).attr('height', h);
            var proj = d3.geoAlbersUsa().fitExtent([[10, 10], [w - 10, h - 10]], states);
            var path = d3.geoPath(proj);
            gS.selectAll('path').data(states.features).join('path')
                .attr('d', path).style('fill', 'var(--map-land)').style('stroke', 'var(--map-line)').style('stroke-width', 0.6);
            var pts = T.parks.map(function (p) { return { p: p, xy: proj([p.lon, p.lat]) }; }).filter(function (d) { return d.xy; })
                .sort(function (a, b) { return a.p.visited - b.p.visited; });
            var r = w < 600 ? 4 : 6;
            var g = gP.selectAll('g').data(pts, function (d) { return d.p.name; }).join(function (enter) {
                var g = enter.append('g').attr('tabindex', 0).attr('role', 'img');
                g.append('circle').attr('class', 'hit').style('fill', 'transparent');
                g.append('circle').attr('class', 'dot');
                return g;
            });
            g.attr('transform', function (d) { return 'translate(' + d.xy + ')'; })
                .attr('aria-label', function (d) { return d.p.name + (d.p.visited ? ', visited' : ', not yet'); });
            g.select('.hit').attr('r', 12);
            g.select('.dot').attr('r', r)
                .style('fill', function (d) { return d.p.visited ? 'var(--map-visited)' : 'var(--map-land)'; })
                .style('stroke', function (d) { return d.p.visited ? 'var(--map-ocean)' : 'var(--ink-faint)'; })
                .style('stroke-width', function (d) { return d.p.visited ? 2 : 1; });
            g.on('pointerenter focus', function (e, d) {
                d3.select(this).select('.dot').attr('r', r + 2);
                var b = this.getBoundingClientRect();
                showTip(e.clientX || b.right, e.clientY || b.bottom, d.p.name + ' National Park', [stateLabel(d.p.state), parkWhen(d.p)]);
            }).on('pointerleave blur', function () { d3.select(this).select('.dot').attr('r', r); hideTip(); });
        }
        new ResizeObserver(debounce(render, 80)).observe(box);

        // poster
        var grid = document.getElementById('parks-grid');
        var shapes = new Map(G.parks.features.map(function (f) { return [f.properties.name, f]; }));
        T.parks.slice().sort(function (a, b) { return a.name.localeCompare(b.name); }).forEach(function (p) {
            var card = el('div', 'park' + (p.visited ? ' on' : ''));
            card.dataset.on = p.visited ? 'on' : 'off';
            var f = shapes.get(p.name);
            var s = d3.select(card).append('svg').attr('viewBox', '0 0 120 76').attr('aria-hidden', 'true');
            if (f) s.append('path').attr('d', d3.geoPath(d3.geoMercator().fitExtent([[4, 4], [116, 72]], f))(f));
            card.appendChild(el('h4', null, p.name));
            card.appendChild(el('p', null, stateLabel(p.state)));
            card.appendChild(el('p', 'when', parkWhen(p)));
            grid.appendChild(card);
        });
        var filt = document.getElementById('park-filter');
        filt.addEventListener('click', function (e) {
            var b = e.target.closest('button');
            if (!b) return;
            filt.querySelectorAll('button').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
            grid.querySelectorAll('.park').forEach(function (c) { c.classList.toggle('hide', b.dataset.f !== 'all' && c.dataset.on !== b.dataset.f); });
        });
    })();
    function stateLabel(s) { return s === 'SAMOA' ? 'AS' : s.replace(/-/g, ' · '); }
    function parkWhen(p) {
        if (!p.visited) return 'Not yet';
        return p.first ? 'First logged ' + fmtMonth(isoDate(p.first)) : 'Visited';
    }

    // ---------- 5. Away from home: calendar, home bases, trips ----------
    var byDay = new Map(T.daily.map(function (d) { return [d[0], d]; }));
    var BANDS = [
        { max: 50, label: 'Home (< 50 km)', c: 'seq-1' },
        { max: 300, label: '50–300 km', c: 'seq-2' },
        { max: 1000, label: '300–1,000 km', c: 'seq-3' },
        { max: 4000, label: '1,000–4,000 km', c: 'seq-4' },
        { max: Infinity, label: '4,000 km +', c: 'seq-5' }
    ];
    function band(kmv) { for (var i = 0; i < BANDS.length; i++) if (kmv < BANDS[i].max) return BANDS[i]; }

    (function calendar() {
        var box = document.getElementById('cal-viz');
        var y0 = isoDate(T.span[0]).getUTCFullYear(), y1 = isoDate(T.span[1]).getUTCFullYear();
        var years = d3.range(y0, y1 + 1);
        var LEFT = 46, TOP = 22, ROW = 20, GAP = 7;
        var surf = Surface(box, function () { return TOP + years.length * (ROW + GAP) + 6; });
        var hover = null;
        function geom() { return { cw: (surf.w - LEFT - 10) / 366 }; }
        function paint() {
            var ctx = surf.ctx, g = geom();
            ctx.clearRect(0, 0, surf.w, surf.h);
            ctx.fillStyle = C.surface; ctx.fillRect(0, 0, surf.w, surf.h);
            ctx.font = '500 11px Manrope, sans-serif';
            ctx.fillStyle = C['ink-faint']; ctx.textBaseline = 'middle';
            d3.range(12).forEach(function (m) {
                var doy = d3.utcDay.count(new Date(Date.UTC(2021, 0, 1)), new Date(Date.UTC(2021, m, 1)));
                ctx.fillText(d3.utcFormat('%b')(new Date(Date.UTC(2021, m, 1))), LEFT + doy * g.cw, 9);
            });
            years.forEach(function (y, r) {
                var top = TOP + r * (ROW + GAP);
                ctx.fillStyle = C['ink-faint'];
                ctx.fillText(String(y), 4, top + ROW / 2);
                ctx.fillStyle = C.rule;
                ctx.fillRect(LEFT, top + ROW / 2, g.cw * 366, 1);
                var start = Date.UTC(y, 0, 1) / DAY;
                for (var d = 0; d < 366; d++) {
                    var rec = byDay.get(start + d);
                    if (!rec || dayDate(start + d).getUTCFullYear() !== y) continue;
                    ctx.fillStyle = C[band(rec[1]).c];
                    ctx.fillRect(LEFT + d * g.cw, top, Math.max(1, g.cw - (g.cw > 3 ? 1 : 0)), ROW);
                }
            });
            if (hover) {
                ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5;
                ctx.strokeRect(LEFT + hover.d * g.cw - 1, TOP + hover.r * (ROW + GAP) - 2, Math.max(2, g.cw) + 2, ROW + 4);
            }
        }
        surf.onResize = paint;
        paint();
        redraws.push(paint);
        surf.canvas.style.cursor = 'default';
        surf.canvas.style.touchAction = 'pan-y';
        surf.canvas.addEventListener('pointermove', function (e) {
            var b = surf.canvas.getBoundingClientRect(), g = geom();
            var x = e.clientX - b.left, y = e.clientY - b.top;
            var r = Math.floor((y - TOP + GAP / 2) / (ROW + GAP)), d = Math.floor((x - LEFT) / g.cw);
            if (r < 0 || r >= years.length || d < 0 || d > 365) { hover = null; paint(); return hideTip(); }
            var dn = Date.UTC(years[r], 0, 1) / DAY + d;
            if (dayDate(dn).getUTCFullYear() !== years[r]) { hover = null; paint(); return hideTip(); }
            hover = { r: r, d: d }; paint();
            var rec = byDay.get(dn);
            showTip(e.clientX, e.clientY, fmtDay(dayDate(dn)), rec
                ? [rec[1] < 50 ? 'Home' : fmt(Math.round(rec[1])) + ' km from home' + (rec[2] ? ' · ' + rec[2] : '')]
                : ['No data recorded']);
        });
        surf.canvas.addEventListener('pointerleave', function () { hover = null; paint(); hideTip(); });

        var lg = document.getElementById('cal-legend');
        BANDS.forEach(function (b) {
            var s = el('span'), i = el('i', 'sq');
            i.style.background = 'var(--' + b.c + ')';
            s.appendChild(i); s.appendChild(document.createTextNode(b.label));
            lg.appendChild(s);
        });
        lg.appendChild(el('span', 'note', 'Farthest point from home that day'));
    })();

    (function homes() {
        var box = document.getElementById('home-viz');
        var names = [];
        T.homes.forEach(function (h) { if (names.indexOf(h.name) < 0) names.push(h.name); });
        var color = function (n) { var i = names.indexOf(n); return i < 4 ? 'var(--viz-' + (i + 1) + ')' : 'var(--ink-faint)'; };
        var svg = d3.select(box).append('svg').attr('role', 'img').attr('aria-label', 'Where home was, over time');
        function render() {
            var w = box.clientWidth, h = 64;
            svg.attr('viewBox', [0, 0, w, h]).attr('height', h);
            var x = d3.scaleUtc().domain([isoDate(T.span[0]), isoDate(T.span[1])]).range([12, w - 12]);
            var bars = T.homes.map(function (p) {
                var a = isoDate(p.start + '-01'), b = d3.utcMonth.offset(isoDate(p.end + '-01'), 1);
                return { p: p, x0: x(Math.max(a, x.domain()[0])), x1: x(Math.min(b, x.domain()[1])) };
            });
            var g = svg.selectAll('g.bar').data(bars).join(function (en) {
                var g = en.append('g').attr('class', 'bar').attr('tabindex', 0);
                g.append('rect'); g.append('text');
                return g;
            });
            g.select('rect').attr('x', function (d) { return d.x0 + 1; }).attr('y', 8)
                .attr('width', function (d) { return Math.max(1, d.x1 - d.x0 - 2); }).attr('height', 26).attr('rx', 3)
                .style('fill', function (d) { return color(d.p.name); });
            g.select('text').attr('x', function (d) { return d.x0 + 8; }).attr('y', 25)
                .style('font-size', '11px').style('font-weight', 600).style('fill', 'var(--bg)')
                .text(function (d) { var s = d.p.name.split(',')[0]; return d.x1 - d.x0 > s.length * 7 + 14 ? s : ''; });
            g.on('pointerenter focus', function (e, d) {
                var b = this.getBoundingClientRect();
                showTip(e.clientX || b.left, e.clientY || b.bottom, d.p.name, [monthLabel(d.p.start) + ' – ' + monthLabel(d.p.end)]);
            }).on('pointerleave blur', hideTip);
            var ax = svg.selectAll('g.axis').data([0]).join('g').attr('class', 'axis').attr('transform', 'translate(0,40)');
            ax.selectAll('text').data(x.ticks(d3.utcYear.every(w < 600 ? 2 : 1))).join('text')
                .attr('x', function (d) { return x(d); }).attr('y', 16).attr('text-anchor', 'middle')
                .style('font-size', '11px').style('fill', 'var(--ink-faint)').text(d3.utcFormat('%Y'));
        }
        new ResizeObserver(debounce(render, 80)).observe(box);
        var lg = document.getElementById('home-legend');
        names.slice(0, 4).forEach(function (n) {
            var s = el('span'), i = el('i', 'sq');
            i.style.background = color(n);
            s.appendChild(i); s.appendChild(document.createTextNode(n));
            lg.appendChild(s);
        });
        lg.appendChild(el('span', 'note', 'Where I slept most, month by month'));
    })();

    (function trips() {
        var away = T.daily.filter(function (d) { return d[1] >= 80; });
        var runs = [];
        away.forEach(function (d) {
            var r = runs[runs.length - 1];
            if (r && d[0] - r.end <= 2) { r.end = d[0]; r.days.push(d); } else runs.push({ start: d[0], end: d[0], days: [d] });
        });
        runs.forEach(function (r) {
            r.len = r.end - r.start + 1;
            var c = d3.rollup(r.days.filter(function (d) { return d[2]; }), function (v) { return v.length; }, function (d) { return d[2]; });
            r.where = Array.from(c).sort(function (a, b) { return b[1] - a[1]; }).slice(0, 3).map(function (x) { return x[0]; }).join(', ');
            r.far = d3.max(r.days, function (d) { return d[1]; });
        });
        var top = runs.sort(function (a, b) { return b.len - a.len; }).slice(0, 10);
        var tb = document.querySelector('#trip-table tbody');
        top.forEach(function (r) {
            var tr = el('tr');
            tr.appendChild(el('td', null, fmtDay(dayDate(r.start)) + ' – ' + fmtDay(dayDate(r.end))));
            tr.appendChild(el('td', 'num', String(r.len)));
            tr.appendChild(el('td', null, r.where || '—'));
            tr.appendChild(el('td', 'num', fmt(Math.round(r.far)) + ' km'));
            tb.appendChild(tr);
        });
    })();

    function legend(id, items, note) {
        var lg = document.getElementById(id);
        items.forEach(function (m) {
            var s = el('span'), i = el('i', 'sq');
            i.style.background = m.c;
            s.appendChild(i); s.appendChild(document.createTextNode(m.label));
            lg.appendChild(s);
        });
        if (note) lg.appendChild(el('span', 'note', note));
    }
    // ---------- 6. The collection ----------
    (function collection() {
        var box = document.getElementById('col-viz');
        var SER = [
            { key: 'admin1', label: 'States & provinces', c: 'var(--viz-1)', items: T.admin1 },
            { key: 'country', label: 'Countries', c: 'var(--viz-2)', items: T.countries },
            { key: 'park', label: 'National parks', c: 'var(--viz-3)', items: T.parks.filter(function (p) { return p.visited; }) }
        ];
        var t0 = isoDate(T.span[0]), t1 = isoDate(T.span[1]);
        SER.forEach(function (s) {
            var before = s.items.filter(function (d) { return !d.first; }).length;
            var ev = s.items.filter(function (d) { return d.first; }).map(function (d) { return { t: isoDate(d.first), name: d.name }; })
                .sort(function (a, b) { return a.t - b.t; });
            var pts = [{ t: t0, n: before }], n = before;
            ev.forEach(function (e) { n++; pts.push({ t: e.t, n: n, name: e.name }); });
            pts.push({ t: t1, n: n });
            s.before = before; s.pts = pts; s.ev = ev;
        });
        var svg = d3.select(box).append('svg').attr('role', 'img').attr('aria-label', 'Running count of states, countries and parks first seen on the timeline');
        function render() {
            var w = box.clientWidth, h = w < 600 ? 280 : 340, narrow = w < 600;
            var m = { t: 18, r: narrow ? 16 : 150, b: 28, l: 40 };
            svg.attr('viewBox', [0, 0, w, h]).attr('height', h).selectAll('*').remove();
            var x = d3.scaleUtc().domain([t0, t1]).range([m.l, w - m.r]);
            var y = d3.scaleLinear().domain([0, d3.max(SER, function (s) { return s.pts[s.pts.length - 1].n; })]).nice().range([h - m.b, m.t]);
            y.ticks(5).forEach(function (t) {
                svg.append('line').attr('x1', m.l).attr('x2', w - m.r).attr('y1', y(t)).attr('y2', y(t)).style('stroke', 'var(--rule)');
                svg.append('text').attr('x', m.l - 8).attr('y', y(t)).attr('dy', '0.32em').attr('text-anchor', 'end')
                    .style('font-size', '11px').style('fill', 'var(--ink-faint)').text(t);
            });
            svg.append('g').selectAll('text').data(x.ticks(narrow ? 5 : 10)).join('text')
                .attr('x', function (d) { return x(d); }).attr('y', h - 8).attr('text-anchor', 'middle')
                .style('font-size', '11px').style('fill', 'var(--ink-faint)').text(d3.utcFormat('%Y'));
            var line = d3.line().curve(d3.curveStepAfter).x(function (d) { return x(d.t); }).y(function (d) { return y(d.n); });
            SER.forEach(function (s) {
                svg.append('path').attr('d', line(s.pts)).style('fill', 'none').style('stroke', s.c).style('stroke-width', 2);
                var end = s.pts[s.pts.length - 1];
                if (!narrow) {
                    svg.append('circle').attr('cx', x(end.t)).attr('cy', y(end.n)).attr('r', 4).style('fill', s.c)
                        .style('stroke', 'var(--surface)').style('stroke-width', 2);
                }
            });
            if (!narrow) {   // end labels, nudged apart
                var labs = SER.map(function (s) { var e = s.pts[s.pts.length - 1]; return { s: s, y: y(e.n), n: e.n }; }).sort(function (a, b) { return a.y - b.y; });
                for (var i = 1; i < labs.length; i++) if (labs[i].y - labs[i - 1].y < 16) labs[i].y = labs[i - 1].y + 16;
                labs.forEach(function (l) {
                    var t = svg.append('text').attr('x', w - m.r + 10).attr('y', l.y).attr('dy', '0.32em').style('font-size', '12px');
                    t.append('tspan').style('font-weight', 600).style('fill', 'var(--ink)').text(l.n + ' ');
                    t.append('tspan').style('fill', 'var(--ink-soft)').text(l.s.label.toLowerCase());
                });
            }
            var cross = svg.append('line').attr('y1', m.t).attr('y2', h - m.b).style('stroke', 'var(--ink-faint)').style('opacity', 0);
            svg.append('rect').attr('x', m.l).attr('y', m.t).attr('width', w - m.l - m.r).attr('height', h - m.t - m.b).style('fill', 'transparent')
                .on('pointermove', function (e) {
                    var px = d3.pointer(e)[0], t = x.invert(px);
                    cross.attr('x1', px).attr('x2', px).style('opacity', 1);
                    var rows = SER.map(function (s) {
                        var cur = s.pts[0];
                        s.pts.forEach(function (p) { if (p.t <= t) cur = p; });
                        return tipRow(s.c, String(cur.n), s.label);
                    });
                    var near = [];
                    SER.forEach(function (s) { s.ev.forEach(function (ev) { if (Math.abs(x(ev.t) - px) < 3) near.push(ev.name); }); });
                    if (near.length) rows.push(el('div', 'sub', 'New: ' + near.slice(0, 6).join(', ') + (near.length > 6 ? '…' : '')));
                    showTip(e.clientX, e.clientY, fmtMonth(t), rows);
                })
                .on('pointerleave', function () { cross.style('opacity', 0); hideTip(); });
        }
        new ResizeObserver(debounce(render, 80)).observe(box);
        legend('col-legend', SER, 'Lines start with places that have no logged stop');
        var all = [];
        SER.forEach(function (s) { s.items.forEach(function (d) { all.push([s.label, d.name, d.first]); }); });
        all.sort(function (a, b) { return (a[2] || '0') < (b[2] || '0') ? -1 : 1; });
        var tbl = d3.select('#col-table');
        tbl.append('thead').append('tr').selectAll('th').data(['First logged stop', 'Place', 'Kind']).join('th').text(function (d) { return d; });
        tbl.append('tbody').selectAll('tr').data(all).join('tr').selectAll('td')
            .data(function (d) { return [d[2] ? fmtDay(isoDate(d[2])) : 'No logged stop', d[1], d[0]]; })
            .join('td').text(function (d) { return d; });
    })();

    // ---------- 7. Compass ----------
    (function compass() {
        var E = T.extremes, dl = document.getElementById('compass-cards');
        function deg(v, pos, neg) { return Math.abs(v).toFixed(2) + '°' + (v >= 0 ? pos : neg); }
        [['North', E.north, deg(E.north.lat, 'N', 'S')], ['South', E.south, deg(E.south.lat, 'N', 'S')],
         ['East', E.east, deg(E.east.lon, 'E', 'W')], ['West', E.west, deg(E.west.lon, 'E', 'W')],
         ['Farthest from home', E.farthest, fmt(E.farthest.km) + ' km']].forEach(function (r) {
            var d = el('div');
            d.appendChild(el('dt', null, r[0]));
            var dd = el('dd', null, r[1].name);
            dd.appendChild(el('small', null, r[2] + ' · ' + fmtMonth(isoDate(r[1].day))));
            d.appendChild(dd);
            dl.appendChild(d);
        });
        document.querySelector('[data-fineprint]').textContent =
            'Built from my Google Timeline export (' + fmtMonth(isoDate(T.span[0])) + ' to ' + fmtMonth(isoDate(T.span[1])) +
            ') and hand-kept lists of earlier travel. Only places where the timeline logged a stop are mapped, not every GPS ping, and ' +
            'every location is generalized to about a kilometer, and how often I visited any one spot is left out. Boundaries from Natural Earth and the National Park Service; place names from GeoNames.';
    })();
})();
