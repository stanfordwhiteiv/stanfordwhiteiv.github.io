// Shared behavior: theme toggle, top-bar state, scroll reveal, photo gallery.
(function () {
    var root = document.documentElement;

    // ---------- Theme ----------
    var toggle = document.querySelector('.theme-toggle');
    if (toggle) {
        toggle.addEventListener('click', function () {
            var dark = root.dataset.theme
                ? root.dataset.theme === 'dark'
                : matchMedia('(prefers-color-scheme: dark)').matches;
            root.dataset.theme = dark ? 'light' : 'dark';
            try { localStorage.setItem('theme', root.dataset.theme); } catch (e) {}
        });
    }

    // ---------- Top bar ----------
    var bar = document.querySelector('.topbar');
    function onScroll() { bar && bar.classList.toggle('scrolled', scrollY > 40); }
    addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    // ---------- Reveal ----------
    if ('IntersectionObserver' in window) {
        var io = new IntersectionObserver(function (entries) {
            entries.forEach(function (e) {
                if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
            });
        }, { rootMargin: '0px 0px -8% 0px' });
        document.querySelectorAll('.reveal').forEach(function (el) { io.observe(el); });
    } else {
        document.querySelectorAll('.reveal').forEach(function (el) { el.classList.add('in'); });
    }

    // ---------- Showcase (landing page) ----------
    var showcase = document.querySelector('.showcase');
    if (showcase && window.PHOTOS) {
        var sBase = showcase.dataset.base || '';
        var featured = PHOTOS.filter(function (p) { return p.featured; });
        var slides = featured.map(function (p) {
            var s = document.createElement('div');
            s.style.backgroundImage = "url('" + sBase + 'full/' + p.file + "')";
            showcase.querySelector('.showcase-slides').appendChild(s);
            return s;
        });
        var dots = showcase.querySelector('.showcase-dots');
        var place = showcase.querySelector('.showcase-place');
        var idx = 0, timer;
        var still = matchMedia('(prefers-reduced-motion: reduce)').matches;
        function label() {
            place.textContent = featured[idx].location || '';
            place.classList.toggle('on', !!featured[idx].location);
        }
        function go(i) {
            slides[idx].classList.remove('on');
            dots.children[idx].classList.remove('on');
            idx = (i + slides.length) % slides.length;
            slides[idx].classList.add('on');
            dots.children[idx].classList.add('on');
            label();
        }
        function play() { clearInterval(timer); if (!still) timer = setInterval(function () { go(idx + 1); }, 7000); }
        slides.forEach(function (_, i) {
            var b = document.createElement('button');
            b.setAttribute('aria-label', 'Show photo ' + (i + 1));
            b.onclick = function () { go(i); play(); };
            dots.appendChild(b);
        });
        if (slides.length) {
            slides[0].classList.add('on');
            dots.children[0].classList.add('on');
            label();
            play();
        }
    }

    // ---------- Gallery ----------
    var gallery = document.querySelector('.gallery');
    if (!gallery || !window.PHOTOS) return;

    var base = gallery.dataset.base || '';
    var photos = gallery.dataset.featured !== undefined
        ? PHOTOS.filter(function (p) { return p.featured; })
        : PHOTOS;

    function describe(p) { return [p.caption, p.location].filter(Boolean).join(' · '); }

    photos.forEach(function (p, i) {
        var fig = document.createElement('figure');
        var img = document.createElement('img');
        img.src = base + 'thumb/' + p.file;
        img.width = p.w;
        img.height = p.h;
        img.loading = 'lazy';
        img.decoding = 'async';
        img.alt = describe(p) || 'Photograph by Stanford White IV';
        img.onload = function () { img.classList.add('loaded'); };
        fig.appendChild(img);
        if (describe(p)) {
            var cap = document.createElement('figcaption');
            cap.textContent = describe(p);
            fig.appendChild(cap);
        }
        fig.addEventListener('click', function () { open(i); });
        gallery.appendChild(fig);
    });

    // Light deterrent only: anything shown on a page can still be saved.
    document.addEventListener('contextmenu', function (e) {
        if (e.target.closest('.gallery, .lightbox')) e.preventDefault();
    });

    var lb = document.createElement('div');
    lb.className = 'lightbox no-save';
    lb.innerHTML =
        '<button class="lb-close" aria-label="Close">&times;</button>' +
        '<button class="lb-prev" aria-label="Previous">&#8249;</button>' +
        '<img alt="">' +
        '<button class="lb-next" aria-label="Next">&#8250;</button>' +
        '<div class="lb-caption"></div>';
    document.body.appendChild(lb);

    var lbImg = lb.querySelector('img');
    var lbCap = lb.querySelector('.lb-caption');
    var current = 0;

    function show(i) {
        current = (i + photos.length) % photos.length;
        var p = photos[current];
        lbImg.src = base + 'full/' + p.file;
        lbImg.alt = describe(p) || 'Photograph by Stanford White IV';
        lbCap.textContent = describe(p);
    }

    function open(i) { show(i); lb.classList.add('open'); document.body.style.overflow = 'hidden'; }
    function close() { lb.classList.remove('open'); document.body.style.overflow = ''; }

    lb.querySelector('.lb-close').onclick = close;
    lb.querySelector('.lb-prev').onclick = function (e) { e.stopPropagation(); show(current - 1); };
    lb.querySelector('.lb-next').onclick = function (e) { e.stopPropagation(); show(current + 1); };
    lb.addEventListener('click', function (e) { if (e.target === lb) close(); });
    addEventListener('keydown', function (e) {
        if (!lb.classList.contains('open')) return;
        if (e.key === 'Escape') close();
        if (e.key === 'ArrowLeft') show(current - 1);
        if (e.key === 'ArrowRight') show(current + 1);
    });
})();
