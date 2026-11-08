/* ==================================================================
   WEEK CALENDAR: shows the current week using this computer's local
   time zone, with a live clock. Updates itself every second and rolls
   over at midnight. Add your own code in the 3 marked spots.
   The HTML for it is in index.html, between the CALENDAR comments.
   ================================================================== */
(function () {
  // ---- 1) YOUR DATA + SETTINGS --------------------------------------
  // Add events keyed by date "YYYY-MM-DD". Each day holds a list of strings.
  const EVENTS = {
    // "2026-11-09": ["Study session, 25 min", "Focused 82%"],
  };
  const SHOW_WEEKEND = true;        // false = Mon–Fri work week only
  const WEEK_STARTS_MONDAY = false; // true = week runs Mon–Sun

  const el = id => document.getElementById(id);
  const key = d => d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' +
                   String(d.getDate()).padStart(2, '0');
  const midnight = d => new Date(d.getFullYear(), d.getMonth(), d.getDate());
  let shown = null;

  function weekStart(now) {
    const d = midnight(now), monday = WEEK_STARTS_MONDAY || !SHOW_WEEKEND;
    d.setDate(d.getDate() - (monday ? (d.getDay() + 6) % 7 : d.getDay()));
    return d;
  }

  function renderWeek(now) {
    const start = weekStart(now), today = midnight(now), n = SHOW_WEEKEND ? 7 : 5;
    const box = el('week'); box.innerHTML = '';
    box.style.gridTemplateColumns = 'repeat(' + n + ',1fr)';
    for (let i = 0; i < n; i++) {
      const date = new Date(start.getFullYear(), start.getMonth(), start.getDate() + i), k = key(date);
      const card = document.createElement('div');
      card.className = 'day' + (k === key(today) ? ' today' : '') + (date < today ? ' past' : '');
      const h = document.createElement('div'); h.className = 'day-h';
      const name = document.createElement('span');
      name.textContent = date.toLocaleDateString(undefined, { weekday: 'short' }).toUpperCase();
      const num = document.createElement('span'); num.className = 'day-n'; num.textContent = date.getDate();
      h.append(name, num);
      const body = document.createElement('div'); body.className = 'day-body';
      renderDay(date, k, body);
      card.append(h, body); box.append(card);
    }
  }

  // ---- 2) YOUR CODE: fill in one day's box ---------------------------
  // `body` is the empty area under the day's name. Put anything in it,
  // e.g. that day's focus score once you start saving session data.
  function renderDay(date, k, body) {
    const items = EVENTS[k] || [];
    if (!items.length) {
      const e = document.createElement('div'); e.className = 'day-empty'; e.textContent = 'No sessions yet';
      body.append(e); return;
    }
    const ul = document.createElement('ul');
    items.forEach(t => { const li = document.createElement('li'); li.textContent = t; ul.append(li); });
    body.append(ul);
  }

  function tick() {
    const now = new Date();
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || '';
    el('clock').textContent =
      now.toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric' }) + ' · ' +
      now.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' }) + ' · ' +
      tz.replace(/_/g, ' ');
    const sig = key(weekStart(now)) + '|' + key(now);
    if (sig !== shown) { shown = sig; renderWeek(now); } // new day or new week
  }

  // ---- 3) HOOKS: call these from anywhere else on the page -----------
  // addCalendarEvent("2026-11-09", "Focused 82%")   refreshCalendar()
  window.addCalendarEvent = (k, text) => { (EVENTS[k] = EVENTS[k] || []).push(text); shown = null; tick(); };
  window.refreshCalendar = () => { shown = null; tick(); };

  tick();
  setInterval(tick, 1000);
})();
