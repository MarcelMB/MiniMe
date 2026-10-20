// VIEW and CMP (run names or null) are defined inline by page.render
const $ = s => document.querySelector(s);
function same(a, b){ a = a.trim(); b = b.trim();
  return a === b || (a !== '' && b !== '' && !isNaN(a) && !isNaN(b) && Number(a) === Number(b)); }
function mark(form){ form.querySelectorAll('input[data-base]').forEach(i => {
  i.classList.toggle('changed', !same(i.value, i.dataset.base)); }); }
document.querySelectorAll('form').forEach(f => { mark(f); f.addEventListener('input', () => mark(f)); });
async function post(url, form){
  const r = await fetch(url, {method:'POST', body:new URLSearchParams(new FormData(form))});
  const j = await r.json(); if(!r.ok) throw new Error(j.error); return j; }
function say(text, err){ const s = $('#status'); s.textContent = text; s.className = err ? 'error' : ''; }
function busy(b){ document.querySelectorAll('button').forEach(x => x.disabled = b); }
async function poll(name, reload){
  const st = await (await fetch('/api/status')).json();
  if(st.state === 'running'){ busy(true);
    say(`running ${st.name}: ${st.stage || '...'} (${st.elapsed} s)`); setTimeout(() => poll(name, reload), 400); return; }
  busy(false);
  if(st.state === 'error'){ say(`${st.name} failed after ${st.elapsed} s: ${st.error}`, true); return; }
  if(reload) location.href = '/?run=' + encodeURIComponent(st.name) + (VIEW && VIEW !== st.name ? '&cmp=' + encodeURIComponent(VIEW) : '');
}
$('#runform').addEventListener('submit', async e => { e.preventDefault();
  try { const j = await post('/api/run', e.target); say('started ' + j.name); poll(j.name, true); }
  catch(err){ say(err.message, true); } });
$('#atform').addEventListener('submit', async e => { e.preventDefault();
  try { const j = await post('/api/autotune', e.target); say('auto-tune started'); poll(j.name, true); }
  catch(err){ say(err.message, true); } });
$('#saveform').addEventListener('submit', async e => { e.preventDefault();
  try { const j = await post('/api/save', e.target);
    $('#saved').textContent = 'saved: ' + j.path + '  ->  cafast run RAW --params ' + j.path; }
  catch(err){ $('#saved').textContent = err.message; } });
$('#pvform').addEventListener('submit', async e => { e.preventDefault(); const b = e.submitter; b.disabled = true;
  $('#pvmsg').textContent = 'reading clip ...';
  try { await post('/api/preview', e.target); $('#pvmsg').textContent = '';
    $('#pvimg').src = '/fig/_shared/preview.png?t=' + Date.now(); $('#pvimg').hidden = false; }
  catch(err){ $('#pvmsg').textContent = err.message; } b.disabled = false; });
document.querySelectorAll('#units tbody tr').forEach(tr => tr.addEventListener('click', () => {
  $('#unitimg').src = '/unit/' + encodeURIComponent(VIEW) + '/' + tr.dataset.k + '.png'; $('#unitimg').hidden = false; }));
document.querySelectorAll('img.thumb').forEach(im => im.addEventListener('click', () => {
  $('#unitimg').src = '/unit/' + encodeURIComponent(VIEW) + '/' + im.dataset.k + '.png'; $('#unitimg').hidden = false;
  $('#unitimg').scrollIntoView({block: 'nearest'}); }));
document.querySelectorAll('#units th').forEach((th, c) => th.addEventListener('click', () => {
  const tb = $('#units tbody'), rows = [...tb.rows], asc = th.dataset.asc !== '1'; th.dataset.asc = asc ? '1' : '0';
  const v = r => { const t = r.cells[c].textContent; const n = parseFloat(t); return isNaN(n) ? t : n; };
  rows.sort((a, b) => (v(a) > v(b) ? 1 : v(a) < v(b) ? -1 : 0) * (asc ? 1 : -1)); rows.forEach(r => tb.appendChild(r)); }));
fetch('/api/status').then(r => r.json()).then(st => { if(st.state === 'running') poll(st.name, true); });
