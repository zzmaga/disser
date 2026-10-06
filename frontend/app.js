const $ = id => document.getElementById(id);
let ready = false;
let busy = false;
let latest = null;

function refreshButtons() {
  const disabled = busy || !ready || !$('text').value.trim();
  $('classify').disabled = disabled;
  $('compare').disabled = disabled;
  $('model').disabled = busy || !ready;
  $('clear').disabled = busy;
  $('text').readOnly = busy;
}

function updateCount() {
  const text = $('text').value;
  const words = text.trim() ? text.trim().split(/\s+/u).length : 0;
  $('count').textContent = `${words} слов · ${text.length.toLocaleString('ru-RU')} символов`;
  refreshButtons();
}

function setView(view) {
  for (const id of ['empty', 'loading', 'error', 'result']) $(id).hidden = id !== view;
}

function showResult(data) {
  latest = data;
  const comparing = data.results.length > 1;
  const styles = new Set(data.results.map(r => r.style));
  $('result-caption').textContent = comparing ? 'Ответы моделей' : 'Предполагаемый стиль';
  $('style').textContent = comparing && styles.size > 1 ? 'Ответы различаются' : data.results[0].style_ru;
  $('chosen-model').textContent = comparing ? (styles.size === 1 ? `Все ${data.results.length} моделей выбрали один стиль` : 'Сравните ответы ниже') : data.results[0].model_name;
  $('comparison').replaceChildren();
  $('comparison').hidden = !comparing;
  if (comparing) for (const row of data.results) {
    const div = document.createElement('div'); div.className = 'comparison-row';
    const name = document.createElement('span'); name.textContent = row.model_name;
    const style = document.createElement('strong'); style.textContent = row.style_ru;
    div.append(name, style); $('comparison').append(div);
  }
  $('warnings').replaceChildren();
  for (const warning of data.warnings) {
    const p = document.createElement('p'); p.className = 'warning'; p.textContent = warning;
    $('warnings').append(p);
  }
  $('excerpt').textContent = data.excerpt;
  $('excerpt-count').textContent = `· ${data.analyzed_words} слов`;
  $('excerpt-note').textContent = data.partial_text
    ? 'Текст длинный: для этой проверки модель прочитала только показанный фрагмент. Оставшаяся часть в ответе не учитывается.'
    : 'Модель прочитала весь текст после удаления ссылок и лишних пробелов.';
  $('elapsed').textContent = `${(data.elapsed_ms / 1000).toFixed(1)} с`;
  $('copy').textContent = 'Скопировать результат';
  setView('result');
}

async function classify(compare = false) {
  if (busy || !ready || !$('text').value.trim()) return;
  busy = true; latest = null; refreshButtons();
  $('output').setAttribute('aria-busy', 'true'); $('elapsed').textContent = '';
  $('loading-label').textContent = compare ? 'Сравниваем ответы моделей…' : 'Модель читает текст…';
  setView('loading');
  try {
    const response = await fetch('/api/classify', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: $('text').value, model: $('model').value, compare})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Сервер не смог обработать текст.');
    showResult(data);
  } catch (error) {
    $('error').textContent = error.message === 'Failed to fetch'
      ? 'Нет связи с сервером. Запустите start_site.cmd и повторите попытку.' : error.message;
    setView('error');
  } finally {
    busy = false; $('output').setAttribute('aria-busy', 'false'); refreshButtons();
  }
}

$('text').addEventListener('input', () => { updateCount(); if (!busy) { latest = null; setView('empty'); $('elapsed').textContent = ''; } });
$('text').addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); classify(); } });
$('model').addEventListener('change', () => { latest = null; setView('empty'); $('elapsed').textContent = ''; });
$('classify').addEventListener('click', () => classify());
$('compare').addEventListener('click', () => classify(true));
$('clear').addEventListener('click', () => { $('text').value = ''; latest = null; updateCount(); setView('empty'); $('elapsed').textContent = ''; $('text').focus(); });
$('copy').addEventListener('click', async () => {
  if (!latest) return;
  const output = ['Текст:', $('text').value, '', ...latest.results.map(r => `${r.model_name}: ${r.style_ru}`), '', ...latest.warnings,
    '', 'Проанализированный фрагмент:', latest.excerpt].join('\n');
  try { await navigator.clipboard.writeText(output); $('copy').textContent = 'Скопировано'; }
  catch { $('copy').textContent = 'Копирование недоступно в этом браузере'; }
});

async function initialize() {
  try {
    const response = await fetch('/api/models');
    if (!response.ok) throw new Error('Сервер недоступен');
    const data = await response.json(); $('model').replaceChildren();
    for (const model of data.models) {
      const option = document.createElement('option'); option.value = model.id; option.textContent = model.name;
      $('model').append(option);
    }
    $('model').value = data.default; ready = true;
    $('style-scope').textContent = `Доступные стили: ${data.styles.join(', ')}. Версия корпуса: ${data.dataset}. Вход модели — только очищенный текст.`;
    $('connection').textContent = '● Сервер подключён'; refreshButtons();
  } catch {
    $('connection').textContent = 'Сервер недоступен';
    $('error').textContent = 'Не удалось подключиться. Запустите start_site.cmd и обновите страницу.'; setView('error');
  }
}
updateCount(); initialize();
