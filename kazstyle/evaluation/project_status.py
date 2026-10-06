"""Render one current research hub from the activated, verified deployment."""
import html
import json
import os
from datetime import datetime,timezone
from pathlib import Path
from kazstyle.data.corpus import load_manifest,file_hash,write_json
from kazstyle.inference.deployment import FAMILIES
from kazstyle.settings import PROJECT_ROOT,evidence_path


def build():
    root=PROJECT_ROOT;reports=root/'reports';technical=reports/'technical'
    profile=json.loads((root/'configs/deployment.json').read_text(encoding='utf-8'))
    active_frame,active_config=load_manifest(root/profile['dataset'])
    active_summary=evidence_path(profile.get('evidence',{}).get('summary','reports/text_only_v4_repeated'))
    if file_hash(active_summary/'results.json')!=profile['summary_sha256']:raise ValueError('Active deployment summary changed')
    summary=reports/'expanded_v6/repeated'
    frame,config=load_manifest(root/'data/processed/expanded_v6_shared')
    provenance=json.loads((summary/'provenance.json').read_text(encoding='utf-8'))
    if provenance['manifest_sha256']!=config['manifest_sha256']:raise ValueError('Research summary dataset differs')
    results=json.loads((summary/'results.json').read_text(encoding='utf-8'))
    rows=''.join('<tr><td>'+html.escape(name)+'</td><td>'+f"{results[family]['validation_macro_f1_mean']:.4f} ± {results[family]['validation_macro_f1_sample_std']:.4f}"+'</td><td>'+f"{results[family]['test_macro_f1_mean']:.4f} ± {results[family]['test_macro_f1_sample_std']:.4f}"+'</td></tr>' for family,(_,name,_) in FAMILIES.items())
    def link(relative,title):
        target=evidence_path(relative)
        if not target.exists():raise FileNotFoundError(target)
        return '<a href="'+html.escape(Path(os.path.relpath(target,reports)).as_posix(),quote=True)+'">'+html.escape(title)+'</a>'
    links=[('reports/corpus_expansion_20261006/report.html','Корпус: очистка, источники и проверки'),
           (str(summary/'report.html') if not summary.is_absolute() else str(summary.relative_to(root)/'report.html'),'Основное сравнение: все seed и интервалы'),
           ('reports/expanded_v6/source_holdout/report.html','Перенос на другие источники'),
           ('reports/expanded_v6/length_ablation/report.html','Проверка коротких обучающих фрагментов'),
           ('reports/expanded_v6/user_diagnostics/report.html','Заявление, фотосинтез и изменения имён/дат'),
           ('reports/expanded_v6/web_probe/report.html','Повторная проверка 14 известных веб-текстов'),
           ('reports/dissertation_expanded_20261006/manuscript.html','Рабочая рукопись с расширенными результатами'),
           ('reports/ipm_progress_20261006.html','Восемь задач ИПМ: сделано и осталось'),
           ('reports/text_only_v4_repeated/report.html','Стабильная версия v4: предыдущая серия'),
           ('reports/morphology_ablation_v1/report.html','Отдельное сравнение лемматизации на v4')]
    # Write the IPM page first, so every hub link is checked against a real file.
    req=json.loads((reports/'dissertation_audit_20261006/requirements.json').read_text(encoding='utf-8'))['requirements']
    updates=[
        ('В работе','Проверены первичные источники по моделям, корпусам, стилям и диагностике; добавлена карта восьми источников.','Прочитать полные тексты и закончить связный литературный обзор.'),
        ('Рабочий корпус построен','13 973 кандидата после объединения; v6 содержит 1 265 сбалансированных текстовых единиц из 15 источников.','Расширить проверенные жанры; размер сам по себе не подтверждает репрезентативность.'),
        ('Инструменты готовы, ответы людей не получены','Две слепые формы по 250 точным фрагментам; код согласия и разрешения расхождений.','Независимая работа двух разметчиков и согласование спорных случаев.'),
        ('Реализовано и проверено','Очистка, общий вход двух токенизаторов, авторские и текстовые группы. Лемматизация Stanza исследована отдельно на пилоте v4.','Проверить качество преобразований на согласованных метках; известный остаток подписи в одном train-тексте v6 исправить в следующей версии.'),
        ('Сравнение завершено','Шесть вариантов, seed 42/43/44; все 12 заданий v6 завершены, веса восстановлены на CPU.','Окончательное сравнение на корпусе с подтверждёнными метками; точные названия encoder.'),
        ('Работает','Локальный сайт и сервер, пять стилей, шесть моделей; внешний ИИ API не используется.','Зафиксировать демонстрационный сценарий и окончательные требования к интерфейсу.'),
        ('Внутренняя и дополнительная оценки выполнены','Метрики, матрицы ошибок, три seed, групповые интервалы, source-holdout, длина/источник и известные примеры.','Независимый финальный тест на новых источниках с человеческими метками.'),
        ('Частично','Исходная статья сохранена; рабочая рукопись дополнена фактическими результатами v6.','Завершить текст, оформление, заключение и доклад; старые неподтверждённые числа не выдавать за этот эксперимент.')]
    for row,update in zip(req,updates):row[1:]=list(update)
    css='body{font:16px/1.6 system-ui;background:#f3f6fa;color:#183048;margin:0}main{max-width:1160px;margin:30px auto;background:white;padding:34px;border-radius:16px}h1,h2{line-height:1.2}a{color:#175ca8}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border:1px solid #d7e0e8;text-align:left;vertical-align:top}.note{padding:16px;background:#fff2d7}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.cards a{padding:16px;background:#edf4fb;border-radius:10px;text-decoration:none}.muted{color:#5e7081}code{overflow-wrap:anywhere}@media(max-width:650px){main{margin:0;padding:18px}table{display:block;overflow:auto}}'
    ipm_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in req)
    (reports/'ipm_progress_20261006.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>ИПМ: актуальное состояние</title><style>'+css+'</style><main><p><a href="PROJECT_STATUS.html">К результатам</a></p><h1>Восемь задач ИПМ</h1><p>Тема: построение корпуса и системы определения текстового стиля на казахском языке с использованием машинного обучения. Завершённые программные этапы отделены от незавершённой независимой проверки.</p><table><tr><th>Задача</th><th>Статус</th><th>Что подтверждено</th><th>Что осталось</th></tr>'+ipm_rows+'</table></main></html>',encoding='utf-8')
    cards=''.join(link(path,title) for path,title in links)
    hold=json.loads((reports/'expanded_v6/source_holdout/results.json').read_text(encoding='utf-8'))
    heldout=hold['kaz_roberta_last']['test']['macro_f1']
    counts=frame.groupby('split').size().to_dict()
    default=profile['models'][profile['default_model']]['name']
    content=f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KazStyle — результаты и диссертация</title><style>{css}</style></head><body><main>
<h1>KazStyle: результаты и дальнейшая работа</h1><p class="muted">Обновлено 6 октября 2026 · одна точка входа в проект.</p>
<p><strong>Работает локальный классификатор пяти стилей.</strong> Вставить текст: <a href="http://127.0.0.1:8765">стабильный сайт v4</a>. Если сервер выключен, запустить <code>start_site.cmd</code>. Основная модель — {html.escape(default)}, корпус {len(active_frame)} текстов. Внешний ИИ API не используется.</p>
<p>Новое исследование v6: <strong>{len(frame):,} текстов</strong>, по {len(frame)//5} на стиль. Train {counts['train']}, validation {counts['validation']}, test {counts['test']}. URL и сведения об источнике хранятся отдельно от признаков.</p>
<p class="note">V6 пока не заменяет стабильный сайт: выбранная по validation модель ошибается на знакомом школьном заявлении, которое v4 распознаёт правильно. <a href="http://127.0.0.1:8766">Отдельный экспериментальный сайт v6</a> позволяет проверить новые веса. Высокая общая метрика не отменяет эту регрессию.</p>
<div class="note">Это рабочие исследовательские результаты на предварительных метках. Диссертация ещё требует человеческой разметки, независимого финального теста и окончательного оформления.</div>
<h2>Открыть нужное</h2><div class="cards">{cards}</div>
<h2>Завершённое сравнение v6</h2><p>Среднее ± стандартное отклонение при seed 42, 43 и 44. Один фиксированный корпус; это не три независимых тестовых набора. По validation выбрана Kaz-RoBERTa last; решение о готовности к замене стабильного сайта принимается отдельно.</p><table><tr><th>Вариант</th><th>Validation Macro-F1</th><th>Внутренний test Macro-F1</th></tr>{rows}</table>
<p>Отдельно обученная Kaz-RoBERTa на диагностике переноса между источниками: <strong>{heldout:.4f} Macro-F1</strong>. Там другой train и другой состав test; напрямую вычитать это число из основной таблицы нельзя.</p>
<h2>Следующий обязательный этап</h2><p>Два человека независимо проверяют <a href="../data/annotation/expanded_v6_audit_20261006/reviewer_A.html">форму A</a> и <a href="../data/annotation/expanded_v6_audit_20261006/reviewer_B.html">форму B</a> — по 250 фрагментов. Реальных ответов пока нет. Затем согласовываются спорные метки и готовится отдельная финальная проверка новых источников.</p>
<p>Память и принятые решения: <a href="../docs/PROJECT.md">PROJECT.md</a>. Команды и архитектура: <a href="../docs/WORKFLOW.md">WORKFLOW.md</a>. Предыдущие эксперименты сохраняются в архиве; старый отчёт не является текущим результатом.</p>
<details><summary>Ограничения и доказательства</summary><p>Связь источника и класса остаётся сильной; длины классов различаются. Автоматическая проверка не доказывает правильность стилевой метки. В одном train-тексте v6 зафиксирован остаток подписи к фото: исправлен очиститель для следующей версии, замороженные данные не переписаны.</p><p>Сайт: <code>{html.escape(profile['dataset'])}</code><br>Исследование: <code>expanded_v6_shared</code><br>Manifest SHA-256 исследования: <code>{config['manifest_sha256']}</code></p></details>
</main></body></html>'''
    (reports/'PROJECT_STATUS.html').write_text(content,encoding='utf-8')
    write_json(technical/'PROJECT_STATUS.evidence.json',{'created_at':datetime.now(timezone.utc).isoformat(),'manifest_sha256':config['manifest_sha256'],'active_manifest_sha256':active_config['manifest_sha256'],'summary_sha256':file_hash(summary/'results.json'),'deployment_sha256':file_hash(root/'configs/deployment.json'),'human_reviews_received':0})
    print(json.dumps({'hub':str(reports/'PROJECT_STATUS.html'),'documents':len(frame),'default':profile['default_model']}))


if __name__=='__main__':build()
