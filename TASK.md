# laya-cli — задача на реалізацію

## Навіщо

Зараз щоразу, коли треба класифікувати список кандидатів (текстів/футажу/тікетів) через
[Laya](https://github.com/NandhaKishorM/laya) (`pip install laya`, non-generative typed-decision
модель), пишеться одноразовий python-скрипт: завантажити модель, прогнати цикл `agent.predict()`,
зібрати результат, відсортувати/відфільтрувати, відформатувати звіт. Реальний кейс, з якого
виросла ця задача: відбір стокового відео (Pexels, через `px` CLI) під відео-епізод — 82
кандидати, кожен з коротким текстовим описом (`state`), треба відфільтрувати релевантні.

Мета — винести цю повторювану частину в окремий CLI (`laya-cli`), який приймає JSONL на stdin
(кожен рядок — кандидат з текстовим полем), питання в JSON-файлі, і віддає JSONL з доданими
відповідями на stdout. Стрімовий Unix-style інструмент, що ставиться в пайплайн з будь-чим,
що вміє JSONL (наприклад `px videos --state`).

```bash
px videos --queries "..." --state --dedupe keep-first \
  | laya-cli classify --questions questions.json \
  | laya-cli filter --where "on_topic>=0.4" --sort -on_topic \
  > scored.jsonl
```

Референс по тому, як правильно юзати Laya (питання, чекпоінти, калібрація, пастки) —
`~/.claude/skills/laya-integration/SKILL.md` на цій машині. Реалізація мусить слідувати
принципам звідти, не вигадувати заново.

## Технічна база

- Python, `pip install laya` як залежність (тягне torch/transformers — важке, тому окремий
  проєкт, а не додаток до легких CLI типу px).
- `laya.load("convaiinnovations/laya", subfolder=...)` для конкретного чекпоінта,
  `laya.Router(preload=True)` коли треба авто-вибір мови/чекпоінта.
- Модель вантажиться **один раз** на весь запуск CLI, не на кожен рядок вводу. Перший виклик
  після завантаження — прогрів (throwaway `predict`) перед основним циклом, це у skill описано
  explicitно ("Warm up").

## Команди

### `laya-cli classify --questions <file.json> [--state-field state] [--model ...] [--lang ...] [--device ...]`

- Читає JSONL з stdin, рядок за рядком.
- Для кожного рядка бере текст з поля `--state-field` (дефолт `state`) **як є, без жодних
  домішок** — не конкатенувати нічого свого до цього тексту. Урок з реального кейсу: коли
  джерело (px) саме собі домішало `(photographer: Ім'я)` в текст, on_topic-скор Laya падав
  на 0.1-0.3 на тих самих кадрах (виміряно). Цей інструмент не повторює ту помилку і не додає
  нічого до `state` мовчки. Якщо хтось хоче domішати контекст — окремий явний флаг
  `--prepend-field <name>` (не дефолт, задокументувати ризик у `--help`).
- `--questions file.json` — той самий словник, що приймає `agent.predict(state, questions)`
  (типи `choice`/`score`/`noul`, кожне питання — `type`, `instructions`, `criteria`). Формат
  identичний до того, що описано в SKILL.md, ніякого власного DSL зверху.
- Виводить у stdout той самий вхідний об'єкт + додані поля з відповідей: для `choice` —
  `{question_id}` = обране значення, `{question_id}_p` = ймовірність обраного,
  `{question_id}_probs` = весь розподіл (опційно, за `--full-probs`, дефолт вимкнено щоб не
  роздувати рядки); для `score` — `{question_id}_score`; для `noul` — `{question_id}` = P(true).
  `{question_id}_confidence` завжди додається.
- `--model` — hf repo id, дефолт `convaiinnovations/laya` (English checkpoint).
- `--subfolder` — `multilingual` / `typed-decisions` / порожньо (bundle).
- `--lang` — примусова мова для `laya.Router` (якщо заданий `--router`); без цього флагу
  Router сам детектить мову, а на мовах поза {en,fr,de,es,pt,it,nl} мовчки лягає на English
  checkpoint — тому CLI мусить **попереджати в stderr**, коли `--router` активний і `--lang`
  не заданий, з посиланням на це обмеження (SKILL.md розділ "Router").
- `--device` — `cpu`/`mps`/`cuda`, прокидається напряму в `laya.load(device=...)`.
- Concurrency: сам CLI однопотоковий по дизайну (один процес, послідовний цикл по stdin) —
  жодних тредів/asyncio, це не потрібно й лише ускладнить lock-семантику з skill
  ("guard predict with a lock, one GPU serves one forward pass at a time").

### `laya-cli filter --where "<expr>" [--sort -field,+field2]`

- Легкий пост-фільтр/сортувальник над JSONL з stdout команди `classify` (або будь-яким JSONL
  з числовими полями). `--where` — прості порівняння (`field>=0.4`, `field==value`,
  `field!=value`), можна кілька через кому (AND). `--sort` — список полів, `-` префікс = desc.
- Це заміняє ручний python-скрипт сортування/фільтрації, який я сам писав під кожен кейс
  (`build_shortlist*.py`).
- Не вигадувати повноцінну query-мову (jq вже існує для складного) — тільки прості
  порівняння, найчастіший випадок.

### `laya-cli questions <preset>`

- Друкує в stdout готовий `questions.json` з вбудованих пресетів Laya:
  `triage`, `email`, `guard`, `moderation`, `router` — прямий прокид
  `laya.triage_questions()` / `laya.email_questions()` / `laya.guard_questions()` /
  `laya.moderation_questions()` / `laya.router_questions()` у JSON, щоб було з чого стартувати
  й відредагувати руками, а не читати python-докстрінги.

### `laya-cli evaluate --questions <file.json> --labeled <file.jsonl> --field <question_id>`

- Реалізація розділу skill "Evaluate before shipping": вхід — JSONL з `state` +
  колонкою правильної відповіді (`--label-field`, дефолт `label`), прогонити класифікацію,
  порахувати accuracy per question, скільки % проходить поріг (`--threshold`, дефолт 0.5) і
  скільки з тих, що пройшли, правильні (precision @ threshold). Вивід — короткий текстовий
  звіт у stdout (не JSONL), саме той, що треба показати юзеру перед тим як довіряти цифрам
  ("Tell the user the measured numbers and the escalation rate, not just that it works" —
  пряма цитата зі skill).
- Це не про калібрацію температури (то окремо, нижче) — це про "чи взагалі working" на
  реальних прикладах юзера, мінімальний обов'язковий крок перед тим, як шортлист комусь
  показати.

## Explicitly НЕ в MVP (не робити, поки не попросять)

- Калібрація температури (`agent.temperature` fit під NLL) — skill описує процедуру, це
  окрема команда `laya-cli calibrate`, але вона складніша (потребує NLL-оптимізацію,
  тестовий split) — робити тільки якщо `evaluate` покаже, що calibration реально потрібна.
  Не будувати наперед.
- HTTP-сервер / sidecar-режим (skill згадує FastAPI-приклад для не-python стеків) — YAGNI,
  поки немає не-python консюмера. `classify` як stdin/stdout CLI покриває поточний кейс
  (px | laya-cli).
- Власна DB/кеш результатів між запусками — CLI stateless, вхід/вихід через JSONL-файли,
  версіонування кешу — задача того, хто його викликає (git, файли в проєкті).
- Multi-model ensemble / голосування кількох чекпоінтів — не було в жодному реальному кейсі.

## Acceptance criteria

- [ ] `laya-cli classify` не додає нічого до тексту `state` за замовчуванням (перевірити на
      прикладі: вхід і те, що фактично йде в `agent.predict`, ідентичні substring).
- [ ] Модель вантажиться рівно раз за запуск (лог/таймер підтверджує — не 82 завантаження на
      82 рядки).
- [ ] `classify | filter --where "on_topic>=0.4" --sort -on_topic` відтворює той самий
      результат, що я зараз отримую вручну через `build_shortlist*.py` на тому ж вхідному
      наборі.
- [ ] `questions triage` (і решта пресетів) видає валідний JSON, який приймає `classify`
      без правок.
- [ ] `evaluate` на 20-50 розмічених прикладів дає числа (accuracy, поріг-precision), а не
      просто "ok".
- [ ] Без мережі (з прогрітим HF-кешем, `HF_HUB_OFFLINE=1`) CLI працює — інструмент має
      прокидати цю env-змінну прозоро, не блокувати офлайн-роботу.
- [ ] `--help` на кожній команді — з прикладом виклику, як у px (уже показав, що це реально
      корисно при перевірці).
