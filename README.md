# Minimal LLM Service

Учебный HTTP-сервис на FastAPI с последовательным LLM pipeline, валидацией,
ретраями, fallback, TTL-кешем и структурированными JSON-логами.

## Архитектура

```text
api/       HTTP-эндпоинты, схемы и ошибки валидации
services/  бизнес-сценарий и orchestration pipeline
llm/       prompt builder и OpenAI-compatible HTTP-клиент
cache/     потокобезопасный in-memory TTL-кеш
config/    настройки окружения и JSON-логирование
tests/     автоматические сценарии
main.py    сборка приложения и точка входа
```

Путь запроса: API-валидация → нормализация → поиск в кеше → формирование
system/user messages → LLM-вызов → проверка и очистка ответа → сохранение в
кеше → HTTP-ответ.

## Установка

Требуется Python 3.9 или новее.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Укажите в `.env` как минимум:

```dotenv
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

Ключ хранится только в `.env`, который исключён из Git. Клиент использует
OpenAI-compatible `POST /chat/completions`; для совместимого провайдера замените
`LLM_BASE_URL` и `LLM_MODEL`.

## Запуск

```bash
python main.py
```

Сервис доступен на `http://localhost:8000`, Swagger UI — на `/docs`. Проверка
процесса не требует API-ключа:

```bash
curl http://localhost:8000/health
```

Если порт занят, его можно изменить через `PORT` в `.env`.

Ответ:

```json
{"status":"ok"}
```

## API и примеры

Корректный запрос:

```bash
curl -i -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Кратко объясни, что такое LLM"}'
```

Первый ответ имеет `cached: false`, повторный идентичный запрос — `cached: true`:

```json
{
  "status": "ok",
  "answer": "LLM — языковая модель, обученная понимать и генерировать текст.",
  "cached": false,
  "request_id": "0fb94063-8a57-4e07-a596-8e625c692fc2"
}
```

Поле `message` обязательно, должно быть строкой от 1 до 1000 символов. Ошибки
ввода имеют HTTP 400:

```json
{
  "status": "error",
  "error": {
    "code": "validation_error",
    "message": "message: String should have at least 1 character"
  },
  "request_id": "..."
}
```

Если ключ отсутствует или провайдер недоступен, `/chat` не падает и возвращает
HTTP 503:

```json
{
  "status": "fallback",
  "answer": "Сервис временно недоступен, попробуйте позже",
  "cached": false,
  "request_id": "..."
}
```

## Устойчивость и кеш

- Каждый HTTP-вызов ограничен `LLM_TIMEOUT_SECONDS` (по умолчанию 10 секунд).
- Таймауты, сетевые ошибки, HTTP 429 и 5xx повторяются максимум три раза.
- Задержка экспоненциальная: 0.5 секунды, затем 1 секунда.
- Остальные 4xx и повреждённые ответы не повторяются.
- Кеш живёт в памяти процесса 10 минут. Перезапуск очищает его.
- SHA-256 ключ учитывает нормализованный текст, model, temperature и полный
  system prompt.

Все параметры находятся в `.env.example`. `SYSTEM_PROMPT` также можно задать
через окружение. Значение `MAX_MESSAGE_LENGTH` может сделать лимит строже, но
верхняя граница API всегда равна 1000 символов.

## Логи

JSON-логи одновременно пишутся в консоль и `logs/service.log`. В них есть
request ID, этап, номер попытки, длительность, prompt, ответ и события
`cache_hit`/`cache_miss`. Заголовок Authorization и API-ключ не логируются.

Учебное задание требует логировать полный ввод и prompt. Для production-сервиса
перед использованием следует добавить маскирование персональных и секретных
данных.

## Тесты

```bash
pytest -q
```

Тесты используют `httpx.MockTransport` и не расходуют токены, не требуют ключа
и не обращаются во внешнюю сеть. Результаты и ручной чек-лист находятся в
[`test_report.md`](test_report.md).
