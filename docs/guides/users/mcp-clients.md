# Подключение Value Stream к Codex и Claude Desktop

Эта инструкция помогает подключить MCP-сервер Value Stream локально или по HTTP, получить
метрики из подготовленного workspace и проверить построение графиков.
Основные примеры рассчитаны на macOS; особенности Windows приведены ниже.

По умолчанию Value Stream использует **stdio**: клиент запускает дочерний процесс
`valuestream serve-mcp` и общается с ним через стандартные потоки.
Отдельно запускать сервер, открывать порт или указывать HTTP URL не нужно.
`serve-api` — другой интерфейс, его адрес нельзя использовать как MCP endpoint.
Для удалённого подключения используйте **Streamable HTTP** с обязательным
Bearer-токеном; запуск и проверка описаны в разделе 5 ниже.

## 1. Подготовьте окружение и данные

Нужны Python 3.11+, установленный `uv`, локальная копия проекта и workspace
с загруженными агрегатами. Установка описана в
[Deployment](../operations/deployment.md#host-requirements), а создание
тестовых данных и загрузка — в [Getting started](../../tutorials/getting-started.md).
Для MCP требуется extra `ai`; если остальные дополнительные зависимости не
нужны, достаточно выполнить в корне репозитория:

```sh
uv sync --extra ai
```

Синхронизация без других extras может удалить их из окружения. Если нужны также
API, PNG и инструменты разработки, используйте установку с `--all-extras`
из руководства Deployment.

Для примеров ниже замените **оба абсолютных пути**:

| Что | Пример для macOS |
|---|---|
| Исполняемый файл из окружения проекта | `/Users/yourname/projects/value_stream_public/.venv/bin/valuestream` |
| Workspace с каталогом и агрегатами | `/Users/yourname/projects/value_stream_public/examples/fat` |

`examples/fat` подходит, если вы уже загрузили в него данные. На чистой копии
репозитория готовых агрегатов нет: сначала выполните учебный сценарий для
`examples/demo`, затем подставьте его путь. Имена метрик в этих каталогах различаются.

Проверьте установленную команду:

```sh
/Users/yourname/projects/value_stream_public/.venv/bin/valuestream serve-mcp --help
```

В конфигурациях используйте абсолютные пути без `~`, `$HOME` или переменных
оболочки. `command` содержит только путь к программе, каждый аргумент задаётся
отдельной строкой в `args`. Активация виртуального окружения не требуется.

Переменная `LITELLM_LOCAL_MODEL_COST_MAP=True` в примерах отключает загрузку
таблицы цен LiteLLM из сети при старте и позволяет избежать задержек без интернета.
MCP-инструменты не требуют отдельного API-ключа модели или настройки LLM
в workspace. Свободный вопрос разбирает AI-клиент: он выбирает метрику,
параметры и вызывает инструменты аналитики. Встроенного инструмента `chat`
в MCP нет; графики выдаются через `metric_chart_query` или `tile_query`.

## 2. Подключите Codex

Выберите один способ: CLI или ручное редактирование конфигурации.

### Через CLI

Если `codex` доступен в терминале:

```sh
codex mcp add valuestream \
  --env LITELLM_LOCAL_MODEL_COST_MAP=True \
  -- /Users/yourname/projects/value_stream_public/.venv/bin/valuestream \
  serve-mcp /Users/yourname/projects/value_stream_public/examples/fat
```

Проверьте сохранённую запись:

```sh
codex mcp list
```

### Через файл конфигурации

Добавьте в `~/.codex/config.toml` следующий блок. Если `valuestream` уже
существует, измените его, не создавая повторную секцию:

```toml
[mcp_servers.valuestream]
command = "/Users/yourname/projects/value_stream_public/.venv/bin/valuestream"
args = ["serve-mcp", "/Users/yourname/projects/value_stream_public/examples/fat"]
startup_timeout_sec = 60

[mcp_servers.valuestream.env]
LITELLM_LOCAL_MODEL_COST_MAP = "True"
```

Глобальная конфигурация используется desktop-приложением, CLI и IDE extension.
Для настройки только одного доверенного проекта можно использовать его
`.codex/config.toml`.

### Проверьте соединение

Перезапустите клиент после изменения настроек. В Codex CLI откройте `/mcp`:
сервер `valuestream` должен подключиться и показать инструменты.
В desktop-приложении проверьте его в настройках MCP Servers.
Запись в `codex mcp list` подтверждает настройку, но сама по себе не доказывает
успешное соединение — выполните сценарии из раздела 4.

Формат и команды: [официальная документация OpenAI](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).

## 3. Подключите Claude Desktop

Откройте настройки **приложения** Claude Desktop → **Developer** →
**Edit Config**. Файл конфигурации:

| Система | Путь |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

Добавьте `valuestream` внутрь существующего объекта `mcpServers`, сохранив
остальные серверы и настройки. Для нового файла используйте:

```json
{
  "mcpServers": {
    "valuestream": {
      "command": "/Users/yourname/projects/value_stream_public/.venv/bin/valuestream",
      "args": [
        "serve-mcp",
        "/Users/yourname/projects/value_stream_public/examples/fat"
      ],
      "env": {
        "LITELLM_LOCAL_MODEL_COST_MAP": "True"
      }
    }
  }
}
```

JSON не допускает комментариев и запятых после последнего элемента.
Полностью завершите Claude Desktop и откройте его снова.
В **Developer** проверьте статус; в новом диалоге откройте **+ → Connectors**
и найдите `valuestream` и его инструменты. Названия пунктов зависят от версии.
[Инструкция MCP](https://modelcontextprotocol.io/docs/develop/connect-local-servers)
описывает ручную конфигурацию и диагностику.

Это локальная интеграция Claude Desktop, а не удалённый Custom Connector:
такой сервер не становится доступен в claude.ai или Cowork.
[Различия локальных и удалённых подключений](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp).

### Если используется Windows

Укажите установленный executable из `.venv\Scripts\valuestream.exe`.
В JSON обратный слеш удваивается, например:

```json
{
  "command": "C:\\projects\\value_stream_public\\.venv\\Scripts\\valuestream.exe",
  "args": ["serve-mcp", "C:\\projects\\value_stream_public\\examples\\fat"]
}
```

Это фрагмент записи сервера, а не полный файл Claude. В секции TOML Codex
используйте строки в одинарных кавычках, где слеши не нужно экранировать:

```toml
command = 'C:\projects\value_stream_public\.venv\Scripts\valuestream.exe'
args = ['serve-mcp', 'C:\projects\value_stream_public\examples\fat']
```

Windows-конфигурация здесь приведена как шаблон,
её выполнение в рамках этой проверки не тестировалось.

## 4. Выполните тестовые сценарии

Одинаковые запросы подходят обоим клиентам. Попросите использовать именно
MCP-инструменты `valuestream`, чтобы ответ не был получен из файлов или догадок.
Ожидаются **9 инструментов**, два ресурса и два шаблона запросов;
с `--enable-sql` инструментов будет 11. Полный список — в
[API & MCP reference](../../reference/api-and-mcp.md#mcp-tools).

### Проверка каталога и готовности

> Через MCP valuestream вызови workspace_status_tool и metric_list с limit=5.
> Покажи доступные метрики и сообщи, готовы ли агрегаты для запросов.

Успех: возвращается каталог, а необходимые процессоры имеют статус `ready`.
Наличие списка метрик без готовых агрегатов подтверждает связь, но ещё не
готовность данных. Если есть `next_offset`, следующую страницу запрашивают
с этим значением `offset`.

### Метрика и пагинация

Для загруженного `examples/fat`:

> Вызови metric_query для CTR, group_by=["Channel"], limit=2.
> Если next_offset не null, получи следующую страницу с теми же параметрами.
> Покажи CTR по всем каналам и сведения о происхождении данных.

Ожидаемый результат содержит строки, `row_count`, `returned_rows`,
`next_offset` и `provenance`. Для `examples/demo` используйте
`VS_Engagement_Rate` вместо `CTR`; сначала подтвердите имя через `metric_list`.

### График

> Вызови metric_chart_query для CTR с chart_kind="bar", x="Channel",
> y="CTR", group_by=["Channel"], color=null, facet_col=null,
> value_format="percent", render="html". Дай путь к готовому HTML.

Успех: `render.path` указывает на созданный локальный файл. Откройте его
в браузере. HTML содержит Plotly и работает без CDN; встраивание графика
непосредственно в диалог зависит от клиента. Для demo замените и `metric`,
и `y` на `VS_Engagement_Rate`.

### KPI и обработка ошибки

> Через dashboard_list найди KPI-плитку. Вызови kpi_query с её dashboard_id,
> page_id и tile_id. Покажи значение, период, изменение и freshness.

> Вызови metric_query с metric="__missing_metric_smoke_test__".
> Сообщи полученную ошибку, не подставляя другую метрику.

KPI должен возвращать период и доступные данные сравнения. Несуществующая
метрика должна дать `isError: true` и сообщение `unknown metric`.

## 5. Подключение по HTTP

Один процесс обслуживает один workspace, выбранный при запуске. На машине
с данными подготовьте окружение с extra `ai` и загрузите агрегаты, как в
разделе 1. Для HTTP не требуется extra `api`: сервер использует MCP SDK и
Uvicorn из extra `ai`.

Задайте секрет через `VALUESTREAM_MCP_TOKEN` в окружении сервиса. Для локальной
проверки ниже используется учебное значение; при развёртывании замените его
своим секретом.

```sh
export VALUESTREAM_MCP_TOKEN=replace-with-your-secret
uv run valuestream serve-mcp examples/demo --transport http --port 8001
```

Endpoint: `http://127.0.0.1:8001/mcp`. Клиент должен поддерживать Streamable HTTP
и передавать `Authorization: Bearer <token>` в каждом запросе. Этот сервер
не запускает OAuth-вход. Используйте клиент, который позволяет задать токен
или заголовок вручную. Настройки `command`/`args` из разделов 2–3 относятся
к stdio; для HTTP задаётся URL.

### Codex по IP в локальной сети

Домен и HTTPS не обязательны для прямого подключения внутри доверенной
локальной сети. Допустим, IP машины с workspace — `192.168.1.50`.
Замените этот адрес своим IP во всех примерах ниже.

На **сервере**, из каталога проекта:

```sh
uv sync --extra ai
export VALUESTREAM_MCP_TOKEN='replace-with-your-secret'
uv run valuestream serve-mcp /absolute/path/to/workspace --transport http \
  --host 0.0.0.0 --port 8001 --allowed-host 192.168.1.50:8001
```

Workspace должен уже содержать загруженные агрегаты. `0.0.0.0` означает
прослушивание всех сетевых интерфейсов и требует явного `--allowed-host`.
Клиент подключается к `http://192.168.1.50:8001/mcp`; `0.0.0.0` не является
адресом для подключения. Разрешите входящий TCP-порт 8001 в firewall сервера
для нужных машин локальной сети. Bearer-токен и ответы при HTTP передаются
без шифрования; для доступа через интернет используйте HTTPS.

На **машине с Codex desktop** добавьте в `~/.codex/config.toml`:

```toml
[mcp_servers.valuestream_remote]
url = "http://192.168.1.50:8001/mcp"
http_headers = { Authorization = "Bearer replace-with-your-secret" }
```

Подставьте тот же секрет, что в `VALUESTREAM_MCP_TOKEN` на сервере. Здесь он
хранится непосредственно в конфигурационном файле. Если секция
`valuestream_remote` уже существует, измените её вместо добавления повторной.
Перезапустите Codex и откройте новый чат.

Для **Codex CLI** можно хранить токен в окружении процесса клиента:

```sh
export VALUESTREAM_MCP_TOKEN='replace-with-your-secret'
codex mcp add valuestream_remote --url http://192.168.1.50:8001/mcp \
  --bearer-token-env-var VALUESTREAM_MCP_TOKEN
codex
```

Эквивалентный блок конфигурации (выберите его вместо `http_headers`):

```toml
[mcp_servers.valuestream_remote]
url = "http://192.168.1.50:8001/mcp"
bearer_token_env_var = "VALUESTREAM_MCP_TOKEN"
```

Переменная должна быть доступна процессу Codex на клиентской машине.
`export` в отдельном терминале не передаёт её уже запущенному desktop-приложению.
Формат настроек описан в
[официальной документации OpenAI](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).

Для проверки соединения попросите Codex: «Используй `valuestream_remote`:
вызови `metric_list` и покажи доступные метрики». Затем можно запросить:
«Построй график выбранной метрики и верни его в PNG». Если нужны
SQL-инструменты, добавьте `--enable-sql` при запуске сервера; для PNG нужны
extra `viz` и Chrome/Chromium на сервере.

### HTTPS и удалённый адрес

Для доступа через интернет поставьте перед сервером HTTPS reverse proxy.
Например, если публичный адрес — `https://metrics.example.com/mcp`, добавьте
hostname в allowlist и оставьте backend на loopback:

```sh
uv run valuestream serve-mcp examples/demo --transport http \
  --host 127.0.0.1 --port 8001 --allowed-host metrics.example.com
```

Прокси должен направлять `/mcp` на `http://127.0.0.1:8001/mcp`, сохранять
Authorization и передавать Host `metrics.example.com`. Если Host содержит
порт, добавьте соответствующее значение, например
`--allowed-host metrics.example.com:443`. Origin обычно отсутствует у desktop
клиентов; если клиент передаёт его, добавьте точное значение через
`--allowed-origin https://client.example.com`. Это не включает поддержку CORS.

### Проверка настоящим MCP-клиентом

Запустите этот скрипт в другом терминале через `uv run python`, пока сервер
работает. Для удалённой проверки установите `VALUESTREAM_MCP_URL` в LAN URL
из примера выше либо публичный HTTPS URL, а `VALUESTREAM_MCP_TOKEN` — в тот
же секрет, что у сервера.

```python
import asyncio
import os

import httpx2
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main():
    url = os.environ.get("VALUESTREAM_MCP_URL", "http://127.0.0.1:8001/mcp")
    token = os.environ["VALUESTREAM_MCP_TOKEN"]
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token}"}
    ) as http:
        async with streamable_http_client(url, http_client=http) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                print([tool.name for tool in (await client.list_tools()).tools])
                result = await client.call_tool("metric_list", {"limit": 5})
                print(result.model_dump(by_alias=True))


asyncio.run(main())
```

Ожидаются 9 инструментов, либо 11 с `--enable-sql`, и успешный ответ
`metric_list`. Без токена или с неверным токеном сервер возвращает HTTP 401;
недопустимые Host/Origin дают 421/403. Проверка установления MCP-соединения
и вызова инструмента надёжнее простого открытия URL в браузере.

Для удалённого графика используйте `render="spec"` либо `render="png"`
(для PNG на сервере нужен extra `viz` и Chrome/Chromium). HTML-режимы
возвращают путь к файлу **на сервере**; HTTP MCP не публикует такие файлы.
Права всех обладателей токена одинаковы. Для смены workspace перезапустите
сервер с другим путём; для нескольких workspace нужны отдельные процессы,
адреса и токены. Подробности — в [Security](../operations/security.md#http-mcp-authentication).

После ручной проверки завершите сервер через Ctrl+C. В рабочем развёртывании
жизненным циклом HTTP-сервера управляет supervisor.

## 6. Если подключение не работает

| Симптом | Что проверить |
|---|---|
| `ENOENT`, программа не найдена | Абсолютный `command`, наличие executable, установка extra `ai` именно в этом окружении |
| Сервер есть в настройках, но инструментов нет | Перезапуск клиента, включён ли сервер, журнал запуска; в управляемом Claude локальные MCP могут быть запрещены администратором |
| Таймаут запуска | Переменная `LITELLM_LOCAL_MODEL_COST_MAP`, запуск executable напрямую; для Codex параметр `startup_timeout_sec` в примере |
| Ошибка JSON/TOML | Синтаксис, кавычки, повторные секции; не переносите JSON Claude в TOML Codex |
| Метрики видны, но запрос не выполняется | `workspace_status_tool`; загрузите или пересчитайте агрегаты по [runbook](../operations/runbook.md) |
| `unknown metric` | Имена из `metric_list` текущего workspace, включая регистр |
| HTML создан, но не виден в диалоге | Откройте локальный файл из `render.path` в браузере |
| Ошибка `render="png"` | Extra `viz` и Chrome/Chromium; подробности в [справке рендеринга](../../reference/api-and-mcp.md); для проверки используйте HTML |

Логи Claude: `~/Library/Logs/Claude/` на macOS или `%APPDATA%\Claude\logs`
на Windows. Проверяйте `mcp.log` и `mcp-server-valuestream.log`.
[Справка по журналам](https://modelcontextprotocol.io/docs/develop/connect-local-servers#troubleshooting).

Если вручную запустить `serve-mcp` без клиента, процесс будет ждать MCP-сообщений;
это нормально, но не является проверкой соединения. Завершайте такой пробный
запуск через Ctrl+C. Обычным жизненным циклом сервера управляет клиент.
Это описание относится к stdio; HTTP-сервер запускается отдельно и слушает порт.

## Смена workspace и дополнительные возможности

Для смены workspace измените второй элемент `args` и перезапустите клиент.
Для нескольких workspace создайте записи с разными именами и путями.
Изменения YAML-каталога подхватываются сервером автоматически, но новые
определения вычислений могут требовать пересчёта агрегатов.

SQL выключен по умолчанию. Если он нужен, добавьте `"--enable-sql"` в `args`.
Это включает только ограниченные запросы к агрегатам; см.
[правила SQL](../../reference/api-and-mcp.md#governed-sql-rules).

Сервер не изменяет каталог и агрегаты. Экспорт графиков создаёт локальные файлы;
выданный путь не является постоянным хранилищем — нужные результаты сохраните
отдельно. Результаты MCP передаются выбранному AI-клиенту для ответа;
локальный транспорт сам по себе не означает локальное выполнение модели.
Подробнее — [Security](../operations/security.md).

## Что проверено

22 сентября 2026 года на macOS выполнена проверка через Codex app-server
0.155.1 с workspace `examples/fat`: обнаружены 10 инструментов и 2 ресурса,
успешно вызваны каталог метрик, запрос CTR, KPI, HTML-рендеринг и ошибочный запрос.
В том наборе данных KPI CTR за сентябрь 2024 года составил около 2,4004%; это
пример результата конкретного набора, а не обязательное значение проверки.

Шаги Claude Desktop сверены с официальной документацией; запуск именно
в Claude Desktop в этой проверке не выполнялся.
