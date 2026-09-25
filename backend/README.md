# backend — сервер сверки (Ф3, задачи DATA-03…05)

Пока пусто: до пилота сверка делается CLI `python -m invent_core reconcile` (логика уже в `core`).

План (on-prem, ADR-9):
- **FastAPI + PostgreSQL**, `docker compose up`;
- загрузка: раскладка склада (JSON `invent_core.layout`), снимок WMS (CSV → позже API/1С), наблюдения вылета;
- выдача: отчёт сверки (статусы `invent_core.reconcile.Status`), экспорт в Excel, кропы этикеток;
- веб-панель: фасад ряда с цветами статусов, фильтр расхождений, список на перескан.

Черновик API:
```
POST /warehouses                 раскладка
POST /warehouses/{id}/wms        снимок WMS (CSV)
POST /missions/{id}/observations наблюдения вылета
GET  /missions/{id}/report       отчёт (JSON/CSV/XLSX)
GET  /missions/{id}/rescan       адреса UNREADABLE/NOT_SCANNED -> план перескана
```
