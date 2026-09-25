# INVENT_DRONE

Автономный дрон для инвентаризации паллетных складов — российский аналог Corvus One.
Ночью облетает стеллажи, читает этикетки паллет (SSCC / GS1-128, DataMatrix), привязывает их
к адресам ячеек, сверяет с WMS/1С и выдаёт отчёт о расхождениях: пересорт, пустые ячейки,
«потерянные» паллеты. Горизонт — 2 года (до защиты ВКР в июне 2028).

## Документы — читать по порядку

| # | Документ | О чём |
|---|---|---|
| 00 | [Разбор литературы](docs/00_literature_review.md) | 4 источника из `books/`: что взять, критика, план чтения, чего не хватает |
| 01 | [Продукт и рынок](docs/01_product_and_market.md) | Corvus как референс, боль, сегменты РФ, юнит-экономика, регуляторика, гранты, custdev |
| 02 | [Архитектура](docs/02_architecture.md) | Система, ADR, оптика этикеток, планирование, безопасность |
| 03 | [Дорожная карта](docs/03_roadmap.md) | Фазы Ф0–Ф5 с датами, Go/No-go, KPI, критический путь, анти-цели |
| 04 | [**Backlog**](docs/04_backlog.md) | 120 задач: ID, приоритет, оценка, фаза, критерий готовности; ✅/🔧 — что уже сделано. **Начинать отсюда** |
| 05 | [Железо](docs/05_hardware_bom.md) | Состав прототипа v1 (РФ), лаборатория |
| 06 | [Риски](docs/06_risks.md) | Реестр рисков и мер |
| 07 | [План ВКР](docs/07_thesis_plan.md) | Тема, новизна, структура, эксперименты, публикации |
| — | [Custdev](docs/custdev/) | Сценарий интервью, чек-лист обследования склада, шаблон контактов |
| — | [Эксперимент LOC-08](research/landmark_ekf/README.md) | EKF-локализация по этикеткам: результаты и выводы |

## Структура репозитория

```
core/          ядро на Python (без ROS): модель склада, планировщик облёта и перескана, камера с дисторсией,
               кадры/повороты, 6-DoF привязка этикетки к ячейке, PnP, GS1, сверка с WMS
sim/           генератор мира Gazebo + модель дрона x500_invent + виртуальный облёт всего конвейера без Gazebo
research/      эксперименты для диплома: landmark_ekf (локализация по этикеткам), virtual_flight (оптика, смаз, поза)
tools/         calibrate_camera.py — калибровка по шахматке -> ROS camera_info
ros2_ws/src/   ROS 2 Jazzy: invent_msgs, invent_mission (миссия, TF, безопасность), invent_perception
               (label_reader, scan_recorder), invent_localization (EKF по этикеткам), invent_bringup
scripts/       setup_ubuntu.sh — вся среда одной командой (ROS 2, PX4 SITL, Gazebo, XRCE-агент)
docker/        slambook — изолированная среда для упражнений из книги Gao & Zhang
backend/ ml/ hardware/   заготовки на Ф2–Ф3 (README с планом)
docs/          документация; docs/notes — конспекты и журнал экспериментов
books/         литература (не в git)
```

## Быстрый старт

Ядро работает на Windows и Linux, Python ≥ 3.10, без зависимостей:

```bash
cd core
python -m unittest discover -s tests                        # тесты ядра (PnP-тесты нужны numpy+opencv)
python -m invent_core plan examples/warehouse_demo.json      # план облёта: вылеты, ячеек/час
python -m invent_core plan examples/warehouse_demo.json --only rescan.csv   # перескан адресов
python -m invent_core optics --distance 1.5 --hfov 29.4 --vfov 22.2   # читается ли этикетка
python -m invent_core label-pose photo.jpg --camera cam.yaml # дистанция до этикетки по фото (PnP)
python -m invent_core make-layout my_wh.json --aisles 4 --bays 20 --levels 5
python -m invent_core reconcile wms.csv observations.csv -o report.csv
cd .. && python research/landmark_ekf/ekf_sim.py --seeds 10 --plot   # эксперимент LOC-08
```

Окружение для разработки: `python -m venv .venv` и `.venv/Scripts/python -m pip install -r requirements-dev.txt`.
Тесты (их же гоняет хук перед коммитом): `core/tests`, `sim/tests`, `tools/tests`, `research/tests`;
медленные (виртуальный облёт 12 Мп) — с `INVENT_SLOW_TESTS=1`.

Симуляция и ROS 2 — на Ubuntu 24.04: `bash scripts/setup_ubuntu.sh`, затем [sim/README.md](sim/README.md).

## Принципы

1. **Склад — структурированная среда.** Маршрут строится по плану склада, восприятие нужно для
   безопасности и коррекции позы, а не для исследования неизвестной среды.
2. **Симуляция раньше железа**; каждая фаза заканчивается демо с цифрами.
3. **Ядро без ROS** — быстрые тесты и переиспользование на сервере; узлы ROS — тонкие обёртки.
4. **Custdev с первого месяца**; без склада-партнёра к марту 2027 железо не строим.
5. **Диплом — побочный продукт стартапа.**
6. **Лицензии:** никакого GPL/AGPL в том, что поставляется клиенту (ADR-5, ADR-7).
