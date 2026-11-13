"""Подключение к настоящей 1С.

Два пути, оба через штатные механизмы платформы «1С:Предприятие 8»:

1. OneCHttpBackend — HTTP-сервис расширения конфигурации (…/hs/rackscan/v1). Полный обмен:
   задание, остатки, результаты. Сообщения те же, что у эмулятора (http_api.py).
2. ODataStockReader — только чтение остатков через стандартный интерфейс OData
   (…/odata/standard.odata). Доработка 1С не нужна, достаточно опубликовать объекты, но
   имена сущностей и полей зависят от конфигурации — они задаются в настройках.

ВАЖНО: с действующей базой «1С:WMS Логистика. Управление складом» этот код не проверялся —
только с эмулятором и поддельным OData-сервером в тестах. Имена объектов конфигурации
нужно взять из её метаданных (Конфигуратор или …/odata/standard.odata/$metadata).
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from .base import Expected, InventoryTask


class WmsError(RuntimeError):
    pass


class AddressMap:
    """Перевод адреса ячейки между складской системой и стендом.

    В «1С:WMS Логистика. Управление складом» адрес ячейки обычно составляется как
    «префикс кода + помещение + проезд + стеллаж + ярус + позиция» (документация с описанием
    функциональных характеристик, редакция 5, разд. 1.2.1). Стенд знает только стеллаж, ярус
    и место. Шаблон описывает, как адрес выглядит в WMS, например:

        "A-01-{rack:02d}-{level:02d}-{pos:02d}"    префикс и проезд постоянны
        "{rack}.{level}.{pos}"                      другие разделители, без ведущих нулей

    Адреса WMS, не подходящие под шаблон (другая зона, другой проезд), стенду не видны.
    """

    FIELD = re.compile(r"\{(rack|level|pos)(?::[^}]*)?\}")

    def __init__(self, template: str | None = None):
        self.template = template or "{rack:02d}-{level:02d}-{pos:02d}"
        if {m.group(1) for m in self.FIELD.finditer(self.template)} != {"rack", "level", "pos"}:
            raise ValueError("в шаблоне адреса должны быть поля {rack}, {level} и {pos}")
        pattern, pos = "", 0
        for m in self.FIELD.finditer(self.template):
            pattern += re.escape(self.template[pos:m.start()]) + "(?P<" + m.group(1) + r">\d+)"
            pos = m.end()
        self._re = re.compile("^" + pattern + re.escape(self.template[pos:]) + "$")

    def to_wms(self, stand_address: str) -> str:
        r, lv, p = (int(x) for x in stand_address.split("-"))
        return self.template.format(rack=r, level=lv, pos=p)

    def to_stand(self, wms_address: str) -> str | None:
        m = self._re.match(wms_address.strip())
        if not m:
            return None
        return f"{int(m['rack']):02d}-{int(m['level']):02d}-{int(m['pos']):02d}"


class _Http:
    def __init__(self, base_url: str, user: str | None, password: str | None, timeout: float = 5.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self._auth = None
        if user:
            token = base64.b64encode(f"{user}:{password or ''}".encode("utf-8")).decode("ascii")
            self._auth = "Basic " + token

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json; charset=utf-8")
        if self._auth:
            req.add_header("Authorization", self._auth)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8-sig") or "{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:300]
            raise WmsError(f"{method} {path}: HTTP {e.code} {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise WmsError(f"{method} {path}: нет связи с 1С ({e})") from e


class OneCHttpBackend:
    """Реализация WmsBackend поверх HTTP-сервиса 1С (или эмулятора)."""

    def __init__(self, base_url: str, user: str | None = None, password: str | None = None, timeout: float = 5.0,
                 address_template: str | None = None):
        self.http = _Http(base_url, user, password, timeout)
        self.addr = AddressMap(address_template)

    @classmethod
    def from_config(cls, wms_cfg: dict) -> "OneCHttpBackend":
        return cls(wms_cfg["url"], os.environ.get(wms_cfg.get("user_env", "")) or None,
                   os.environ.get(wms_cfg.get("password_env", "")) or None,
                   address_template=wms_cfg.get("address_template"))

    def ping(self) -> dict:
        return self.http.call("GET", "/ping")

    def cells(self) -> list[str]:
        """Ячейки WMS, адрес которых подходит под шаблон стенда (в адресах стенда)."""
        return [a for a in (self.addr.to_stand(c) for c in self.http.call("GET", "/cells")["cells"]) if a]

    def _expected(self, e: dict) -> Expected | None:
        a = self.addr.to_stand(e["address"])
        return None if a is None else Expected(**{**e, "address": a})

    def stock(self, address: str | None = None) -> list[Expected]:
        q = "" if address is None else "?cell=" + urllib.parse.quote(self.addr.to_wms(address))
        return [x for x in (self._expected(e) for e in self.http.call("GET", "/stock" + q)["stock"]) if x]

    def create_task(self, addresses: list[str] | None = None) -> InventoryTask:
        # без явного списка задание создаётся только по ячейкам, которые стенд умеет адресовать
        wanted = self.cells() if addresses is None else addresses
        d = self.http.call("POST", "/inventory", {"cells": [self.addr.to_wms(a) for a in wanted]})
        for ln in d["lines"]:
            ln["address"] = self.addr.to_stand(ln["address"]) or ln["address"]
            for e in ln.get("expected", []):
                e["address"] = ln["address"]
        return InventoryTask.from_dict(d)

    def post_result(self, task_id: str, result: dict) -> dict:
        body = {**result, "address": self.addr.to_wms(result["address"])}
        out = self.http.call("POST", f"/inventory/{urllib.parse.quote(str(task_id))}/results", body)
        out["address"] = result["address"]
        return out

    def _with_summary(self, rep: dict) -> dict:
        """Адреса — в вид стенда; итоги ведомости считаются на стенде, если сервис 1С их не прислал."""
        for ln in rep.get("lines", []):
            ln["address"] = self.addr.to_stand(ln["address"]) or ln["address"]
        if "summary" not in rep and "lines" in rep:
            from .reconcile import summarize
            rep["summary"] = summarize(rep["lines"])
        return rep

    def report(self, task_id: str) -> dict:
        return self._with_summary(self.http.call("GET", f"/inventory/{urllib.parse.quote(str(task_id))}"))

    def close_task(self, task_id: str) -> dict:
        return self._with_summary(
            self.http.call("POST", f"/inventory/{urllib.parse.quote(str(task_id))}/close", {}))


# Пример соответствия для чтения остатков через OData. Имена ниже — ЗАГЛУШКИ: в вашей
# конфигурации регистр остатков и его измерения называются по-своему.
ODATA_MAPPING_EXAMPLE = {
    "entity": "AccumulationRegister_ТоварыВЯчейках/Balance",
    "select": "Ячейка/Description,Контейнер/Code,Номенклатура/Артикул,Номенклатура/Description,КоличествоBalance",
    "expand": "Ячейка,Контейнер,Номенклатура",
    "address": "Ячейка/Description",
    "sscc": "Контейнер/Code",
    "sku": "Номенклатура/Артикул",
    "name": "Номенклатура/Description",
    "qty": "КоличествоBalance",
}


def _dig(row: dict, path: str):
    cur = row
    for part in path.split("/"):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


class ODataStockReader:
    """Чтение остатков по ячейкам через стандартный интерфейс OData платформы 1С."""

    def __init__(self, base_url: str, mapping: dict, user: str | None = None, password: str | None = None,
                 timeout: float = 10.0, address_template: str | None = None):
        self.http = _Http(base_url, user, password, timeout)
        self.mapping = mapping
        self.addr = AddressMap(address_template)

    def stock(self) -> list[Expected]:
        m = self.mapping
        params = {"$format": "json"}
        if m.get("select"):
            params["$select"] = m["select"]
        if m.get("expand"):
            params["$expand"] = m["expand"]
        if m.get("filter"):
            params["$filter"] = m["filter"]
        path = "/" + urllib.parse.quote(m["entity"], safe="/_()") + "?" + urllib.parse.urlencode(
            params, quote_via=urllib.parse.quote, safe="/,$")
        rows = self.http.call("GET", path).get("value", [])
        out = []
        for r in rows:
            address, sscc = _dig(r, m["address"]), _dig(r, m["sscc"])
            address = self.addr.to_stand(str(address)) if address else None
            if not address or not sscc:
                continue
            qty = _dig(r, m.get("qty", "")) if m.get("qty") else 1
            out.append(Expected(address, str(sscc).strip(),
                                str(_dig(r, m.get("sku", "")) or ""), str(_dig(r, m.get("name", "")) or ""),
                                float(qty or 0)))
        return out


# Слова, по которым в составе OData ищутся объекты, нужные стенду. Имена объектов конфигурации
# «1С:WMS Логистика. Управление складом» в открытой документации не приведены, поэтому при
# первом подключении их находят этой функцией, а не угадывают.
DISCOVER_KEYWORDS = {
    "ячейки": ("Ячейк", "Ячеек", "Топологи", "Адрес"),
    "контейнеры": ("Контейнер", "SSCC", "ГрузовыеМеста", "Паллет"),
    "остатки": ("Остатк", "ТоварыВ", "ТоварыНа", "Запас"),
    "инвентаризация": ("Инвентаризац", "Пересчет", "Пересчёт"),
    "номенклатура": ("Номенклатур",),
}


def parse_odata_metadata(xml_text: str) -> dict[str, list[str]]:
    """Имена сущностей и их свойств из документа $metadata стандартного интерфейса OData 1С."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_text.lstrip("﻿"))
    out: dict[str, list[str]] = {}
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] == "EntityType" and el.get("Name"):
            out[el.get("Name")] = [p.get("Name") for p in el
                                   if p.tag.rsplit("}", 1)[-1] in ("Property", "NavigationProperty") and p.get("Name")]
    return out


def discover(base_url: str, user: str | None = None, password: str | None = None, timeout: float = 20.0) -> dict:
    """Прочитать $metadata базы 1С и отобрать объекты, похожие на нужные стенду.

    Возвращает {группа: [{"entity": имя, "properties": [...]}], "всего сущностей": N}. По этому
    списку заполняется соответствие для ODataStockReader и шаблон HTTP-сервиса в каталоге onec.
    """
    http = _Http(base_url, user, password, timeout)
    req = urllib.request.Request(http.base + "/$metadata")
    if http._auth:
        req.add_header("Authorization", http._auth)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            entities = parse_odata_metadata(resp.read().decode("utf-8-sig"))
    except urllib.error.HTTPError as e:
        raise WmsError(f"$metadata: HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise WmsError(f"$metadata: нет связи с 1С ({e})") from e
    found: dict = {k: [] for k in DISCOVER_KEYWORDS}
    for name, props in sorted(entities.items()):
        for group, words in DISCOVER_KEYWORDS.items():
            if any(w.lower() in name.lower() for w in words):
                found[group].append({"entity": name, "properties": props})
    found["всего сущностей"] = len(entities)
    return found
