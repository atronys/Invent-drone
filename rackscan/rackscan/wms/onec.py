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
import urllib.error
import urllib.parse
import urllib.request

from .base import Expected, InventoryTask


class WmsError(RuntimeError):
    pass


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

    def __init__(self, base_url: str, user: str | None = None, password: str | None = None, timeout: float = 5.0):
        self.http = _Http(base_url, user, password, timeout)

    @classmethod
    def from_config(cls, wms_cfg: dict) -> "OneCHttpBackend":
        return cls(wms_cfg["url"], os.environ.get(wms_cfg.get("user_env", "")) or None,
                   os.environ.get(wms_cfg.get("password_env", "")) or None)

    def ping(self) -> dict:
        return self.http.call("GET", "/ping")

    def cells(self) -> list[str]:
        return list(self.http.call("GET", "/cells")["cells"])

    def stock(self, address: str | None = None) -> list[Expected]:
        q = "" if address is None else "?cell=" + urllib.parse.quote(address)
        return [Expected(**e) for e in self.http.call("GET", "/stock" + q)["stock"]]

    def create_task(self, addresses: list[str] | None = None) -> InventoryTask:
        body = {} if addresses is None else {"cells": addresses}
        return InventoryTask.from_dict(self.http.call("POST", "/inventory", body))

    def post_result(self, task_id: str, result: dict) -> dict:
        return self.http.call("POST", f"/inventory/{urllib.parse.quote(str(task_id))}/results", result)

    def report(self, task_id: str) -> dict:
        return self.http.call("GET", f"/inventory/{urllib.parse.quote(str(task_id))}")

    def close_task(self, task_id: str) -> dict:
        return self.http.call("POST", f"/inventory/{urllib.parse.quote(str(task_id))}/close", {})


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
                 timeout: float = 10.0):
        self.http = _Http(base_url, user, password, timeout)
        self.mapping = mapping

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
            if not address or not sscc:
                continue
            qty = _dig(r, m.get("qty", "")) if m.get("qty") else 1
            out.append(Expected(str(address).strip(), str(sscc).strip(),
                                str(_dig(r, m.get("sku", "")) or ""), str(_dig(r, m.get("name", "")) or ""),
                                float(qty or 0)))
        return out
