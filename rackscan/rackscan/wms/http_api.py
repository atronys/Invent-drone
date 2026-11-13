"""Эмулятор склада как HTTP-сервис в стиле 1С.

Адреса и формат сообщений совпадают с теми, что должно отдавать расширение конфигурации
1С:WMS (шаблон модуля — onec/РэкСканHTTPСервис.bsl):

    GET  …/hs/rackscan/v1/ping                         проверка связи
    GET  …/hs/rackscan/v1/cells                        справочник ячеек
    GET  …/hs/rackscan/v1/stock[?cell=СС-ЯЯ-ММ]        остатки по ячейкам
    POST …/hs/rackscan/v1/inventory                    создать задание  {"cells": [...]} | {}
    GET  …/hs/rackscan/v1/inventory/<id>               сличительная ведомость
    POST …/hs/rackscan/v1/inventory/<id>/results       результат по ячейке
    POST …/hs/rackscan/v1/inventory/<id>/close         закрыть задание

Поэтому стенд, отлаженный на эмуляторе, переключается на настоящую 1С сменой одного адреса.
"""
from __future__ import annotations

import base64
import hmac

from flask import Blueprint, Response, jsonify, request

from .emulator import EmulatorBackend


def make_blueprint(wms: EmulatorBackend, user: str | None = None, password: str | None = None) -> Blueprint:
    bp = Blueprint("wms_hs", __name__)

    @bp.before_request
    def _auth():
        if not user:
            return None
        head = request.headers.get("Authorization", "")
        ok = False
        if head.startswith("Basic "):
            try:
                u, _, p = base64.b64decode(head[6:]).decode("utf-8").partition(":")
                # сравнение по байтам: compare_digest не принимает строки с кириллицей
                ok = (hmac.compare_digest(u.encode(), user.encode())
                      and hmac.compare_digest(p.encode(), (password or "").encode()))
            except Exception:
                ok = False
        if not ok:
            return Response("требуется авторизация", 401, {"WWW-Authenticate": 'Basic realm="rackscan"'})
        return None

    def _err(e: Exception, code: int):
        return jsonify({"error": str(e)}), code

    @bp.get("/ping")
    def ping():
        return jsonify({"service": "rackscan", "version": 1, "backend": "emulator"})

    @bp.get("/cells")
    def cells():
        return jsonify({"cells": wms.cells()})

    @bp.get("/stock")
    def stock():
        return jsonify({"stock": [e.to_dict() for e in wms.stock(request.args.get("cell"))]})

    @bp.post("/inventory")
    def create():
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(wms.create_task(body.get("cells")).to_dict()), 201
        except KeyError as e:
            return _err(e, 400)

    @bp.get("/inventory/<task_id>")
    def report(task_id):
        try:
            return jsonify(wms.report(task_id))
        except KeyError as e:
            return _err(e, 404)

    @bp.post("/inventory/<task_id>/results")
    def post(task_id):
        body = request.get_json(silent=True)
        if not body or "address" not in body or "scan_status" not in body:
            return _err(ValueError("нужны поля address и scan_status"), 400)
        try:
            return jsonify(wms.post_result(task_id, body))
        except KeyError as e:
            return _err(e, 404)
        except ValueError as e:
            return _err(e, 409)

    @bp.post("/inventory/<task_id>/close")
    def close(task_id):
        try:
            return jsonify(wms.close_task(task_id))
        except KeyError as e:
            return _err(e, 404)

    return bp
