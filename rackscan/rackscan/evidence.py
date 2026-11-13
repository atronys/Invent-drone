"""Фото-доказательства и журнал событий.

На каждую ячейку сохраняются: исходный кадр (JPEG без пометок), кадр с пометками,
паспорт результата (JSON) с контрольной суммой SHA-256 исходного кадра. Журнал — файл
JSON Lines, только дозапись: каждое событие — одна строка с номером и временем.
Этого достаточно, чтобы позже проверить любое решение стенда по первичным данным.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def _now() -> datetime:
    return datetime.now().astimezone()


class EvidenceStore:
    def __init__(self, root: str | Path, session: str | None = None):
        self.root = Path(root)
        self.session = session or _now().strftime("%Y%m%d-%H%M%S")
        self.dir = self.root / self.session
        self.dir.mkdir(parents=True, exist_ok=True)
        self._journal = self.dir / "journal.jsonl"
        self._seq = 0
        self._lock = threading.Lock()

    def journal(self, kind: str, **data) -> dict:
        """Добавить событие в журнал."""
        with self._lock:
            self._seq += 1
            rec = {"seq": self._seq, "ts": _now().isoformat(timespec="milliseconds"), "kind": kind, **data}
            with self._journal.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=_jsonable) + "\n")
        return rec

    def save_result(self, result: dict, raw: np.ndarray | None, annotated: np.ndarray | None = None) -> dict:
        """Сохранить кадры и паспорт результата. Возвращает result, дополненный путями и суммой."""
        ts = _now()
        stem = f"{result['address']}_{result['scan_status']}_{ts.strftime('%H%M%S')}"
        out = dict(result)
        out["ts"] = ts.isoformat(timespec="seconds")
        if raw is not None:
            ok, buf = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                (self.dir / f"{stem}.jpg").write_bytes(buf.tobytes())
                out["photo"] = f"{self.session}/{stem}.jpg"
                out["photo_sha256"] = hashlib.sha256(buf.tobytes()).hexdigest()
                out["photo_size"] = [int(raw.shape[1]), int(raw.shape[0])]
        if annotated is not None:
            cv2.imwrite(str(self.dir / f"{stem}_annot.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out["photo_annotated"] = f"{self.session}/{stem}_annot.jpg"
        (self.dir / f"{stem}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2, default=_jsonable) + "\n", encoding="utf-8", newline="\n")
        self.journal("result", address=out["address"], scan_status=out["scan_status"],
                     codes=out.get("codes", []), flags=out.get("flags", []), photo=out.get("photo", ""))
        return out

    def read_journal(self, last: int = 50) -> list[dict]:
        if not self._journal.exists():
            return []
        lines = self._journal.read_text(encoding="utf-8").splitlines()
        return [json.loads(x) for x in lines[-last:]]


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
