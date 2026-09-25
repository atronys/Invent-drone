"""Запись вылета (PER-07): LabelRead + MissionStatus -> CSV наблюдений для сверки.

Логика — `invent_core.reconcile.SortieRecorder`; узел только подписывается и сохраняет файл,
когда миссия завершилась (DONE/ABORT) или узел остановлен.
Дальше: python -m invent_core reconcile wms.csv <файл> -o report.csv

Запуск: ros2 run invent_perception scan_recorder --ros-args -p output_dir:=$HOME/invent_runs
"""
import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node

from invent_core.reconcile import SortieRecorder, write_observations_csv
from invent_msgs.msg import LabelRead, MissionStatus


class ScanRecorder(Node):
    def __init__(self):
        super().__init__("scan_recorder")
        self.declare_parameter("output_dir", str(Path.home() / "invent_runs"))
        self.out_dir = Path(self.get_parameter("output_dir").value)
        self.rec = SortieRecorder()
        self.sortie = 0
        self.saved = True
        self.create_subscription(LabelRead, "labels", self.on_label, 100)
        self.create_subscription(MissionStatus, "mission/status", self.on_status, 10)

    def on_label(self, msg: LabelRead):
        self.rec.read(msg.address, msg.sscc or msg.text, msg.confidence)
        self.saved = False

    def on_status(self, msg: MissionStatus):
        self.sortie = msg.sortie
        if msg.state == "FOLLOW":
            self.rec.expect(msg.targets)
            self.saved = False
        elif msg.state in ("DONE", "ABORT") and not self.saved:
            self.save(msg.state)

    def save(self, reason: str):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"{stamp}_sortie{self.sortie}.csv"
        obs = self.rec.observations()
        write_observations_csv(obs, path)
        n_read = sum(1 for o in obs if o.code)
        self.get_logger().info(f"[{reason}] сохранено {path}: прочитано {n_read}, без чтения {len(obs) - n_read}")
        self.rec = SortieRecorder()
        self.saved = True


def main():
    rclpy.init()
    node = ScanRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node.saved:
            node.save("SHUTDOWN")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
