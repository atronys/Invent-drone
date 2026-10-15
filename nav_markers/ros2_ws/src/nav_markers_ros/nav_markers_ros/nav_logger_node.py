"""Журнал прогона: истина Gazebo, оценка EKF2, поза по меткам, события миссии -> CSV для eval_nav.py.

    track.csv   t; истина x y z yaw; EKF2 x y z yaw; дисперсия EKF2 по x, y; возраст позы по меткам; состояние миссии
    fixes.csv   t; поза по меткам x y z yaw; σ_xy; истина x y z yaw (на момент прихода)
    events.csv  t; событие (JSON миссии); истина x y z на момент события
    status.jsonl  статус marker_localizer раз в секунду: счётчики, отказы, состояние фильтра

Истинная поза — плагин PosePublisher модели (мост ros_gz -> /gz/true_pose, child_frame_id = имя модели).
"""
import csv
import json
import math
import os

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from px4_msgs.msg import VehicleOdometry

from invent_core.frames import matrix_from_quat, px4_pose_to_map
from invent_core.layout import Warehouse


def yaw_q(w, x, y, z) -> float:
    r = matrix_from_quat(w, x, y, z)
    return math.atan2(r[1][0], r[0][0])


class NavLogger(Node):
    def __init__(self):
        super().__init__("nav_logger")
        for name, default in [("layout", ""), ("out_dir", "."), ("model_name", "x500_nav_0"), ("rate_hz", 10.0)]:
            self.declare_parameter(name, default)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.dock = Warehouse.load(g("layout")).dock
        self.model = g("model_name")
        out = g("out_dir")
        os.makedirs(out, exist_ok=True)
        self.files = {k: open(os.path.join(out, f"{k}.csv"), "w", newline="", encoding="utf-8")
                      for k in ("track", "fixes", "events")}
        self.w = {k: csv.writer(f, delimiter=";") for k, f in self.files.items()}
        self.w["track"].writerow(["t", "tx", "ty", "tz", "tyaw", "ex", "ey", "ez", "eyaw", "evar_x", "evar_y",
                                  "fix_age", "state"])
        self.w["fixes"].writerow(["t", "fx", "fy", "fz", "fyaw", "fsigma_xy", "tx", "ty", "tz", "tyaw"])
        self.w["events"].writerow(["t", "event", "tx", "ty", "tz"])
        self.truth = None
        self.est = None
        self.fix_t = None
        self.state = ""
        self.warned = False
        px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(TFMessage, "/gz/true_pose", self.on_truth, 50)
        self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.on_odom, px4_qos)
        self.create_subscription(PoseWithCovarianceStamped, "/nav/marker_fix", self.on_fix, 50)
        self.create_subscription(String, "/nav/mission_event", self.on_event, 50)
        self.create_subscription(String, "/nav/mission_state", lambda m: setattr(self, "state", m.data), 10)
        self.status_file = open(os.path.join(out, "status.jsonl"), "w", encoding="utf-8")
        self.create_subscription(String, "/nav/localizer_status", self.on_status, 10)
        self.create_timer(1.0 / g("rate_hz"), self.tick)

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_truth(self, msg: TFMessage):
        tr = next((t for t in msg.transforms if t.child_frame_id == self.model), None)
        if tr is None:
            if not self.warned:
                self.warned = True
                self.get_logger().warn(f"в /gz/true_pose нет {self.model}: {[t.child_frame_id for t in msg.transforms]}")
            return
        p, q = tr.transform.translation, tr.transform.rotation
        self.truth = (p.x, p.y, p.z, yaw_q(q.w, q.x, q.y, q.z))

    def on_odom(self, msg: VehicleOdometry):
        if not all(math.isfinite(v) for v in msg.position):
            return
        p, q = px4_pose_to_map(tuple(msg.position), tuple(msg.q), self.dock)
        var = msg.position_variance
        self.est = (*p, yaw_q(*q), var[1], var[0])

    def on_fix(self, msg: PoseWithCovarianceStamped):
        self.fix_t = self.now_s()
        p, q, c = msg.pose.pose.position, msg.pose.pose.orientation, msg.pose.covariance
        self.w["fixes"].writerow([f"{self.fix_t:.3f}", f"{p.x:.4f}", f"{p.y:.4f}", f"{p.z:.4f}",
                                  f"{yaw_q(q.w, q.x, q.y, q.z):.5f}", f"{math.sqrt(max(0.0, c[0] + c[7])):.4f}",
                                  *(f"{v:.4f}" for v in (self.truth or (math.nan,) * 4))])

    def on_event(self, msg: String):
        self.w["events"].writerow([f"{self.now_s():.3f}", msg.data,
                                   *(f"{v:.4f}" for v in (self.truth or (math.nan,) * 4)[:3])])
        self.files["events"].flush()

    def on_status(self, msg: String):
        """Статус marker_localizer раз в секунду (+ время и истина) — для разбора отказов меток."""
        self.status_file.write(json.dumps({"t": round(self.now_s(), 2), "truth": self.truth, **json.loads(msg.data)},
                                          ensure_ascii=False) + "\n")
        self.status_file.flush()

    def tick(self):
        if self.truth is None or self.est is None:
            return
        age = math.nan if self.fix_t is None else self.now_s() - self.fix_t
        self.w["track"].writerow([f"{self.now_s():.3f}", *(f"{v:.4f}" for v in self.truth),
                                  *(f"{v:.4f}" for v in self.est[:4]), f"{self.est[4]:.5f}", f"{self.est[5]:.5f}",
                                  f"{age:.3f}", self.state])
        for f in self.files.values():      # прогон обрывают сигналом — данные не должны остаться в буфере
            f.flush()

    def close(self):
        for f in self.files.values():
            f.close()
        self.status_file.close()


def main():
    rclpy.init()
    node = NavLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
