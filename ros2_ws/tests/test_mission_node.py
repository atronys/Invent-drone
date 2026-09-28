"""Автомат узла миссии без ROS 2: заглушки rclpy (ros_stubs) + игрушечный PX4.

Игрушечный PX4 исполняет DO_SET_MODE(Offboard), ARM, NAV_LAND и идеально отслеживает уставку позиции.
Проверяется логика узла (вылет, STOP, выход из Offboard, серия вылетов FLT-03), а не динамика —
динамика и настоящий PX4 — в scripts/smoke_sitl.sh.
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ros_stubs as rs  # noqa: E402
from invent_core.coverage import ScanParams, plan  # noqa: E402
from invent_core.layout import Warehouse, make_grid_warehouse  # noqa: E402

PKG = str(ROOT / "ros2_ws" / "src" / "invent_mission")
PERCEPTION = str(ROOT / "ros2_ws" / "src" / "invent_perception")


class ToyPX4:
    """Offboard: позиция = последняя уставка (NED); посадка — 1 м/с вниз, на земле — disarm."""

    def __init__(self, status_msg, pos_msg, cmd_msg):
        self.S, self.P, self.C = status_msg, pos_msg, cmd_msg
        self.armed, self.nav = False, self.S.NAVIGATION_STATE_POSCTL
        self.ned = [0.0, 0.0, 0.0]
        self.heading = 0.0
        self.sp = None
        self.last_t = 0.0
        self.max_step = 0.0            # наибольший скачок уставки за шаг, м
        rs.BUS["/fmu/in/trajectory_setpoint"].append(self.on_sp)
        rs.BUS["/fmu/in/vehicle_command"].append(self.on_cmd)

    def on_sp(self, m):
        if self.sp is not None:
            self.max_step = max(self.max_step, math.dist(self.sp.position, m.position))
        self.sp = m

    def on_cmd(self, m):
        C = self.C
        if m.command == C.VEHICLE_CMD_DO_SET_MODE and m.param2 == 6.0 and self.sp is not None:
            self.nav = self.S.NAVIGATION_STATE_OFFBOARD
        elif m.command == C.VEHICLE_CMD_COMPONENT_ARM_DISARM:
            self.armed = m.param1 == 1.0
        elif m.command == C.VEHICLE_CMD_NAV_LAND:
            self.nav = self.S.NAVIGATION_STATE_AUTO_LAND

    def step(self, t):
        dt, self.last_t = t - self.last_t, t
        if self.armed and self.nav == self.S.NAVIGATION_STATE_OFFBOARD and self.sp is not None:
            self.ned = list(self.sp.position)
            self.heading = self.sp.yaw
        elif self.armed and self.nav == self.S.NAVIGATION_STATE_AUTO_LAND:
            self.ned[2] = min(0.0, self.ned[2] + 1.0 * dt)
            if self.ned[2] >= 0.0:
                self.armed, self.nav = False, self.S.NAVIGATION_STATE_POSCTL
        rs.publish("/fmu/out/vehicle_local_position_v1",
                   self.P(x=self.ned[0], y=self.ned[1], z=self.ned[2], heading=self.heading,
                          xy_valid=True, z_valid=True))
        rs.publish("/fmu/out/vehicle_status_v1",
                   self.S(arming_state=self.S.ARMING_STATE_ARMED if self.armed else self.S.ARMING_STATE_DISARMED,
                          nav_state=self.nav))


class MissionNodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        wh = make_grid_warehouse(n_aisles=1, bays=4, levels=4, name="t")
        wh.save(d / "layout.json")
        # короткая АКБ -> несколько вылетов; быстрые скорости — короче прогон
        params = {"battery_s": 80.0, "v_scan": 0.8, "v_transit": 2.0, "v_z": 1.0, "yaw_rate_deg": 90.0,
                  "sortie_overhead_s": 5.0}
        (d / "params.json").write_text(json.dumps(params), encoding="utf-8")
        cls.layout, cls.params = str(d / "layout.json"), str(d / "params.json")
        cls.plan = plan(Warehouse.load(cls.layout), ScanParams.load(cls.params))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_node(self, seconds, hook=None, recorder_dir=None, **params):
        with rs.installed(PKG, PERCEPTION):
            rs.PARAMS["invent_mission"].update(layout=self.layout, scan_params=self.params, **params)
            from invent_mission.mission_node import MissionNode, State
            from px4_msgs.msg import VehicleCommand, VehicleLocalPosition, VehicleStatus
            node = MissionNode()
            if recorder_dir:
                rs.PARAMS["scan_recorder"]["output_dir"] = recorder_dir
                from invent_perception.scan_recorder_node import ScanRecorder
                node.recorder = ScanRecorder()
            px4 = ToyPX4(VehicleStatus, VehicleLocalPosition, VehicleCommand)

            def world(t):
                px4.step(t)
                if hook:
                    hook(t, node, px4)
            rs.spin_for(seconds, hook=world)
            return node, px4, State, [(m.state, m.sortie) for m in rs.SENT["mission/status"]]

    def test_multiple_sorties_planned(self):
        self.assertGreaterEqual(len(self.plan.sorties), 2, "для теста серии нужен план из >= 2 вылетов")

    def test_single_sortie_reaches_done(self):
        node, px4, State, statuses = self.run_node(self.plan.sorties[0].duration_s * 3 + 30)
        self.assertEqual(node.state, State.DONE, node.get_logger().text())
        self.assertFalse(px4.armed)
        self.assertEqual(node.i, len(node.wps) - 1)
        self.assertIn("посадка завершена", node.get_logger().text())
        self.assertEqual({s for _, s in statuses}, {0})
        self.assertLess(px4.max_step, 0.2, "уставка не должна прыгать: «морковка» ползёт с ограниченной скоростью")

    def test_stop_freezes_setpoint(self):
        frozen = {}

        def hook(t, node, px4):
            if node.state.value == "FOLLOW" and "t0" not in frozen and node.i >= 2:
                frozen["t0"] = t
                rs.publish("safety/state", sys.modules["std_msgs.msg"].String(data="STOP"))
                frozen["carrot"], frozen["i"] = node.carrot, node.i
            if "t0" in frozen and t - frozen["t0"] > 5.0 and "after" not in frozen:
                frozen["after"] = (node.carrot, node.i)

        self.run_node(60.0, hook)
        self.assertIn("after", frozen)
        self.assertEqual(frozen["after"][1], frozen["i"])
        self.assertLess(math.dist(frozen["after"][0], frozen["carrot"]), 1e-9)

    def test_leaving_offboard_aborts(self):
        def hook(t, node, px4):
            if node.state.value == "FOLLOW" and node.i >= 2:
                px4.nav = px4.S.NAVIGATION_STATE_POSCTL      # пилот переключил режим / failsafe PX4

        node, _, State, _ = self.run_node(60.0, hook)
        self.assertEqual(node.state, State.ABORT)

    def test_series_auto_continue(self):
        n = len(self.plan.sorties)
        total = sum(s.duration_s for s in self.plan.sorties) * 3 + n * 40
        node, _, State, statuses = self.run_node(total, series=True, continue_after_s=2.0)
        self.assertEqual(node.state, State.DONE, node.get_logger().text())
        self.assertEqual(node.sortie, n - 1)
        done_per_sortie = {s for st, s in statuses if st == "DONE"}
        self.assertEqual(done_per_sortie, set(range(n)))
        self.assertIn(f"серия завершена: {n} вылетов", node.get_logger().text())

    def test_series_waits_for_service(self):
        calls = []

        def hook(t, node, px4):
            if node.state.value == "FOLLOW" and not calls:
                calls.append(node.call("mission/continue"))   # в полёте — отказ
            if node.state.value == "DONE" and len(calls) == 1:
                calls.append(("waited", node.sortie, t))
            if len(calls) == 2 and t - calls[1][2] > 10.0:
                calls.append(node.call("mission/continue"))   # «АКБ заменили»

        node, _, State, _ = self.run_node(self.plan.sorties[0].duration_s * 3 + 60, hook, series=True)
        self.assertFalse(calls[0].success)
        self.assertEqual(calls[1][1], 0, "без команды узел не должен начинать следующий вылет")
        self.assertTrue(calls[2].success, calls[2].message)
        self.assertEqual(node.sortie, 1)
        self.assertIn("/mission/continue", node.get_logger().text())

    def test_series_saves_csv_per_sortie(self):
        """PER-07 + FLT-03: scan_recorder сохраняет CSV на каждый DONE; прочитанное — в файле своего вылета."""
        from invent_core.reconcile import load_observations_csv

        def hook(t, node, px4):
            if node.state.value == "FOLLOW" and node.wps[node.i].targets:
                addr = node.wps[node.i].targets[0]
                LabelRead = sys.modules["invent_msgs.msg"].LabelRead
                rs.publish("labels", LabelRead(text=f"SSCC-{addr}", sscc=f"SSCC-{addr}", address=addr, confidence=1.0))

        n = len(self.plan.sorties)
        with tempfile.TemporaryDirectory() as out:
            total = sum(s.duration_s for s in self.plan.sorties) * 3 + n * 40
            self.run_node(total, hook, recorder_dir=out, series=True, continue_after_s=2.0)
            files = sorted(Path(out).glob("*.csv"))
            self.assertEqual(sorted(f.stem.split("_sortie")[1] for f in files), [str(k) for k in range(n)])
            for f in files:
                k = int(f.stem.split("_sortie")[1])
                sortie_targets = {a for w in self.plan.sorties[k].waypoints for a in w.targets}
                read = {o.address for o in load_observations_csv(f) if o.code}
                self.assertTrue(read, f.name)
                self.assertLessEqual(read, sortie_targets, f"{f.name}: чтения чужого вылета")

    def test_continue_without_series_refused(self):
        node, _, State, _ = self.run_node(self.plan.sorties[0].duration_s * 3 + 30)
        self.assertEqual(node.state, State.DONE)
        r = node.call("mission/continue")
        self.assertFalse(r.success)
        self.assertIn("series", r.message)


if __name__ == "__main__":
    unittest.main()
