"""Статическая проверка sim_mission.launch.py без ROS 2 (заглушки launch / launch_ros).

ROS 2 молча игнорирует параметр, которого узел не объявил, — опечатка в имени параметра в launch
не даёт ошибки, узел просто работает со значением по умолчанию. Здесь сверяется:
* каждый LaunchConfiguration объявлен через DeclareLaunchArgument;
* package/executable наших узлов есть в console_scripts setup.py;
* каждый параметр из launch объявлен в коде узла, каждая remap-строка — имя топика в коде узла;
* файлы модели, на которые ссылается launch, генерирует sim/make_drone_model.py.
"""
import contextlib
import re
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "ros2_ws" / "src"
LAUNCH = SRC / "invent_bringup" / "launch" / "sim_mission.launch.py"
BUILTIN_PARAMS = {"use_sim_time"}


class Cfg:
    used: set = set()

    def __init__(self, name):
        self.name = name
        Cfg.used.add(name)


class Record:
    def __init__(self, *args, **kw):
        self.args, self.kw = args, kw


class PathJoin(Record):
    pass


@contextlib.contextmanager
def fake_launch():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        return m

    node_cls = type("Node", (Record,), {})
    declare = type("DeclareLaunchArgument", (Record,), {})
    fakes = {
        "launch": mod("launch", LaunchDescription=lambda actions: list(actions)),
        "launch.actions": mod("launch.actions", DeclareLaunchArgument=declare,
                              ExecuteProcess=type("ExecuteProcess", (Record,), {})),
        "launch.conditions": mod("launch.conditions", IfCondition=Record, UnlessCondition=Record),
        "launch.substitutions": mod("launch.substitutions", LaunchConfiguration=Cfg, PathJoinSubstitution=PathJoin),
        "launch_ros": mod("launch_ros"),
        "launch_ros.actions": mod("launch_ros.actions", Node=node_cls),
        "launch_ros.parameter_descriptions": mod("launch_ros.parameter_descriptions",
                                                 ParameterValue=lambda value, value_type=None: value),
    }
    saved = {k: sys.modules.get(k) for k in fakes}
    sys.modules.update(fakes)
    try:
        yield node_cls, declare
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def entry_points() -> dict[tuple[str, str], Path]:
    """(package, executable) -> файл модуля узла."""
    out = {}
    for setup in SRC.glob("*/setup.py"):
        pkg = setup.parent.name
        for exe, module in re.findall(r'"(\w+)\s*=\s*([\w.]+):main"', setup.read_text(encoding="utf-8")):
            out[(pkg, exe)] = setup.parent / (module.replace(".", "/") + ".py")
    return out


class LaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Cfg.used = set()
        with fake_launch() as (node_cls, declare):
            ns = {"__file__": str(LAUNCH), "__name__": "sim_mission_launch"}
            exec(compile(LAUNCH.read_text(encoding="utf-8"), str(LAUNCH), "exec"), ns)
            actions = ns["generate_launch_description"]()
            cls.nodes = [a for a in actions if isinstance(a, node_cls)]
            cls.declared = {a.args[0] if a.args else a.kw["name"] for a in actions if isinstance(a, declare)}
        cls.eps = entry_points()

    def ours(self):
        return [n for n in self.nodes if n.kw["package"].startswith("invent_")]

    def test_every_launch_configuration_is_declared(self):
        self.assertEqual(Cfg.used - self.declared, set())

    def test_our_executables_exist(self):
        for n in self.ours():
            key = (n.kw["package"], n.kw["executable"])
            self.assertIn(key, self.eps, f"нет console_scripts {key}")

    def test_parameters_are_declared_by_nodes(self):
        for n in self.ours():
            code = self.eps[(n.kw["package"], n.kw["executable"])].read_text(encoding="utf-8")
            for block in n.kw.get("parameters", []):
                for name in block:
                    if name in BUILTIN_PARAMS:
                        continue
                    with self.subTest(node=n.kw["executable"], param=name):
                        self.assertIn(f'"{name}"', code, f"{n.kw['executable']} не объявляет параметр {name}")

    def test_remapped_topics_exist_in_nodes(self):
        for n in self.ours():
            code = self.eps[(n.kw["package"], n.kw["executable"])].read_text(encoding="utf-8")
            for src, _dst in n.kw.get("remappings", []):
                with self.subTest(node=n.kw["executable"], topic=src):
                    self.assertIn(f'"{src}"', code)

    def test_model_files_are_generated(self):
        sys.path.insert(0, str(ROOT / "core"))
        sys.path.insert(0, str(ROOT / "sim"))
        import make_drone_model as mdm
        from invent_core.camera import Camera
        generated = set(mdm.build(Camera()))
        referenced = set()
        for n in self.nodes:
            for block in n.kw.get("parameters", []):
                for v in block.values():
                    if isinstance(v, PathJoin):
                        referenced.add(v.args[0][-1])
        self.assertTrue(referenced)
        self.assertLessEqual(referenced, generated)


if __name__ == "__main__":
    unittest.main()
