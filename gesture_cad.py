"""
GestureCAD - webcam controlled 2D/3D sketching in one Python file.

Controls
--------
Hand:
  Index finger only      draw with the Freehand tool
  Thumb/index pinch      click toolbar, place shapes, select and drag
  Index + middle fingers rotate the selected object (also extrudes 2D objects)

Mouse fallback:
  Left click/drag        toolbar, draw, place, select and drag
  Right drag             rotate selected object

Keyboard:
  D/S/R/T/C/B/P/O/Y      freehand/select/rectangle/triangle/circle/
                         3D box/3D prism/3D sphere/3D cylinder
  Delete                 delete selected
  Ctrl+Z                 undo
  C                      circle tool (use toolbar CLEAR to clear everything)
  Q or Escape            quit
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


WINDOW_NAME = "GestureCAD - Natural 2D/3D Editor"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

BG = (20, 23, 31)
PANEL = (31, 36, 48)
WHITE = (238, 241, 246)
MUTED = (153, 163, 181)
CYAN = (255, 204, 64)
GREEN = (112, 224, 151)
ORANGE = (66, 151, 255)
RED = (91, 91, 245)


@dataclass
class SceneObject:
    kind: str
    position: np.ndarray
    size: np.ndarray = field(default_factory=lambda: np.array([100.0, 100.0]))
    points: list[tuple[float, float]] = field(default_factory=list)
    depth: float = 70.0
    angle_x: float = 0.0
    angle_y: float = 0.0
    angle_z: float = 0.0
    is_3d: bool = False
    color: tuple[int, int, int] = CYAN


@dataclass
class Button:
    label: str
    tool: str
    key: str = ""


class HandTracker:
    """MediaPipe Tasks hand tracker, loaded lazily so self-tests need no camera."""

    def __init__(self, model_path: Path):
        import mediapipe as mp

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.mp = mp
        self.last_timestamp = 0

    def detect(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(time.monotonic() * 1000)
        timestamp = max(timestamp, self.last_timestamp + 1)
        self.last_timestamp = timestamp
        result = self.landmarker.detect_for_video(image, timestamp)
        return result.hand_landmarks[0] if result.hand_landmarks else None

    def close(self):
        self.landmarker.close()


class GestureCAD:
    TOOLBAR_H = 104

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.objects: list[SceneObject] = []
        self.selected: Optional[int] = None
        self.tool = "freehand"
        self.buttons = [
            Button("SELECT", "select", "S"),
            Button("DRAW", "freehand", "D"),
            Button("RECT", "rect", "R"),
            Button("TRI", "triangle", "T"),
            Button("CIRCLE", "circle", "C"),
            Button("3D RECT", "box", "B"),
            Button("3D TRI", "prism", "P"),
            Button("3D CIRCLE", "sphere", "O"),
            Button("CYLINDER", "cylinder", "Y"),
            Button("UNDO", "undo", "Z"),
            Button("DELETE", "delete", "DEL"),
            Button("CLEAR", "clear", ""),
        ]
        self.button_rects: list[tuple[int, int, int, int]] = []
        self.active_points: list[tuple[float, float]] = []
        self.drag_start: Optional[np.ndarray] = None
        self.drag_current: Optional[np.ndarray] = None
        self.drag_offset = np.zeros(2)
        self.interacting = False
        self.rotating = False
        self.last_cursor: Optional[np.ndarray] = None
        self.pinch_was_down = False
        self.smooth_cursor: Optional[np.ndarray] = None
        self.status = "Point at a tool and pinch"
        self.toast_until = 0.0
        self.fps = 0.0
        self._fps_time = time.perf_counter()
        self._fps_frames = 0
        self.mouse = np.zeros(2)
        self.mouse_left = False
        self.mouse_right = False
        self.mouse_prev_left = False
        self.mouse_prev_right = False

    # ---------------------------- geometry ----------------------------

    @staticmethod
    def _rotation_matrix(obj: SceneObject) -> np.ndarray:
        cx, sx = math.cos(obj.angle_x), math.sin(obj.angle_x)
        cy, sy = math.cos(obj.angle_y), math.sin(obj.angle_y)
        cz, sz = math.cos(obj.angle_z), math.sin(obj.angle_z)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], float)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], float)
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], float)
        return rz @ ry @ rx

    @staticmethod
    def _outline(obj: SceneObject, segments=32) -> np.ndarray:
        w, h = np.maximum(obj.size, 8.0)
        if obj.kind in ("rect", "box"):
            return np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                             [w / 2, h / 2], [-w / 2, h / 2]], float)
        if obj.kind in ("triangle", "prism"):
            return np.array([[0, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], float)
        if obj.kind in ("circle", "cylinder"):
            a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
            return np.column_stack((np.cos(a) * w / 2, np.sin(a) * h / 2))
        return np.asarray(obj.points, dtype=float) - obj.position

    def _mesh_for_object(self, obj: SceneObject):
        """Return vertices, edges, faces in object-local coordinates."""
        if obj.kind == "sphere":
            w, h = np.maximum(obj.size, 12.0)
            radius = min(w, h) / 2
            rings, sectors = 9, 16
            vertices = []
            for i in range(rings + 1):
                phi = -np.pi / 2 + np.pi * i / rings
                for j in range(sectors):
                    theta = 2 * np.pi * j / sectors
                    vertices.append([radius * np.cos(phi) * np.cos(theta),
                                     radius * np.sin(phi),
                                     radius * np.cos(phi) * np.sin(theta)])
            edges = []
            for i in range(rings + 1):
                for j in range(sectors):
                    k = i * sectors + j
                    edges.append((k, i * sectors + (j + 1) % sectors))
                    if i < rings:
                        edges.append((k, (i + 1) * sectors + j))
            return np.asarray(vertices), edges, []

        outline = self._outline(obj)
        if len(outline) < 2:
            return np.empty((0, 3)), [], []
        closed = obj.kind != "stroke"
        use_3d = obj.is_3d or obj.kind in ("box", "prism", "cylinder")
        if not use_3d:
            vertices = np.column_stack((outline, np.zeros(len(outline))))
            count = len(outline) if closed else len(outline) - 1
            edges = [(i, (i + 1) % len(outline)) for i in range(count)]
            return vertices, edges, [list(range(len(outline)))] if closed else []

        z = max(20.0, obj.depth) / 2
        vertices = np.vstack((
            np.column_stack((outline, np.full(len(outline), -z))),
            np.column_stack((outline, np.full(len(outline), z))),
        ))
        n = len(outline)
        edge_count = n if closed else n - 1
        edges = []
        for i in range(edge_count):
            nxt = (i + 1) % n
            edges.extend([(i, nxt), (i + n, nxt + n)])
        # Fewer connectors for a long freehand line keeps the rendering clean.
        connector_step = max(1, n // 30) if not closed else 1
        edges.extend((i, i + n) for i in range(0, n, connector_step))
        if not closed:
            edges.append((n - 1, 2 * n - 1))
        faces = []
        if closed:
            faces = [list(range(n)), list(range(n, 2 * n))]
            faces += [[i, (i + 1) % n, (i + 1) % n + n, i + n] for i in range(n)]
        return vertices, edges, faces

    def _project_mesh(self, obj: SceneObject):
        vertices, edges, faces = self._mesh_for_object(obj)
        if not len(vertices):
            return np.empty((0, 2)), vertices, edges, faces
        rotated = vertices @ self._rotation_matrix(obj).T
        camera_distance, focal = 900.0, 820.0
        denom = np.maximum(150.0, camera_distance - rotated[:, 2])
        scale = focal / denom
        projected = np.column_stack((
            obj.position[0] + rotated[:, 0] * scale,
            obj.position[1] + rotated[:, 1] * scale,
        ))
        return projected, rotated, edges, faces

    # ---------------------------- scene editing ----------------------------

    def _toast(self, message):
        self.status = message
        self.toast_until = time.time() + 1.8

    def set_tool(self, tool):
        if tool == "undo":
            if self.objects:
                self.objects.pop()
                self.selected = None
                self._toast("Undid last object")
            return
        if tool == "delete":
            if self.selected is not None and self.selected < len(self.objects):
                self.objects.pop(self.selected)
                self.selected = None
                self._toast("Deleted selected object")
            return
        if tool == "clear":
            self.objects.clear()
            self.selected = None
            self._toast("Canvas cleared")
            return
        self.tool = tool
        self.active_points.clear()
        self.drag_start = None
        self._toast(f"{tool.replace('_', ' ').title()} tool")

    def _button_at(self, point) -> Optional[str]:
        x, y = point
        for button, (x1, y1, x2, y2) in zip(self.buttons, self.button_rects):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return button.tool
        return None

    def _hit_test(self, point) -> Optional[int]:
        p = np.asarray(point, float)
        best, best_dist = None, 28.0
        for index in reversed(range(len(self.objects))):
            projected, _, edges, _ = self._project_mesh(self.objects[index])
            if not len(projected):
                continue
            minimum, maximum = projected.min(axis=0) - 10, projected.max(axis=0) + 10
            if np.all(p >= minimum) and np.all(p <= maximum):
                # Bounding-box interior makes solid shapes easy to grab.
                return index
            for a, b in edges:
                start, end = projected[a], projected[b]
                line = end - start
                t = np.clip(np.dot(p - start, line) / max(np.dot(line, line), 1), 0, 1)
                dist = np.linalg.norm(p - (start + t * line))
                if dist < best_dist:
                    best, best_dist = index, dist
        return best

    def _start_action(self, cursor):
        tool = self._button_at(cursor)
        if tool:
            self.set_tool(tool)
            return
        if cursor[1] <= self.TOOLBAR_H:
            return
        if self.tool == "select":
            self.selected = self._hit_test(cursor)
            if self.selected is not None:
                self.drag_offset = self.objects[self.selected].position - cursor
                self.interacting = True
                self._toast("Object selected - drag or use two fingers to rotate")
            else:
                self._toast("No object there")
        elif self.tool != "freehand":
            self.drag_start = cursor.copy()
            self.drag_current = cursor.copy()
            self.interacting = True

    def _continue_action(self, cursor):
        if self.tool == "select" and self.interacting and self.selected is not None:
            self.objects[self.selected].position = cursor + self.drag_offset
        elif self.interacting and self.drag_start is not None:
            self.drag_current = cursor.copy()

    def _finish_action(self, cursor):
        if self.tool != "select" and self.drag_start is not None:
            start = self.drag_start
            end = cursor
            delta = np.abs(end - start)
            if np.linalg.norm(delta) >= 18:
                center = (start + end) / 2
                kind_map = {
                    "rect": ("rect", False), "triangle": ("triangle", False),
                    "circle": ("circle", False), "box": ("box", True),
                    "prism": ("prism", True), "sphere": ("sphere", True),
                    "cylinder": ("cylinder", True),
                }
                kind, is_3d = kind_map[self.tool]
                obj = SceneObject(kind=kind, position=center, size=np.maximum(delta, 28),
                                  depth=max(45.0, min(delta) * 0.65), is_3d=is_3d)
                self.objects.append(obj)
                self.selected = len(self.objects) - 1
                self._toast(f"Created {self.tool}")
        self.drag_start = None
        self.drag_current = None
        self.interacting = False

    def _update_freehand(self, cursor, drawing):
        if self.tool != "freehand" or cursor[1] <= self.TOOLBAR_H:
            drawing = False
        if drawing:
            if not self.active_points:
                self.active_points.append(tuple(cursor))
            elif np.linalg.norm(cursor - self.active_points[-1]) > 4:
                self.active_points.append(tuple(cursor))
            self.status = "Drawing - lower another finger to finish"
        elif self.active_points:
            if len(self.active_points) >= 4:
                pts = np.asarray(self.active_points, float)
                center = (pts.min(axis=0) + pts.max(axis=0)) / 2
                # Uniformly thin very long strokes to keep 3D rotation fast.
                if len(pts) > 180:
                    pts = pts[np.linspace(0, len(pts) - 1, 180).astype(int)]
                obj = SceneObject(kind="stroke", position=center,
                                  points=[tuple(p) for p in pts],
                                  size=np.maximum(pts.max(axis=0) - pts.min(axis=0), 20),
                                  depth=55.0, is_3d=False, color=GREEN)
                self.objects.append(obj)
                self.selected = len(self.objects) - 1
                self._toast("Freehand object created - two fingers make it 3D")
            self.active_points.clear()

    def _rotate_selected(self, cursor):
        if self.selected is None or self.selected >= len(self.objects):
            return
        obj = self.objects[self.selected]
        if self.last_cursor is not None:
            delta = cursor - self.last_cursor
            obj.angle_y += float(delta[0]) * 0.012
            obj.angle_x -= float(delta[1]) * 0.012
            obj.is_3d = True
            obj.depth = max(obj.depth, min(max(obj.size) * 0.45, 120))
            self.status = "Rotating in 3D"
        self.last_cursor = cursor.copy()

    # ---------------------------- gestures/input ----------------------------

    @staticmethod
    def _finger_states(lm):
        # The y comparison remains stable under webcam mirroring.
        index = lm[8].y < lm[6].y
        middle = lm[12].y < lm[10].y
        ring = lm[16].y < lm[14].y
        pinky = lm[20].y < lm[18].y
        pinch = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y) < 0.055
        return index, middle, ring, pinky, pinch

    def handle_hand(self, landmarks):
        if landmarks is None:
            self._update_freehand(np.zeros(2), False)
            if self.pinch_was_down:
                self._finish_action(self.last_cursor if self.last_cursor is not None else np.zeros(2))
            self.pinch_was_down = False
            self.rotating = False
            self.smooth_cursor = None
            return None, False, None

        raw = np.array([landmarks[8].x * self.width, landmarks[8].y * self.height])
        self.smooth_cursor = raw if self.smooth_cursor is None else self.smooth_cursor * 0.65 + raw * 0.35
        cursor = self.smooth_cursor
        index, middle, ring, pinky, pinch = self._finger_states(landmarks)
        rotate_gesture = index and middle and not ring and not pinky and not pinch

        if rotate_gesture and self.selected is not None:
            if not self.rotating:
                self.last_cursor = cursor.copy()
            self.rotating = True
            self._rotate_selected(cursor)
        else:
            if self.rotating:
                self.last_cursor = None
            self.rotating = False
            if pinch and not self.pinch_was_down:
                self._start_action(cursor)
            elif pinch:
                self._continue_action(cursor)
            elif self.pinch_was_down:
                self._finish_action(cursor)
            self._update_freehand(cursor, index and not middle and not ring and not pinky and not pinch)

        self.pinch_was_down = pinch
        return cursor, pinch, landmarks

    def on_mouse(self, event, x, y, flags, _param):
        self.mouse[:] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_left = True
            self._start_action(self.mouse.copy())
            if self.tool == "freehand" and y > self.TOOLBAR_H:
                self.active_points = [(float(x), float(y))]
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_left:
                if self.tool == "freehand":
                    self._update_freehand(self.mouse.copy(), True)
                else:
                    self._continue_action(self.mouse.copy())
            if self.mouse_right:
                self._rotate_selected(self.mouse.copy())
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_left = False
            if self.tool == "freehand":
                self._update_freehand(self.mouse.copy(), False)
            self._finish_action(self.mouse.copy())
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_right = True
            self.last_cursor = self.mouse.copy()
        elif event == cv2.EVENT_RBUTTONUP:
            self.mouse_right = False
            self.last_cursor = None

    # ---------------------------- rendering ----------------------------

    def _draw_toolbar(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.TOOLBAR_H), PANEL, -1)
        cv2.addWeighted(overlay, 0.94, frame, 0.06, 0, frame)
        margin, gap = 10, 7
        usable = self.width - 2 * margin - gap * (len(self.buttons) - 1)
        button_w = usable // len(self.buttons)
        self.button_rects.clear()
        for i, button in enumerate(self.buttons):
            x1 = margin + i * (button_w + gap)
            x2 = x1 + button_w
            y1, y2 = 15, 79
            self.button_rects.append((x1, y1, x2, y2))
            active = button.tool == self.tool
            color = CYAN if active else (57, 64, 82)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE if active else (80, 89, 111), 1)
            scale = 0.37 if len(button.label) > 7 else 0.43
            tw = cv2.getTextSize(button.label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
            text_color = BG if active else WHITE
            cv2.putText(frame, button.label, (x1 + (button_w - tw) // 2, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)
            if button.key:
                kw = cv2.getTextSize(button.key, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0][0]
                cv2.putText(frame, button.key, (x1 + (button_w - kw) // 2, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            (28, 37, 50) if active else MUTED, 1, cv2.LINE_AA)

    def _draw_grid(self, frame):
        for x in range(0, self.width, 50):
            cv2.line(frame, (x, self.TOOLBAR_H), (x, self.height), (35, 39, 50), 1)
        for y in range(self.TOOLBAR_H, self.height, 50):
            cv2.line(frame, (0, y), (self.width, y), (35, 39, 50), 1)

    def _draw_object(self, frame, obj, selected=False):
        projected, rotated, edges, faces = self._project_mesh(obj)
        if not len(projected):
            return
        points = np.round(projected).astype(np.int32)

        # Painter's algorithm for translucent faces.
        if faces:
            ordered = sorted(faces, key=lambda f: float(np.mean(rotated[f, 2])))
            face_layer = frame.copy()
            for face in ordered:
                polygon = points[face]
                if len(polygon) >= 3:
                    cv2.fillConvexPoly(face_layer, polygon, obj.color)
            cv2.addWeighted(face_layer, 0.13, frame, 0.87, 0, frame)

        line_color = WHITE if selected else obj.color
        thickness = 3 if selected else 2
        for a, b in edges:
            cv2.line(frame, tuple(points[a]), tuple(points[b]), line_color,
                     thickness, cv2.LINE_AA)
        if selected:
            lo, hi = points.min(axis=0) - 8, points.max(axis=0) + 8
            cv2.rectangle(frame, tuple(lo), tuple(hi), CYAN, 1)
            for corner in ((lo[0], lo[1]), (hi[0], lo[1]), (lo[0], hi[1]), (hi[0], hi[1])):
                cv2.circle(frame, corner, 4, CYAN, -1)

    def _draw_preview(self, frame):
        if self.active_points:
            pts = np.asarray(self.active_points, np.int32)
            cv2.polylines(frame, [pts], False, GREEN, 4, cv2.LINE_AA)
        if self.drag_start is not None and self.drag_current is not None:
            start, end = self.drag_start.astype(int), self.drag_current.astype(int)
            x1, x2 = sorted((start[0], end[0]))
            y1, y2 = sorted((start[1], end[1]))
            if self.tool in ("rect", "box"):
                cv2.rectangle(frame, (x1, y1), (x2, y2), CYAN, 2)
            elif self.tool in ("triangle", "prism"):
                pts = np.array([[(x1 + x2) // 2, y1], [x2, y2], [x1, y2]])
                cv2.polylines(frame, [pts], True, CYAN, 2, cv2.LINE_AA)
            else:
                cv2.ellipse(frame, ((x1 + x2) // 2, (y1 + y2) // 2),
                            (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)),
                            0, 0, 360, CYAN, 2, cv2.LINE_AA)

    def _draw_hand(self, frame, landmarks, cursor, pinch):
        if landmarks:
            pts = np.array([(int(p.x * self.width), int(p.y * self.height))
                            for p in landmarks], np.int32)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
                (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
            ]
            for a, b in connections:
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (91, 103, 126), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, tuple(p), 2, WHITE, -1)
        if cursor is not None:
            c = tuple(cursor.astype(int))
            cv2.circle(frame, c, 13 if pinch else 10, ORANGE if pinch else CYAN, 2)
            cv2.circle(frame, c, 3, WHITE, -1)

    def render(self, camera_frame, cursor=None, pinch=False, landmarks=None):
        # Darkened camera image keeps the CAD content legible while preserving context.
        camera_frame = cv2.resize(camera_frame, (self.width, self.height))
        canvas = cv2.addWeighted(camera_frame, 0.20, np.full_like(camera_frame, BG), 0.80, 0)
        self._draw_grid(canvas)
        for i, obj in enumerate(self.objects):
            self._draw_object(canvas, obj, i == self.selected)
        self._draw_preview(canvas)
        self._draw_toolbar(canvas)
        self._draw_hand(canvas, landmarks, cursor, pinch)

        self._fps_frames += 1
        elapsed = time.perf_counter() - self._fps_time
        if elapsed >= 0.5:
            self.fps = self._fps_frames / elapsed
            self._fps_frames, self._fps_time = 0, time.perf_counter()

        cv2.rectangle(canvas, (12, self.height - 54), (self.width - 12, self.height - 12), PANEL, -1)
        help_text = self.status if time.time() < self.toast_until else (
            "Pinch: click/drag   |   Index only: draw   |   Index + middle: rotate/extrude"
        )
        cv2.putText(canvas, help_text, (28, self.height - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        info = f"{len(self.objects)} objects   {self.fps:.0f} FPS"
        tw = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        cv2.putText(canvas, info, (self.width - tw - 28, self.height - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, MUTED, 1, cv2.LINE_AA)
        return canvas


def ensure_model(model_path: Path):
    if model_path.exists() and model_path.stat().st_size > 1_000_000:
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the official MediaPipe hand-landmark model (about 8 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {MODEL_URL}\n"
            f"Download it manually and save it as: {model_path}\nOriginal error: {exc}"
        ) from exc


def run_self_test():
    app = GestureCAD(1280, 720)
    cases = [
        SceneObject("rect", np.array([300.0, 300.0]), is_3d=True),
        SceneObject("triangle", np.array([300.0, 300.0]), is_3d=True),
        SceneObject("sphere", np.array([300.0, 300.0]), is_3d=True),
        SceneObject("cylinder", np.array([300.0, 300.0]), is_3d=True),
        SceneObject("stroke", np.array([300.0, 300.0]),
                    points=[(250, 250), (300, 220), (350, 260), (320, 340)],
                    is_3d=True),
    ]
    for obj in cases:
        vertices, edges, _ = app._mesh_for_object(obj)
        projected, _, _, _ = app._project_mesh(obj)
        assert len(vertices) and len(edges) and projected.shape == (len(vertices), 2)
        assert np.isfinite(projected).all()
    print("GestureCAD self-test passed: 2D extrusion and all 3D meshes are valid.")


def main():
    parser = argparse.ArgumentParser(description="GestureCAD webcam 2D/3D editor")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--self-test", action="store_true", help="Test geometry without opening a webcam")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    model_path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
    ensure_model(model_path)
    try:
        tracker = HandTracker(model_path)
    except Exception as exc:
        print(f"Could not initialize MediaPipe: {exc}", file=sys.stderr)
        print("Install dependencies with: pip install -r requirements.txt", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        tracker.close()
        print(f"Could not open camera {args.camera}. Try --camera 1", file=sys.stderr)
        return 1

    app = GestureCAD(args.width, args.height)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)
    cv2.setMouseCallback(WINDOW_NAME, app.on_mouse)
    key_tools = {
        ord("s"): "select", ord("d"): "freehand", ord("r"): "rect",
        ord("t"): "triangle", ord("c"): "circle", ord("b"): "box",
        ord("p"): "prism", ord("o"): "sphere", ord("y"): "cylinder",
    }

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame could not be read.", file=sys.stderr)
                break
            frame = cv2.flip(frame, 1)
            landmarks = tracker.detect(frame)
            cursor, pinch, visible_landmarks = app.handle_hand(landmarks)
            output = app.render(frame, cursor, pinch, visible_landmarks)
            cv2.imshow(WINDOW_NAME, output)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in key_tools:
                app.set_tool(key_tools[key])
            elif key in (8, 127):  # Backspace/Delete varies by platform.
                app.set_tool("delete")
            elif key == 26:  # Ctrl+Z
                app.set_tool("undo")
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
