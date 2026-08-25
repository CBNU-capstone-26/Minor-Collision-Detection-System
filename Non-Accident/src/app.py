from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from PySide6.QtCore import QSignalBlocker, QTimer, Qt
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QListWidget, QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter, QVBoxLayout, QWidget, QComboBox, QSlider, QMenu

from .annotation_model import Box, Event, VideoAnnotation, load_annotations, save_annotations, validate_annotation
from .exporter import export_event, export_nonaccident
from .video_view import VideoCanvas
from .ui_helpers import sort_video_paths


class AnnotationWindow(QMainWindow):
    STATUS_LABELS = {"not_started": "미작성", "in_progress": "작업 중", "needs_review": "검토 필요", "normal": "정상", "confirmed": "완료", "excluded": "제외"}

    def __init__(self, source_root: Path, annotation_path: Path, output_root: Path, mode: str = "accident") -> None:
        super().__init__()
        self.setWindowTitle("FRAME / Manual Collision Annotation")
        self.resize(1440, 900)
        self.source_root, self.annotation_path, self.output_root, self.mode = source_root, annotation_path, output_root, mode
        # 시작 시에는 사용자가 작업 폴더를 선택할 때까지 영상과 기록을 불러오지 않는다.
        self.annotations = {}
        self.videos: list[Path] = []
        self.capture: cv2.VideoCapture | None = None
        self.current: VideoAnnotation | None = None
        self.frame_index = 0
        self._undo_boxes: list[tuple[list[Box], int | None]] = []
        self._redo_boxes: list[tuple[list[Box], int | None]] = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self._build_ui()
        self.status_label.setText("폴더 열기를 눌러 작업할 영상 폴더를 선택하세요.")

    def _build_ui(self) -> None:
        self.setStyleSheet("QMainWindow, QWidget { background:#101820; color:#F4F0E8; } QListWidget, QSpinBox, QComboBox { background:#18232B; border:1px solid #34434D; padding:6px; } QPushButton { background:#22343D; border:1px solid #48606C; padding:8px 12px; } QPushButton:hover { background:#2D4A55; } QLabel#frame { color:#68D5D0; font-family:monospace; font-size:18px; }")
        self.video_list = QListWidget()
        self.video_list.currentRowChanged.connect(self.open_row)
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("번호순", "number")
        self.sort_combo.addItem("파일명순", "name")
        self.sort_combo.addItem("검수 상태순", "status")
        self.sort_combo.currentIndexChanged.connect(lambda _index: self.refresh_video_list())
        self.canvas = VideoCanvas()
        self.canvas.box_created.connect(self.create_box)
        self.canvas.box_selected.connect(self.select_box)
        self.canvas.box_deselected.connect(self.clear_box_selection)
        self.canvas.box_edit_started.connect(self.record_box_action)
        self.canvas.box_deleted.connect(self.delete_box)
        self.canvas.box_changed.connect(self.update_box)
        self.canvas.frame_step_requested.connect(self.step_frame)
        self.zoom_out_button = QPushButton("−")
        self.zoom_reset_button = QPushButton("원래대로")
        self.zoom_in_button = QPushButton("+")
        self.zoom_label = QLabel("100%")
        self.zoom_out_button.clicked.connect(self.canvas.zoom_out)
        self.zoom_reset_button.clicked.connect(self.canvas.reset_view)
        self.zoom_in_button.clicked.connect(self.canvas.zoom_in)
        self.canvas.zoom_changed.connect(lambda value: self.zoom_label.setText(f"{value * 100:.0f}%"))
        self.frame_label = QLabel("frame 0000")
        self.frame_label.setObjectName("frame")
        self.status_label = QLabel("작업할 영상을 선택하세요.")
        self.vehicle_list = QListWidget()
        self.vehicle_list.currentRowChanged.connect(self.select_vehicle_row)
        self.vehicle_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.vehicle_list.customContextMenuRequested.connect(self.open_vehicle_menu)
        self.vehicle_combo = QComboBox()
        self.event_list = QListWidget()
        self.event_list.currentRowChanged.connect(self.select_event_row)
        self.event_status_combo = QComboBox()
        for value, label in (("needs_review", "검수 필요"), ("confirmed", "완료"), ("excluded", "제외")):
            self.event_status_combo.addItem(label, value)
        self.event_note = QLineEdit()
        self.event_note.setPlaceholderText("이벤트 메모")
        self.status_combo = QComboBox()
        for value, label in self.STATUS_LABELS.items():
            self.status_combo.addItem(label, value)
        self.status_combo.currentIndexChanged.connect(self.change_status)
        self.start_spin = QSpinBox(); self.end_spin = QSpinBox()
        self.confirm_button = QPushButton("검수 완료")
        self.confirm_button.clicked.connect(self.confirm_nonaccident if self.mode == "non-accident" else self.confirm_event)
        self.start_event_button = QPushButton("시작 시점 기록")
        self.start_event_button.clicked.connect(self.record_start_frame)
        self.end_event_button = QPushButton("종료 시점 기록")
        self.end_event_button.clicked.connect(self.record_end_frame)
        self.save_event_button = QPushButton("선택 이벤트 저장")
        self.save_event_button.clicked.connect(self.save_event)
        self.delete_event_button = QPushButton("선택 이벤트 삭제")
        self.delete_event_button.clicked.connect(self.delete_event)
        self.save_button = QPushButton("저장")
        self.save_button.clicked.connect(self.save_current)
        self.export_button = QPushButton("TXT + 검수 MP4 생성")
        self.export_button.clicked.connect(self.export_current)
        self.open_folder_button = QPushButton("폴더 열기")
        self.open_folder_button.clicked.connect(self.choose_folder)
        self.new_annotation_button = QPushButton("어노테이션 새로 만들기")
        self.new_annotation_button.clicked.connect(self.choose_new_annotation_file)
        self.load_annotation_button = QPushButton("어노테이션 불러오기")
        self.load_annotation_button.clicked.connect(self.choose_annotation_file)
        self.delete_vehicle_button = QPushButton("선택 박스 삭제")
        self.delete_vehicle_button.clicked.connect(self.delete_selected_vehicle)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.setSingleStep(1)
        self.timeline.setPageStep(10)
        self.timeline.valueChanged.connect(self.seek_frame)
        self.undo_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Undo), self)
        self.undo_shortcut.activated.connect(self.undo_boxes)
        self.redo_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Redo), self)
        self.redo_shortcut.activated.connect(self.redo_boxes)
        self.redo_alt_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.redo_alt_shortcut.activated.connect(self.redo_boxes)

        left = QVBoxLayout(); left.addWidget(QLabel("영상 목록")); left.addWidget(self.sort_combo); left.addWidget(self.video_list); left.addWidget(self.open_folder_button); left.addWidget(self.new_annotation_button); left.addWidget(self.load_annotation_button)
        controls = QHBoxLayout()
        self.previous_button = QPushButton("이전")
        self.play_button = QPushButton("재생")
        self.next_button = QPushButton("다음")
        self.previous_button.clicked.connect(self.previous_frame)
        self.play_button.clicked.connect(self.toggle_play)
        self.next_button.clicked.connect(self.next_frame)
        for button in (self.previous_button, self.next_button):
            button.setAutoRepeat(True)
            button.setAutoRepeatDelay(350)
            button.setAutoRepeatInterval(60)
        for button in (self.previous_button, self.play_button, self.next_button):
            controls.addWidget(button)
        controls.addWidget(self.zoom_out_button); controls.addWidget(self.zoom_reset_button); controls.addWidget(self.zoom_in_button); controls.addWidget(self.zoom_label); controls.addWidget(self.frame_label); controls.addStretch()
        center = QVBoxLayout(); center.addWidget(self.status_label); center.addWidget(self.canvas, 1); center.addWidget(self.timeline); center.addLayout(controls)
        form = QFormLayout(); form.addRow("영상 현황", self.status_combo)
        if self.mode == "non-accident":
            self.confirm_button.setText("기준 차량 2대 확정")
        else:
            form.addRow("사고 차량 ID", self.vehicle_combo); form.addRow("시작 프레임", self.start_spin); form.addRow("종료 프레임", self.end_spin); form.addRow("이벤트 상태", self.event_status_combo); form.addRow("메모", self.event_note)
        right = QVBoxLayout(); right.setContentsMargins(8, 8, 8, 8); right.addWidget(QLabel("기준 차량 박스")); right.addWidget(self.vehicle_list); right.addWidget(self.delete_vehicle_button)
        if self.mode == "non-accident":
            right.addWidget(QLabel("비사고 모드: 이동 차량 주변의 기준 차량 2대를 선택하세요."))
        else:
            right.addWidget(QLabel("이벤트 목록")); right.addWidget(self.event_list); right.addWidget(self.start_event_button); right.addWidget(self.end_event_button); right.addWidget(self.save_event_button); right.addWidget(self.delete_event_button)
        right.addLayout(form); right.addWidget(self.confirm_button); right.addWidget(self.save_button); right.addWidget(self.export_button); right.addStretch()
        root = QSplitter(); left_widget = QWidget(); left_widget.setLayout(left); center_widget = QWidget(); center_widget.setLayout(center); right_widget = QWidget(); right_widget.setMinimumWidth(350); right_widget.setLayout(right)
        right_scroll = QScrollArea(); right_scroll.setWidgetResizable(True); right_scroll.setFrameShape(QScrollArea.Shape.NoFrame); right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); right_scroll.setWidget(right_widget)
        root.addWidget(left_widget); root.addWidget(center_widget); root.addWidget(right_scroll); root.setSizes([220, 900, 370]); self.setCentralWidget(root)

    def load_folder(self, folder: Path) -> None:
        if self.capture:
            self.capture.release()
            self.capture = None
        self.current = None
        self.frame_index = 0
        self.videos = []
        self.canvas.set_frame(QImage(), 1, 1, [])
        self.refresh_vehicle_list()
        self.refresh_event_list()
        self.videos = list(folder.rglob("*.mp4")) if folder.is_dir() else []
        saved_annotations = load_annotations(self.annotation_path) if self.annotation_path.is_file() else {}
        self._replace_annotations(saved_annotations)
        self.refresh_video_list()
        if self.videos:
            self.video_list.setCurrentRow(0)
        else:
            self.status_label.setText("선택한 폴더에 MP4 영상이 없습니다.")

    def _replace_annotations(self, saved_annotations) -> None:
        video_paths = {path.resolve() for path in self.videos}
        video_ids = {path.stem for path in self.videos}
        self.annotations = {
            video_id: item for video_id, item in saved_annotations.items()
            if video_id in video_ids or Path(item.source_video).expanduser().resolve() in video_paths
        }
        self.refresh_video_list()

    def _annotation_for_path(self, path: Path) -> VideoAnnotation | None:
        resolved = path.resolve()
        for annotation in self.annotations.values():
            if Path(annotation.source_video).expanduser().resolve() == resolved:
                return annotation
        return self.annotations.get(path.stem)

    def refresh_video_list(self) -> None:
        current_path = self.videos[self.video_list.currentRow()] if 0 <= self.video_list.currentRow() < len(self.videos) else None
        self.videos = sort_video_paths(self.videos, str(self.sort_combo.currentData()), self.annotations)
        self.video_list.blockSignals(True)
        self.video_list.clear()
        for path in self.videos:
            annotation = self._annotation_for_path(path)
            status = self.STATUS_LABELS.get(annotation.status, "미작성") if annotation is not None else "미작성"
            self.video_list.addItem(f"{path.stem}  ·  {status}")
        self.video_list.blockSignals(False)
        if self.videos:
            self.video_list.setCurrentRow(self.videos.index(current_path) if current_path in self.videos else 0)

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "영상 폴더 선택", str(self.source_root))
        if folder: self.load_folder(Path(folder))

    def choose_new_annotation_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "새 어노테이션 JSON 생성", str(self.annotation_path), "JSON 파일 (*.json)")
        if not path:
            return
        target = Path(path).with_suffix(".json")
        if target.exists():
            answer = QMessageBox.question(self, "파일 덮어쓰기", f"'{target.name}' 파일을 빈 JSON으로 덮어쓸까요?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.create_annotation_file(target)
        self.status_label.setText(f"새 어노테이션 파일 생성됨 · {target.name}")

    def choose_annotation_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "어노테이션 JSON 불러오기", str(self.annotation_path.parent), "JSON 파일 (*.json)")
        if not path:
            return
        try:
            self.load_annotation_file(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "불러오기 실패", str(exc))
            return
        self.status_label.setText(f"어노테이션 불러옴 · {Path(path).name}")

    def create_annotation_file(self, path: Path) -> None:
        self.annotation_path = Path(path)
        self.annotations = {}
        save_annotations(self.annotation_path, self.annotations)
        self._replace_annotations({})

    def load_annotation_file(self, path: Path) -> None:
        loaded = load_annotations(path)
        self.annotation_path = Path(path)
        self._replace_annotations(loaded)

    def open_row(self, row: int) -> None:
        if row < 0 or row >= len(self.videos): return
        path = self.videos[row]
        if self.capture: self.capture.release()
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened(): QMessageBox.critical(self, "열기 실패", str(path)); return
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0; count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        split = "learning" if "학습용" in path.parts or "learning" in path.parts else "testing" if "테스트용" in path.parts or "testing" in path.parts else "normal"
        saved = self._annotation_for_path(path)
        video_id = saved.video_id if saved is not None else path.stem
        self.current = saved or VideoAnnotation(video_id, str(path), split, width, height, count, fps)
        self.current.source_video = str(path); self.frame_index = 0; self._undo_boxes.clear(); self._redo_boxes.clear(); self.start_spin.setRange(0, count - 1); self.end_spin.setRange(0, count - 1); self.timeline.setRange(0, max(0, count - 1)); self.set_status_combo(self.current.status); self.refresh_vehicle_list(); self.refresh_event_list(); self.show_frame()

    def show_frame(self) -> None:
        if not self.capture or not self.current: return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index); ok, frame = self.capture.read()
        if not ok: return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.canvas.set_frame(image, self.current.width, self.current.height, [(box.vehicle_id, box.bbox) for box in self.current.boxes]); self.frame_label.setText(f"frame {self.frame_index:04d} / {self.current.frame_count - 1:04d}")
        with QSignalBlocker(self.timeline):
            self.timeline.setValue(self.frame_index)

    def previous_frame(self) -> None: self.frame_index = max(0, self.frame_index - 1); self.show_frame()
    def next_frame(self) -> None: self.frame_index = min((self.current.frame_count - 1) if self.current else 0, self.frame_index + 1); self.show_frame()
    def step_frame(self, amount: int) -> None:
        if amount < 0:
            self.previous_frame()
        elif amount > 0:
            self.next_frame()
    def seek_frame(self, frame: int) -> None:
        if self.current and 0 <= frame < self.current.frame_count and frame != self.frame_index:
            self.frame_index = frame; self.show_frame()
    def toggle_play(self) -> None:
        if self.timer.isActive(): self.timer.stop()
        else: self.timer.start(max(1, round(1000 / (self.current.fps if self.current else 30))))

    def refresh_vehicle_list(self) -> None:
        self.vehicle_list.clear(); self.vehicle_combo.clear()
        if not self.current: return
        for box in self.current.boxes: self.vehicle_list.addItem(f"car {box.vehicle_id}  {box.bbox}  @ {box.reference_frame}"); self.vehicle_combo.addItem(str(box.vehicle_id), box.vehicle_id)

    def refresh_event_list(self, selected_index: int = 0) -> None:
        self.event_list.blockSignals(True)
        self.event_list.clear()
        if self.current:
            for event in self.current.events:
                label = {"needs_review": "검수 필요", "confirmed": "완료", "excluded": "제외"}.get(event.status, event.status)
                self.event_list.addItem(f"{event.event_id} · {label}")
            if self.current.events:
                self.event_list.setCurrentRow(min(max(selected_index, 0), len(self.current.events) - 1))
        self.event_list.blockSignals(False)
        if self.current and self.current.events:
            self.load_event(self.event_list.currentRow())

    def select_event_row(self, row: int) -> None:
        self.load_event(row)

    def load_event(self, row: int) -> None:
        if not self.current or not (0 <= row < len(self.current.events)):
            return
        event = self.current.events[row]
        with QSignalBlocker(self.event_status_combo):
            index = self.event_status_combo.findData(event.status)
            self.event_status_combo.setCurrentIndex(max(0, index))
        self.vehicle_combo.setCurrentIndex(max(0, self.vehicle_combo.findData(event.vehicle_id)))
        self.start_spin.setValue(event.start_frame)
        self.end_spin.setValue(event.end_frame)
        self.event_note.setText(event.note)

    def selected_event(self) -> Event | None:
        if not self.current:
            return None
        row = self.event_list.currentRow()
        return self.current.events[row] if 0 <= row < len(self.current.events) else None

    def set_status_combo(self, status: str) -> None:
        with QSignalBlocker(self.status_combo):
            index = self.status_combo.findData(status)
            self.status_combo.setCurrentIndex(max(0, index))

    def change_status(self, index: int) -> None:
        if self.current and index >= 0:
            self.current.status = str(self.status_combo.itemData(index))
            self.status_label.setText(f"현황 변경됨 · {self.STATUS_LABELS[self.current.status]} · 저장 버튼을 눌러 확정하세요.")

    def create_box(self, rect) -> None:
        if not self.current: return
        self.record_box_action()
        vehicle_id = max([box.vehicle_id for box in self.current.boxes], default=-1) + 1
        self.current.boxes.append(Box(vehicle_id, [round(rect.left()), round(rect.top()), round(rect.right()), round(rect.bottom())], self.frame_index)); self.canvas.selected_id = vehicle_id; self.current.status = "in_progress"; self.set_status_combo(self.current.status); self.refresh_vehicle_list(); self.show_frame()
    def select_box(self, vehicle_id: int) -> None:
        self.canvas.selected_id = vehicle_id; self.show_frame()
    def clear_box_selection(self) -> None:
        self.canvas.selected_id = None
        with QSignalBlocker(self.vehicle_list):
            self.vehicle_list.setCurrentRow(-1)
        self.canvas.update()

    def _box_state(self) -> tuple[list[Box], int | None]:
        if not self.current:
            return [], None
        boxes = [Box(box.vehicle_id, list(box.bbox), box.reference_frame) for box in self.current.boxes]
        return boxes, self.canvas.selected_id

    def record_box_action(self) -> None:
        if self.current:
            self._undo_boxes.append(self._box_state())
            self._redo_boxes.clear()

    def _restore_box_state(self, state: tuple[list[Box], int | None]) -> None:
        if not self.current:
            return
        boxes, selected_id = state
        self.current.boxes = [Box(box.vehicle_id, list(box.bbox), box.reference_frame) for box in boxes]
        self.canvas.selected_id = selected_id
        self.refresh_vehicle_list()
        self.show_frame()

    def undo_boxes(self) -> None:
        if not self.current or not self._undo_boxes:
            return
        self._redo_boxes.append(self._box_state())
        self._restore_box_state(self._undo_boxes.pop())
        self.status_label.setText("박스 작업을 되돌렸습니다 · 저장 버튼을 눌러 확정하세요.")

    def redo_boxes(self) -> None:
        if not self.current or not self._redo_boxes:
            return
        self._undo_boxes.append(self._box_state())
        self._restore_box_state(self._redo_boxes.pop())
        self.status_label.setText("박스 작업을 다시 적용했습니다 · 저장 버튼을 눌러 확정하세요.")
    def select_vehicle_row(self, row: int) -> None:
        if self.current and 0 <= row < len(self.current.boxes): self.canvas.selected_id = self.current.boxes[row].vehicle_id; self.show_frame()
    def delete_box(self, vehicle_id: int) -> None:
        if self.current and any(box.vehicle_id == vehicle_id for box in self.current.boxes):
            self.record_box_action(); self.current.boxes = [box for box in self.current.boxes if box.vehicle_id != vehicle_id]; self.canvas.selected_id = None; self.refresh_vehicle_list(); self.show_frame()
    def delete_selected_vehicle(self) -> None:
        row = self.vehicle_list.currentRow()
        if self.current and 0 <= row < len(self.current.boxes):
            self.delete_box(self.current.boxes[row].vehicle_id)
    def open_vehicle_menu(self, position) -> None:
        row = self.vehicle_list.indexAt(position).row()
        if row < 0:
            return
        self.vehicle_list.setCurrentRow(row)
        menu = QMenu(self)
        action = menu.addAction("선택 박스 삭제")
        action.triggered.connect(self.delete_selected_vehicle)
        menu.exec(self.vehicle_list.mapToGlobal(position))
    def update_box(self, vehicle_id: int, rect) -> None:
        if self.current:
            for box in self.current.boxes:
                if box.vehicle_id == vehicle_id:
                    box.bbox = [round(rect.left()), round(rect.top()), round(rect.right()), round(rect.bottom())]
                    break
            self.refresh_vehicle_list()
    def add_event(self) -> None:
        if not self.current or not self.current.boxes: return
        used_ids = {event.event_id for event in self.current.events}
        event_number = len(self.current.events) + 1
        event_id = f"{self.current.video_id}-a{event_number}"
        while event_id in used_ids:
            event_number += 1
            event_id = f"{self.current.video_id}-a{event_number}"
        vehicle_id = int(self.vehicle_combo.currentData() or self.current.boxes[0].vehicle_id)
        self.current.events.append(Event(event_id, vehicle_id, self.frame_index, self.frame_index, "needs_review")); self.current.status = "in_progress"; self.set_status_combo(self.current.status); self.start_spin.setValue(self.frame_index); self.end_spin.setValue(self.frame_index)
        self.refresh_event_list(len(self.current.events) - 1)

    def record_start_frame(self) -> None:
        if not self.current or not self.current.boxes:
            QMessageBox.warning(self, "차량 박스 없음", "먼저 차량 박스를 하나 이상 입력하세요.")
            return
        event = self.selected_event()
        if event is None:
            self.add_event()
            event = self.selected_event()
        if event is None:
            return
        event.start_frame = self.frame_index
        if event.end_frame < event.start_frame:
            event.end_frame = event.start_frame
        self.start_spin.setValue(event.start_frame)
        self.end_spin.setValue(event.end_frame)
        self.current.status = "in_progress"
        self.set_status_combo(self.current.status)
        self.refresh_event_list(self.event_list.currentRow())

    def record_end_frame(self) -> None:
        event = self.selected_event()
        if event is None:
            QMessageBox.warning(self, "이벤트 없음", "먼저 이벤트를 선택하거나 시작 시점 기록을 누르세요.")
            return
        if self.frame_index < event.start_frame:
            QMessageBox.warning(self, "프레임 순서 오류", "종료 시점은 시작 시점 이후 프레임이어야 합니다.")
            return
        event.end_frame = self.frame_index
        self.end_spin.setValue(event.end_frame)
        self.current.status = "in_progress"
        self.set_status_combo(self.current.status)
        self.refresh_event_list(self.event_list.currentRow())

    def save_event(self) -> None:
        event = self.selected_event()
        if not self.current or event is None or self.vehicle_combo.currentData() is None:
            QMessageBox.warning(self, "이벤트 없음", "저장할 이벤트를 먼저 선택하세요.")
            return
        event.vehicle_id = int(self.vehicle_combo.currentData())
        event.start_frame = self.start_spin.value()
        event.end_frame = self.end_spin.value()
        event.status = str(self.event_status_combo.currentData())
        event.note = self.event_note.text()
        self.current.status = "confirmed" if self.current.events and all(item.status == "confirmed" for item in self.current.events) else "in_progress"
        self.set_status_combo(self.current.status)
        self.refresh_event_list(self.event_list.currentRow())
        self.save_current()

    def delete_event(self) -> None:
        if not self.current:
            return
        row = self.event_list.currentRow()
        if not (0 <= row < len(self.current.events)):
            QMessageBox.warning(self, "이벤트 없음", "삭제할 이벤트를 먼저 선택하세요.")
            return
        event = self.current.events[row]
        answer = QMessageBox.question(self, "이벤트 삭제 확인", f"'{event.event_id}' 이벤트를 삭제할까요?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.current.events.pop(row)
        if self.current.status == "confirmed":
            self.current.status = "in_progress" if self.current.events else "not_started"
            self.set_status_combo(self.current.status)
        self.refresh_event_list(max(0, row - 1))
        self.save_current()

    def confirm_event(self) -> None:
        if not self.current or not self.current.events: self.add_event()
        event = self.selected_event()
        if self.current and event is not None:
            event.vehicle_id = int(self.vehicle_combo.currentData()); event.start_frame = self.start_spin.value(); event.end_frame = self.end_spin.value(); event.status = "confirmed"; event.note = self.event_note.text(); self.current.status = "confirmed"; self.set_status_combo(self.current.status); self.refresh_event_list(self.event_list.currentRow()); self.save_current()
    def confirm_nonaccident(self) -> None:
        if not self.current or len(self.current.boxes) != 2:
            QMessageBox.warning(self, "기준 차량 확인 필요", "비사고 영상은 기준 차량 박스 2개가 필요합니다.")
            return
        self.current.boxes = [Box(index, list(box.bbox), box.reference_frame) for index, box in enumerate(self.current.boxes)]
        self.current.status = "normal"
        self.set_status_combo(self.current.status)
        self.save_current()
        self.status_label.setText("기준 차량 2대가 확정되었습니다.")
    def save_current(self) -> None:
        if self.current:
            self.annotations[self.current.video_id] = self.current; save_annotations(self.annotation_path, self.annotations); self.refresh_video_list(); self.status_label.setText(f"저장됨 · {self.current.video_id}")
    def export_current(self) -> None:
        if not self.current: return
        if self.mode == "non-accident":
            try:
                self.save_current(); export_nonaccident(self.current, self.output_root)
            except Exception as exc:
                QMessageBox.critical(self, "생성 실패", str(exc)); return
            self.status_label.setText("비사고 TXT와 검수용 MP4를 생성했습니다.")
            return
        self.save_current(); errors = validate_annotation(self.current)
        if errors: QMessageBox.warning(self, "검증 실패", "\n".join(errors)); return
        try:
            for index, event in enumerate(self.current.events): export_event(self.current, event, self.output_root, index)
        except Exception as exc: QMessageBox.critical(self, "생성 실패", str(exc)); return
        self.status_label.setText("TXT와 검수용 MP4를 생성했습니다.")

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() == Qt.Key.Key_Escape:
            self.clear_box_selection()
        elif event.key() == Qt.Key.Key_Left:
            self.previous_frame()
        elif event.key() == Qt.Key.Key_Right:
            self.next_frame()
        elif event.key() == Qt.Key.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key.Key_S:
            self.save_current()
        else:
            super().keyPressEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-compatible manual vehicle annotation GUI")
    parser.add_argument("--source-root", type=Path, default=Path("Accident"))
    parser.add_argument("--annotations", type=Path, default=Path("Pre-Processing/work/accident_annotations.json"))
    parser.add_argument("--output-root", type=Path, default=Path("Pre-Processing/output"))
    parser.add_argument("--mode", choices=("accident", "non-accident"), default="accident")
    args = parser.parse_args()
    app = QApplication(sys.argv); window = AnnotationWindow(args.source_root, args.annotations, args.output_root, args.mode); window.show(); sys.exit(app.exec())


if __name__ == "__main__":
    main()
