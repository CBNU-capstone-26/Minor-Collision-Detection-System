from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .ui_helpers import clamp_box, resize_box


class VideoCanvas(QWidget):
    box_created = Signal(QRectF)
    box_selected = Signal(int)
    box_deselected = Signal()
    box_edit_started = Signal()
    box_edit_finished = Signal()
    box_deleted = Signal(int)
    box_changed = Signal(int, QRectF)
    frame_step_requested = Signal(int)
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.image = QImage()
        self.video_width = 1
        self.video_height = 1
        self.boxes: list[tuple[int, list[int]]] = []
        self.selected_id: int | None = None
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._dragging_existing = False
        self._drag_mode = ""
        self._drag_vehicle_id: int | None = None
        self._drag_box: list[int] | None = None
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self._panning = False
        self._pan_start: QPointF | None = None
        self._pan_origin = QPointF(0, 0)

    def set_frame(self, image: QImage, width: int, height: int, boxes: Iterable[tuple[int, list[int]]]) -> None:
        self.image = image
        self.video_width = max(1, width)
        self.video_height = max(1, height)
        self.boxes = [(int(vehicle_id), clamp_box(list(bbox), self.video_width, self.video_height)) for vehicle_id, bbox in boxes]
        self.update()

    def zoom_in(self) -> None:
        self.set_zoom(self.zoom_factor * 1.25, QPointF(self._to_widget(self.video_width / 2, self.video_height / 2)))

    def zoom_out(self) -> None:
        self.set_zoom(self.zoom_factor / 1.25, QPointF(self._to_widget(self.video_width / 2, self.video_height / 2)))

    def reset_view(self) -> None:
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.zoom_changed.emit(self.zoom_factor)
        self.update()

    def set_zoom(self, value: float, anchor: QPointF | None = None) -> None:
        value = max(1.0, min(5.0, float(value)))
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        if value == 1.0:
            self.reset_view()
            return
        old_video = self._to_video(anchor.toPoint())
        base = self._base_image_rect()
        scale = base.width() / self.video_width * value if not base.isNull() else 1.0
        self.zoom_factor = value
        centering_x = (base.width() * (value - 1.0)) / 2
        centering_y = (base.height() * (value - 1.0)) / 2
        self.pan_offset = QPointF(
            anchor.x() - (base.left() + old_video[0] * scale) + centering_x,
            anchor.y() - (base.top() + old_video[1] * scale) + centering_y,
        )
        self.zoom_changed.emit(self.zoom_factor)
        self.update()

    def pan_by(self, dx: float, dy: float) -> None:
        self.pan_offset += QPointF(dx, dy)
        self.update()

    def _base_image_rect(self) -> QRectF:
        if self.image.isNull():
            return QRectF()
        scale = min(self.width() / self.video_width, self.height() / self.video_height)
        width = self.video_width * scale
        height = self.video_height * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _image_rect(self) -> QRectF:
        if self.image.isNull():
            return QRectF()
        base = self._base_image_rect()
        if base.isNull():
            return base
        width = base.width() * self.zoom_factor
        height = base.height() * self.zoom_factor
        return QRectF(
            base.left() + self.pan_offset.x() - (width - base.width()) / 2,
            base.top() + self.pan_offset.y() - (height - base.height()) / 2,
            width,
            height,
        )

    def _to_video(self, point: QPoint) -> tuple[float, float]:
        rect = self._image_rect()
        if rect.isNull():
            return 0, 0
        return ((point.x() - rect.left()) * self.video_width / rect.width(), (point.y() - rect.top()) * self.video_height / rect.height())

    def _to_widget(self, x: float, y: float) -> QPoint:
        rect = self._image_rect()
        return QPoint(round(rect.left() + x * rect.width() / self.video_width), round(rect.top() + y * rect.height() / self.video_height))

    @staticmethod
    def _rect_from_box(box: list[int]) -> QRectF:
        return QRectF(box[0], box[1], box[2] - box[0], box[3] - box[1])

    def _box_at(self, point: QPoint) -> int | None:
        x, y = self._to_video(point)
        for vehicle_id, (x1, y1, x2, y2) in reversed(self.boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return vehicle_id
        return None

    def _box_for(self, vehicle_id: int) -> list[int] | None:
        for current_id, bbox in self.boxes:
            if current_id == vehicle_id:
                return bbox
        return None

    def _handle_points(self, bbox: list[int]) -> dict[str, QPoint]:
        x1, y1, x2, y2 = bbox
        return {
            "nw": self._to_widget(x1, y1), "n": self._to_widget((x1 + x2) / 2, y1), "ne": self._to_widget(x2, y1),
            "e": self._to_widget(x2, (y1 + y2) / 2), "se": self._to_widget(x2, y2), "s": self._to_widget((x1 + x2) / 2, y2),
            "sw": self._to_widget(x1, y2), "w": self._to_widget(x1, (y1 + y2) / 2),
        }

    def _handle_at(self, point: QPoint, bbox: list[int]) -> str | None:
        # The visible handle is small, especially when a large video is fitted
        # into the canvas. Keep a generous click target around each handle.
        radius = 14
        for handle, center in self._handle_points(bbox).items():
            if (point.x() - center.x()) ** 2 + (point.y() - center.y()) ** 2 <= radius**2:
                return handle
        return None

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#080B0D"))
        rect = self._image_rect()
        if not self.image.isNull():
            painter.drawImage(rect, self.image)
        for vehicle_id, (x1, y1, x2, y2) in self.boxes:
            painter.save()
            top_left = self._to_widget(x1, y1)
            bottom_right = self._to_widget(x2, y2)
            color = QColor("#F2B84B") if vehicle_id == self.selected_id else QColor("#68D5D0")
            pen = QPen(color, 3 if vehicle_id == self.selected_id else 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(top_left, bottom_right))
            painter.drawText(top_left + QPoint(4, -6), f"car {vehicle_id}")
            if vehicle_id == self.selected_id:
                painter.setBrush(QColor("#F4F0E8"))
                for center in self._handle_points([x1, y1, x2, y2]).values():
                    painter.drawEllipse(center, 5, 5)
            painter.restore()
        if self._drag_start and self._drag_current and not self._dragging_existing:
            painter.setPen(QPen(QColor("#F4F0E8"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(self._drag_start, self._drag_current).normalized())

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.MiddleButton and not self.image.isNull() and self.zoom_factor > 1.0:
            self._panning = True
            self._pan_start = event.position()
            self._pan_origin = QPointF(self.pan_offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton or self.image.isNull():
            return
        point = event.position().toPoint()
        # Check the selected box's handles first. A handle can overlap another
        # box or sit just outside its rectangle, so body hit-testing must not
        # win over an intentional resize click.
        handle = None
        hit = None
        if self.selected_id is not None:
            selected_box = self._box_for(self.selected_id)
            if selected_box is not None:
                handle = self._handle_at(point, selected_box)
                if handle is not None:
                    hit = self.selected_id
        if hit is None:
            hit = self._box_at(point)
        if hit is not None:
            self.selected_id = hit
            self._dragging_existing = True
            self._drag_vehicle_id = hit
            self._drag_box = list(self._box_for(hit) or [0, 0, 0, 0])
            self._drag_mode = handle or self._handle_at(point, self._drag_box) or "move"
            self.box_edit_started.emit()
            self.box_selected.emit(hit)
        else:
            self.selected_id = None
            self.box_deselected.emit()
            self._dragging_existing = False
            self._drag_mode = "create"
        self._drag_start = point
        self._drag_current = self._drag_start
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._panning and self._pan_start is not None:
            delta = event.position() - self._pan_start
            self.pan_offset = self._pan_origin + delta
            self.update()
            event.accept()
            return
        if self._drag_start:
            self._drag_current = event.position().toPoint()
            if self._dragging_existing and self._drag_vehicle_id is not None and self._drag_box is not None:
                start_x, start_y = self._to_video(self._drag_start)
                current_x, current_y = self._to_video(self._drag_current)
                dx, dy = current_x - start_x, current_y - start_y
                if self._drag_mode in {"nw", "n", "ne", "e", "se", "s", "sw", "w"}:
                    new_box = resize_box(self._drag_box, self._drag_mode, dx, dy, self.video_width, self.video_height)
                else:
                    width, height = self._drag_box[2] - self._drag_box[0], self._drag_box[3] - self._drag_box[1]
                    x1 = round(max(0, min(self.video_width - width, self._drag_box[0] + dx)))
                    y1 = round(max(0, min(self.video_height - height, self._drag_box[1] + dy)))
                    new_box = [x1, y1, x1 + width, y1 + height]
                new_box = clamp_box(new_box, self.video_width, self.video_height)
                self.boxes = [(vehicle_id, new_box if vehicle_id == self._drag_vehicle_id else bbox) for vehicle_id, bbox in self.boxes]
                self.box_changed.emit(self._drag_vehicle_id, self._rect_from_box(new_box))
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.unsetCursor()
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton or not self._drag_start:
            return
        end = event.position().toPoint()
        if not self._dragging_existing:
            x1, y1 = self._to_video(self._drag_start)
            x2, y2 = self._to_video(end)
            raw_box = [round(min(x1, x2)), round(min(y1, y2)), round(max(x1, x2)), round(max(y1, y2))]
            if raw_box[2] - raw_box[0] >= 4 and raw_box[3] - raw_box[1] >= 4:
                box = clamp_box(raw_box, self.video_width, self.video_height)
                self.box_created.emit(self._rect_from_box(box))
        else:
            self.box_edit_finished.emit()
        self._drag_start = None
        self._drag_current = None
        self._dragging_existing = False
        self._drag_mode = ""
        self._drag_vehicle_id = None
        self._drag_box = None
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.image.isNull():
            return
        delta = event.angleDelta().y()
        if delta:
            self.set_zoom(self.zoom_factor * (1.15 if delta > 0 else 1 / 1.15), event.position())
            event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() == Qt.Key.Key_Left:
            self.frame_step_requested.emit(-1)
        elif event.key() == Qt.Key.Key_Right:
            self.frame_step_requested.emit(1)
        elif event.key() == Qt.Key.Key_Escape:
            self._drag_start = None
            self._drag_current = None
            self._dragging_existing = False
            self._drag_mode = ""
            self._drag_vehicle_id = None
            self._drag_box = None
            if self.selected_id is not None:
                self.selected_id = None
                self.box_deselected.emit()
            self.update()
        elif event.key() == Qt.Key.Key_Delete and self.selected_id is not None:
            self.box_deleted.emit(self.selected_id)
            self.selected_id = None
            self.update()
        else:
            super().keyPressEvent(event)
