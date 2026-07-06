from pathlib import Path
from typing import Optional
from types import SimpleNamespace
from datetime import datetime
import re


from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.database import SessionLocal
from src.export_yolo_obb_db import export_dataset
from src.db.models import (
    Annotation,
    AnnotationBox,
    Frame,
    Prediction,
    QueueItem,
    Review,
)


app = FastAPI(title="DB Product Review UI")


class ReviewRequest(BaseModel):
    queue_item_id: int
    review_status: str
    note: Optional[str] = None


class ReviewResetRequest(BaseModel):
    queue_item_id: int


class AnnotationBoxRequest(BaseModel):
    class_id: int = 0
    label_name: str = "product"
    points: list[list[float]]
    box_source: str = "manual"
    confidence: Optional[float] = None


class SaveAnnotationsRequest(BaseModel):
    frame_id: int
    queue_item_id: Optional[int] = None
    boxes: list[AnnotationBoxRequest] = []


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path

    return Path.cwd() / path


def model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def normalize_points(points: list[list[float]], width: int | None, height: int | None) -> list[list[float]]:
    normalized = []

    for point in points:
        x = float(point[0])
        y = float(point[1])

        nx = x / width if width else 0.0
        ny = y / height if height else 0.0

        normalized.append(
            [
                round(max(0.0, min(1.0, nx)), 6),
                round(max(0.0, min(1.0, ny)), 6),
            ]
        )

    return normalized


def review_to_dict(review: Review | None) -> dict | None:
    if not review:
        return None

    return {
        "id": review.id,
        "review_status": review.review_status,
        "note": review.note,
        "reviewed_by": review.reviewed_by,
        "created_at": review.created_at.isoformat() if review.created_at else None,
        "updated_at": review.updated_at.isoformat() if review.updated_at else None,
    }


def get_reset_status(queue_item: QueueItem) -> str:
    if queue_item.queue_type == "skipped_queue" or queue_item.decision == "skipped_high_confidence":
        return "skipped"

    return "pending"


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>DB Product Review System</title>
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #151515;
            color: #e9e9e9;
        }

        header {
            padding: 14px 20px;
            background: #0b0b0b;
            border-bottom: 1px solid #333333;
            font-size: 20px;
            font-weight: bold;
        }

        .toolbar {
            display: flex;
            gap: 10px;
            padding: 10px 20px;
            background: #242424;
            align-items: center;
            flex-wrap: wrap;
            border-bottom: 1px solid #333333;
        }

        .toolbar label,
        .inline-label {
            color: #bbbbbb;
            font-size: 14px;
        }

        select,
        button,
        input,
        textarea {
            border-radius: 6px;
            border: 1px solid #555555;
            background: #111111;
            color: #eeeeee;
            font-family: inherit;
            font-size: 14px;
        }

        select,
        button,
        input {
            min-height: 36px;
            padding: 7px 10px;
        }

        select {
            min-width: 120px;
        }

        input.small-input {
            width: 74px;
        }

        button {
            cursor: pointer;
            white-space: nowrap;
        }

        button:hover {
            background: #333333;
        }

        button:disabled {
            cursor: not-allowed;
            opacity: 0.45;
        }

        .primary-button {
            border-color: #22c55e;
        }

        .danger-button {
            border-color: #ef4444;
        }

        .nav-button {
            min-width: 76px;
        }

        .play-button {
            min-width: 112px;
            border-color: #f59e0b;
        }

        .jump-input {
            width: 90px;
        }

        .layout {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 390px;
            gap: 16px;
            padding: 16px 20px;
        }

        .canvas-panel {
            background: #050505;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 10px;
            overflow: auto;
            max-height: calc(100vh - 220px);
            cursor: grab;
            user-select: none;
        }

        .canvas-panel.panning,
        .canvas-panel.panning canvas {
            cursor: grabbing !important;
        }

        canvas {
            max-width: none;
            height: auto;
            display: block;
            margin: auto;
            background: black;
            cursor: crosshair;
            transform-origin: 0 0;
            will-change: transform;
        }

        .side-panel {
            background: #050505;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 14px;
            max-height: calc(100vh - 220px);
            overflow-y: auto;
        }

        .section-title {
            margin: 0 0 10px;
            color: #86efac;
            font-size: 16px;
        }

        .info-row {
            margin: 8px 0;
            font-size: 14px;
            line-height: 1.35;
        }

        .label {
            color: #aaaaaa;
        }

        .value {
            color: #ffffff;
            font-weight: bold;
            overflow-wrap: anywhere;
        }

        .review-buttons,
        .annotation-buttons {
            display: grid;
            grid-template-columns: 1fr;
            gap: 8px;
            margin-top: 12px;
        }

        .review-buttons button,
        .annotation-buttons button {
            text-align: left;
            min-height: 38px;
        }

        .review-buttons button.active-review {
            border-color: #22c55e;
            background: #12351f;
        }

        .annotation-buttons button.active-tool {
            border-color: #38bdf8;
            background: #102b3d;
        }

        .object-item,
        .manual-box-item,
        .review-card {
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 8px;
            margin-bottom: 8px;
            background: #151515;
            font-size: 13px;
            line-height: 1.4;
        }

        .manual-box-item {
            cursor: pointer;
        }

        .manual-box-item.selected {
            border-color: #f97316;
            background: #2c1c12;
        }

        .manual-box-item.multi-selected {
            border-color: #38bdf8;
            background: #102b3d;
        }

        .confidence-input {
            width: 78px;
        }

        .empty-state {
            color: #aaaaaa;
        }

        textarea {
            width: 100%;
            height: 70px;
            padding: 8px;
            resize: vertical;
        }

        .status-text {
            min-height: 36px;
            padding: 9px 20px;
            color: #facc15;
            background: #151515;
            border-bottom: 1px solid #333333;
        }

        .panel-block {
            margin-bottom: 18px;
        }

        .inline-controls {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            margin: 8px 0;
        }

        .editor-status {
            min-height: 20px;
            color: #facc15;
            font-size: 13px;
            margin-top: 8px;
        }

        @media (max-width: 960px) {
            .layout {
                grid-template-columns: 1fr;
            }

            .canvas-panel,
            .side-panel {
                max-height: none;
            }
        }
    </style>
</head>
<body>
    <header>Product Take-Out Review System - DB Viewer <span style="color:#facc15; font-size:14px;">v2.4 export</span></header>

    <div class="toolbar">
        <label for="queueReason">Queue:</label>
        <select id="queueReason"></select>

        <label for="statusFilter">Status:</label>
        <select id="statusFilter">
            <option value="all">all</option>
            <option value="pending">pending</option>
            <option value="completed">completed</option>
            <option value="skipped">skipped</option>
        </select>

        <label for="orderFilter">Order:</label>
        <select id="orderFilter"></select>

        <label for="cameraFilter">Camera:</label>
        <select id="cameraFilter"></select>

        <button type="button" title="Load frames using the selected queue filters" onclick="loadFrames()">
            Load Filtered Queue
        </button>
        <button
            type="button"
            class="primary-button"
            title="Load all frames for the selected order and camera"
            onclick="loadSelectedVideo()"
        >
            Load Selected Video
        </button>
        <button
            type="button"
            class="primary-button"
            title="Auto-save current frame, export the selected Order/Camera as a CVAT-style YOLO OBB dataset ZIP"
            onclick="downloadExportZip()"
        >
            Download Export ZIP
        </button>
    </div>

    <div class="toolbar">
        <button id="firstButton" class="nav-button" type="button" title="Go to first frame (Home)" onclick="goFirst()">
            First
        </button>
        <button id="back10Button" class="nav-button" type="button" title="Go back 10 frames (C)" onclick="stepFrame(-10)">
            Back 10
        </button>
        <button id="back1Button" class="nav-button" type="button" title="Go back 1 frame (D or ArrowLeft)" onclick="stepFrame(-1)">
            Back 1
        </button>
        <button id="playButton" class="play-button" type="button" title="Play / Pause (Space)" onclick="togglePlay()">
            Play
        </button>
        <button id="next1Button" class="nav-button" type="button" title="Go next 1 frame (F or ArrowRight)" onclick="stepFrame(1)">
            Next 1
        </button>
        <button id="next10Button" class="nav-button" type="button" title="Go next 10 frames (V)" onclick="stepFrame(10)">
            Next 10
        </button>
        <button id="lastButton" class="nav-button" type="button" title="Go to last frame (End)" onclick="goLast()">
            Last
        </button>

        <label for="jumpInput">Jump:</label>
        <input id="jumpInput" class="jump-input" type="number" min="1" value="1">
        <button id="jumpButton" type="button" title="Jump to frame index" onclick="jumpToIndex()">Go</button>

        <label for="playbackSpeed">Speed:</label>
        <select id="playbackSpeed" title="Playback speed in frames per second" onchange="handleSpeedChange()">
            <option value="4">4 FPS</option>
            <option value="8" selected>8 FPS</option>
            <option value="12">12 FPS</option>
            <option value="24">24 FPS</option>
        </select>

        <label for="confidenceFilter">Conf ≥</label>
        <input id="confidenceFilter" class="confidence-input" type="number" min="0" max="1" step="0.05" value="0.25" title="Only show model detections at or above this confidence. Click Apply Conf after changing it.">
        <button type="button" title="Apply confidence filter to the loaded frames and model boxes" onclick="loadFrames()">Apply Conf</button>
    </div>

    <div id="statusText" class="status-text">Loading...</div>

    <div class="layout">
        <div class="canvas-panel">
            <canvas id="imageCanvas"></canvas>
        </div>

        <div class="side-panel">
            <div class="panel-block">
                <h3 class="section-title">Frame Info</h3>
                <div id="frameInfo"></div>
            </div>

            <div class="panel-block">
                <h3 class="section-title">Objects</h3>
                <div id="objectsList"></div>
            </div>

            <div class="panel-block">
                <h3 class="section-title">Annotation Boxes</h3>
                <div class="inline-controls">
                    <span class="inline-label">Class:</span>
                    <input id="classIdInput" class="small-input" type="number" min="0" value="0" readonly title="All objects are saved as class 0: product">
                    <input id="labelNameInput" type="text" value="product" readonly title="All objects are saved as product for now">
                    <label class="inline-label" title="Show or hide green original model prediction boxes as a reference layer">
                        <input id="showModelBoxesInput" type="checkbox" onchange="redrawCanvas()">
                        Show Original Model Boxes
                    </label>
                </div>
                <div class="annotation-buttons">
                    <button id="copyBoxesButton" type="button" title="Reset editable boxes back to the original model detections" onclick="copyModelBoxes()">
                        Reset From Model
                    </button>
                    <button id="addBoxButton" type="button" title="Smart Draw Mode: drag empty area to draw. Click existing boxes to edit without turning this off. Press Escape to leave draw mode." onclick="startAddBox()">
                        Draw Box
                    </button>
                    <button id="lockBoxButton" type="button" title="Lock the selected box so nearby boxes are not accidentally selected. Click another box to unlock and switch selection. You can also Ctrl+Click a box to lock it." onclick="toggleSelectedBoxLock()">
                        Lock Selected
                    </button>
                    <button id="multiSelectButton" type="button" title="Optional selection mode (M). Drag a rectangle around boxes. After selecting, drag any selected box to move the whole group, or press Delete." onclick="toggleMultiSelectMode()">
                        Select Many
                    </button>
                    <button id="deleteBoxButton" type="button" title="Delete selected annotation box or all group-selected boxes (Delete key)" onclick="deleteSelectedBox()">
                        Delete Selected Box
                    </button>
                    <button id="clearBoxesButton" type="button" title="Remove all unsaved annotation boxes from this frame" onclick="clearManualBoxes()">
                        Clear Boxes
                    </button>
                    <button id="saveBoxesButton" class="primary-button" type="button" title="Save annotation boxes for this frame to PostgreSQL" onclick="saveAnnotations()">
                        Save Boxes
                    </button>
                </div>
                <div id="editorStatus" class="editor-status"></div>
                <div id="manualBoxesList"></div>
            </div>

            <div class="panel-block">
                <h3 class="section-title">Review</h3>
                <div id="currentReview"></div>
                <textarea id="reviewNote" placeholder="Optional note..."></textarea>

                <div class="review-buttons">
                    <button type="button" data-review-status="correct_detection" title="Save or update review as correct detection" onclick="saveReview('correct_detection')">
                        Correct Detection
                    </button>
                    <button type="button" data-review-status="wrong_box" title="Save or update review as wrong box" onclick="saveReview('wrong_box')">
                        Wrong Box
                    </button>
                    <button type="button" data-review-status="box_needs_split" title="Save or update review as box needs split" onclick="saveReview('box_needs_split')">
                        Box Needs Split
                    </button>
                    <button type="button" data-review-status="real_missed_detection" title="Save or update review as real missed detection" onclick="saveReview('real_missed_detection')">
                        Real Missed Detection
                    </button>
                    <button type="button" data-review-status="correct_no_detection" title="Save or update review as correct no detection" onclick="saveReview('correct_no_detection')">
                        Correct No Detection
                    </button>
                    <button type="button" data-review-status="ignore_frame" title="Save or update review as ignore frame" onclick="saveReview('ignore_frame')">
                        Ignore Frame
                    </button>
                    <button id="resetReviewButton" class="danger-button" type="button" title="Undo review and return this queue item to its original queue status" onclick="resetReview()">
                        Undo / Reset Review
                    </button>
                </div>
            </div>
        </div>
    </div>

<script>
let frames = [];
let currentIndex = 0;
let isPlaying = false;
let playbackTimer = null;
let renderToken = 0;
let currentImage = null;
let manualBoxes = [];
let selectedBoxIndex = -1;
let annotationMode = 'select';
let draftBox = null;
let dragState = null;
let selectedBoxLocked = false;
let selectedBoxIndexes = new Set();
let multiSelectMode = false;
let selectionRect = null;
let annotationsDirty = false;
let boxClipboard = null;
let undoStack = [];
let redoStack = [];
let playbackAdvancing = false;
let canvasZoomScale = 1;
let canvasPanX = 0;
let canvasPanY = 0;
let canvasPanState = null;

const MAX_UNDO_STATES = 50;
const MIN_CANVAS_ZOOM = 0.25;
const MAX_CANVAS_ZOOM = 8;
const CANVAS_ZOOM_STEP = 1.15;

const HANDLE_RADIUS = 7;
const ROTATE_HANDLE_OFFSET = 36;
const MIN_BOX_SIZE = 8;

const navButtonIds = [
    'firstButton',
    'back10Button',
    'back1Button',
    'playButton',
    'next1Button',
    'next10Button',
    'lastButton',
    'jumpButton'
];

const annotationButtonIds = [
    'copyBoxesButton',
    'addBoxButton',
    'lockBoxButton',
    'deleteBoxButton',
    'clearBoxesButton',
    'saveBoxesButton'
];

async function loadFilters() {
    const res = await fetch('/api/filters');
    const data = await res.json();

    fillSelect('queueReason', ['all', ...data.queue_reasons]);
    fillSelect('orderFilter', ['all', ...data.order_codes]);
    fillSelect('cameraFilter', ['all', ...data.cameras]);
}

function fillSelect(id, values) {
    const select = document.getElementById(id);
    select.innerHTML = '';

    values.forEach(value => {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
    });
}

async function loadFrames() {
    const canLeave = await autoSaveCurrentAnnotationsIfDirty();

    if (!canLeave) {
        return;
    }

    pausePlayback();

    const params = new URLSearchParams({
        queue_reason: document.getElementById('queueReason').value,
        status: document.getElementById('statusFilter').value,
        order_code: document.getElementById('orderFilter').value,
        camera: document.getElementById('cameraFilter').value,
        min_confidence: getConfidenceFilterValue()
    });

    document.getElementById('statusText').textContent = 'Loading frames...';

    const res = await fetch(`/api/frames?${params.toString()}`);

    if (!res.ok) {
        document.getElementById('statusText').textContent = 'Failed to load frames.';
        return;
    }

    frames = await res.json();
    currentIndex = 0;
    updateControlState();

    if (frames.length === 0) {
        document.getElementById('statusText').textContent = 'No frames found.';
        clearCanvas();
        clearPanels();
        return;
    }

    renderCurrentFrame();
}

async function loadSelectedVideo() {
    const orderCode = document.getElementById('orderFilter').value;
    const camera = document.getElementById('cameraFilter').value;

    if (orderCode === 'all' || camera === 'all') {
        alert('Choose one Order and one Camera before loading a selected video.');
        return;
    }

    document.getElementById('queueReason').value = 'all';
    document.getElementById('statusFilter').value = 'all';

    await loadFrames();
}

function clearCanvas() {
    const canvas = document.getElementById('imageCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 800;
    canvas.height = 450;
    updateCanvasDisplaySize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function clearPanels() {
    document.getElementById('frameInfo').innerHTML = '';
    document.getElementById('objectsList').innerHTML = '';
    document.getElementById('currentReview').innerHTML = '';
    document.getElementById('manualBoxesList').innerHTML = '';
    document.getElementById('editorStatus').textContent = '';
    document.getElementById('jumpInput').value = '1';
    annotationsDirty = false;
}

function renderCurrentFrame() {
    if (frames.length === 0) {
        return;
    }

    if (annotationMode !== 'draw') {
        annotationMode = 'select';
    }
    draftBox = null;
    dragState = null;
    selectedBoxIndex = -1;
    selectedBoxIndexes.clear();
    selectedBoxLocked = false;
    selectionRect = null;

    const item = frames[currentIndex];
    manualBoxes = getEditableBoxesForItem(item);
    annotationsDirty = false;

    document.getElementById('statusText').textContent =
        `Showing ${currentIndex + 1} / ${frames.length}`;

    const jumpInput = document.getElementById('jumpInput');
    jumpInput.max = frames.length;
    jumpInput.value = currentIndex + 1;

    renderInfo(item);
    renderObjects(item);
    renderReview(item);
    renderManualBoxesList();
    updateControlState();

    const token = ++renderToken;
    const img = new Image();

    img.onload = function () {
        if (token !== renderToken) {
            return;
        }

        currentImage = img;
        const canvas = document.getElementById('imageCanvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        updateCanvasDisplaySize();
        redrawCanvas();
    };

    img.onerror = function () {
        if (token !== renderToken) {
            return;
        }

        currentImage = null;
        clearCanvas();
        document.getElementById('statusText').textContent =
            `Image failed to load for frame_id ${item.frame_id}`;
    };

    img.src = `/api/image/${item.frame_id}?t=${Date.now()}`;
}

function getCanvasPanel() {
    return document.querySelector('.canvas-panel');
}

function getCanvasFitScale() {
    const canvas = document.getElementById('imageCanvas');
    const panel = getCanvasPanel();

    if (!canvas || !panel || canvas.width === 0) {
        return 1;
    }

    const panelStyle = window.getComputedStyle(panel);
    const horizontalPadding = parseFloat(panelStyle.paddingLeft) + parseFloat(panelStyle.paddingRight);
    const availableWidth = Math.max(panel.clientWidth - horizontalPadding, 1);

    return Math.min(1, availableWidth / canvas.width);
}

function updateCanvasDisplaySize() {
    const canvas = document.getElementById('imageCanvas');

    if (!canvas || canvas.width === 0 || canvas.height === 0) {
        return;
    }

    const displayScale = getCanvasFitScale() * canvasZoomScale;
    canvas.style.width = `${canvas.width * displayScale}px`;
    canvas.style.height = `${canvas.height * displayScale}px`;
    updateCanvasTransform();
}

function updateCanvasTransform() {
    const canvas = document.getElementById('imageCanvas');

    if (!canvas) {
        return;
    }

    canvas.style.transform = `translate(${canvasPanX}px, ${canvasPanY}px)`;
}

function panCanvasBy(dx, dy) {
    const panel = getCanvasPanel();
    const beforeLeft = panel.scrollLeft;
    const beforeTop = panel.scrollTop;

    panel.scrollLeft -= dx;
    panel.scrollTop -= dy;

    const consumedX = beforeLeft - panel.scrollLeft;
    const consumedY = beforeTop - panel.scrollTop;

    canvasPanX += dx - consumedX;
    canvasPanY += dy - consumedY;
    updateCanvasTransform();
}

function handleCanvasWheel(event) {
    if (!currentImage) {
        return;
    }

    const canvas = document.getElementById('imageCanvas');
    const panel = getCanvasPanel();
    const canvasRect = canvas.getBoundingClientRect();

    if (canvasRect.width === 0 || canvasRect.height === 0) {
        return;
    }

    event.preventDefault();

    const imageXRatio = clamp((event.clientX - canvasRect.left) / canvasRect.width, 0, 1);
    const imageYRatio = clamp((event.clientY - canvasRect.top) / canvasRect.height, 0, 1);
    const zoomFactor = event.deltaY < 0 ? CANVAS_ZOOM_STEP : 1 / CANVAS_ZOOM_STEP;
    const nextZoom = clamp(canvasZoomScale * zoomFactor, MIN_CANVAS_ZOOM, MAX_CANVAS_ZOOM);

    if (nextZoom === canvasZoomScale) {
        return;
    }

    canvasZoomScale = nextZoom;
    updateCanvasDisplaySize();

    const newCanvasRect = canvas.getBoundingClientRect();
    const newImageScreenX = newCanvasRect.left + imageXRatio * newCanvasRect.width;
    const newImageScreenY = newCanvasRect.top + imageYRatio * newCanvasRect.height;

    panCanvasBy(event.clientX - newImageScreenX, event.clientY - newImageScreenY);
}

function startCanvasPan(event) {
    canvasPanState = {
        lastClientX: event.clientX,
        lastClientY: event.clientY
    };

    getCanvasPanel().classList.add('panning');
    event.preventDefault();
}

function handleCanvasPanelMouseDown(event) {
    if (![0, 1, 2].includes(event.button) || event.target !== event.currentTarget) {
        return;
    }

    startCanvasPan(event);
}

function handleCanvasPanMove(event) {
    if (!canvasPanState) {
        return;
    }

    const dx = event.clientX - canvasPanState.lastClientX;
    const dy = event.clientY - canvasPanState.lastClientY;

    panCanvasBy(dx, dy);
    canvasPanState.lastClientX = event.clientX;
    canvasPanState.lastClientY = event.clientY;
    event.preventDefault();
}

function stopCanvasPan() {
    if (!canvasPanState) {
        return;
    }

    canvasPanState = null;
    getCanvasPanel().classList.remove('panning');
}

function redrawCanvas() {
    const canvas = document.getElementById('imageCanvas');
    const ctx = canvas.getContext('2d');

    if (!currentImage) {
        return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImage, 0, 0);

    const item = frames[currentIndex];
    if (isShowingModelBoxes()) {
        drawModelBoxes(ctx, item.prediction.boxes || []);
    }
    drawManualBoxes(ctx);
    drawDraftBox(ctx);
}

function isShowingModelBoxes() {
    return document.getElementById('showModelBoxesInput').checked;
}

function getEditableBoxesForItem(item) {
    if (item.annotations_saved) {
        return cloneBoxes(item.annotations || []).map(box => ({
            ...box,
            class_id: 0,
            label_name: 'product',
            points: normalizeBoxPoints(box.points || [])
        })).filter(box => box.points.length === 4);
    }

    return getModelBoxesAsEditable(item);
}

function getModelBoxesAsEditable(item) {
    return (item.prediction.boxes || [])
        .filter(box => box.points && box.points.length === 4)
        .map(box => ({
            class_id: 0,
            label_name: 'product',
            points: normalizeBoxPoints(box.points),
            box_source: 'model_editable',
            confidence: box.confidence ?? null
        }));
}

function drawModelBoxes(ctx, boxes) {
    boxes.forEach((box, index) => {
        const points = box.points;

        if (!points || points.length !== 4) {
            return;
        }

        const cleanPoints = points.map(point => [Number(point[0]), Number(point[1])]);
        drawPolygon(ctx, cleanPoints, '#22c55e', 3);

        const confidence = Number(box.confidence || 0);
        drawBoxText(ctx, cleanPoints, `M${index + 1} conf:${confidence.toFixed(2)}`, '#22c55e');
    });
}

function drawManualBoxes(ctx) {
    manualBoxes.forEach((box, index) => {
        const points = box.points;

        if (!points || points.length !== 4) {
            return;
        }

        const cleanPoints = points.map(point => [Number(point[0]), Number(point[1])]);
        const isSelected = isBoxSelected(index);
        const color = isSelected ? '#38bdf8' : '#f97316';
        drawPolygon(ctx, cleanPoints, color, isSelected ? 4 : 3);

        const confText = box.confidence === null || box.confidence === undefined
            ? ''
            : ` conf:${Number(box.confidence).toFixed(2)}`;
        drawBoxText(ctx, cleanPoints, `A${index + 1} product${confText}`, color);

        if (index === selectedBoxIndex && selectedBoxIndexes.size <= 1) {
            drawEditHandles(ctx, cleanPoints);
        }
    });
}

function drawDraftBox(ctx) {
    if (draftBox && draftBox.points && draftBox.points.length === 4) {
        drawPolygon(ctx, draftBox.points, '#facc15', 3);
    }

    if (selectionRect) {
        const rect = rectFromPoints(selectionRect.startPoint, selectionRect.endPoint);
        ctx.save();
        ctx.setLineDash([8, 6]);
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2;
        ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
        ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
        ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
        ctx.restore();
    }
}

function drawEditHandles(ctx, points) {
    const rotateHandles = getRotateHandlePoints(points);
    const topCenter = midpoint(points[0], points[1]);
    const bottomCenter = midpoint(points[2], points[3]);
    const sideHandles = getSideHandlePoints(points);

    ctx.beginPath();
    ctx.moveTo(topCenter[0], topCenter[1]);
    ctx.lineTo(rotateHandles[0][0], rotateHandles[0][1]);
    ctx.moveTo(bottomCenter[0], bottomCenter[1]);
    ctx.lineTo(rotateHandles[1][0], rotateHandles[1][1]);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.stroke();

    points.forEach(point => {
        ctx.beginPath();
        ctx.rect(point[0] - HANDLE_RADIUS, point[1] - HANDLE_RADIUS, HANDLE_RADIUS * 2, HANDLE_RADIUS * 2);
        ctx.fillStyle = '#38bdf8';
        ctx.fill();
        ctx.strokeStyle = '#111111';
        ctx.lineWidth = 2;
        ctx.stroke();
    });

    sideHandles.forEach(point => {
        ctx.beginPath();
        ctx.rect(point[0] - HANDLE_RADIUS + 1, point[1] - HANDLE_RADIUS + 1, (HANDLE_RADIUS - 1) * 2, (HANDLE_RADIUS - 1) * 2);
        ctx.fillStyle = '#a78bfa';
        ctx.fill();
        ctx.strokeStyle = '#111111';
        ctx.lineWidth = 2;
        ctx.stroke();
    });

    rotateHandles.forEach(point => {
        ctx.beginPath();
        ctx.arc(point[0], point[1], HANDLE_RADIUS + 1, 0, Math.PI * 2);
        ctx.fillStyle = '#facc15';
        ctx.fill();
        ctx.strokeStyle = '#111111';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

function drawPolygon(ctx, points, color, lineWidth) {
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    ctx.lineTo(points[1][0], points[1][1]);
    ctx.lineTo(points[2][0], points[2][1]);
    ctx.lineTo(points[3][0], points[3][1]);
    ctx.closePath();
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;
    ctx.stroke();
}

function drawBoxText(ctx, points, text, color) {
    const x = Math.min(...points.map(point => point[0]));
    const y = Math.min(...points.map(point => point[1]));
    const textX = Math.max(x, 4);
    const textY = Math.max(y - 8, 24);

    ctx.font = '20px Arial';
    const metrics = ctx.measureText(text);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.70)';
    ctx.fillRect(textX - 3, textY - 20, metrics.width + 6, 25);
    ctx.fillStyle = color;
    ctx.fillText(text, textX, textY);
}

function renderInfo(item) {
    const html = `
        <div class="info-row"><span class="label">Queue Item:</span> <span class="value">${formatValue(item.queue_item_id)}</span></div>
        <div class="info-row"><span class="label">Frame ID:</span> <span class="value">${formatValue(item.frame_id)}</span></div>
        <div class="info-row"><span class="label">Video ID:</span> <span class="value">${formatValue(item.video_id)}</span></div>
        <div class="info-row"><span class="label">Order:</span> <span class="value">${escapeHtml(item.order_code)}</span></div>
        <div class="info-row"><span class="label">Camera:</span> <span class="value">${escapeHtml(item.camera)}</span></div>
        <div class="info-row"><span class="label">Frame Number:</span> <span class="value">${formatValue(item.frame_number)}</span></div>
        <div class="info-row"><span class="label">Timeline Index:</span> <span class="value">${currentIndex + 1} / ${frames.length}</span></div>
        <div class="info-row"><span class="label">Timestamp:</span> <span class="value">${formatValue(item.timestamp)}</span></div>
        <div class="info-row"><span class="label">Queue Reason:</span> <span class="value">${escapeHtml(item.queue_reason)}</span></div>
        <div class="info-row"><span class="label">Queue Status:</span> <span class="value">${escapeHtml(item.queue_status)}</span></div>
        <div class="info-row"><span class="label">Prediction:</span> <span class="value">${escapeHtml(item.prediction.status)}</span></div>
        <div class="info-row"><span class="label">Max Confidence:</span> <span class="value">${formatConfidence(item.prediction.max_confidence)}</span></div>
        <div class="info-row"><span class="label">Confidence Filter:</span> <span class="value">≥ ${getConfidenceFilterValue().toFixed(2)}</span></div>
        <div class="info-row"><span class="label">Box Count:</span> <span class="value">${formatValue(item.prediction.box_count)}</span></div>
        <div class="info-row"><span class="label">Editable Boxes:</span> <span class="value">${manualBoxes.length}</span></div>
    `;

    document.getElementById('frameInfo').innerHTML = html;
}

function renderObjects(item) {
    const boxes = item.prediction.boxes || [];
    const container = document.getElementById('objectsList');

    if (boxes.length === 0) {
        container.innerHTML = '<div class="object-item empty-state">No prediction boxes</div>';
        return;
    }

    container.innerHTML = '';

    boxes.forEach((box, index) => {
        const div = document.createElement('div');
        div.className = 'object-item';
        div.innerHTML = `
            <b>Model Object ${index + 1}</b><br>
            class_id: ${formatValue(box.class_id)}<br>
            confidence: ${formatConfidence(box.confidence)}<br>
            type: ${escapeHtml(box.box_type || 'obb')}
        `;
        container.appendChild(div);
    });
}

function renderReview(item) {
    const review = item.review;
    const note = document.getElementById('reviewNote');
    const currentReview = document.getElementById('currentReview');

    document.querySelectorAll('.review-buttons button[data-review-status]').forEach(button => {
        button.classList.toggle('active-review', review && button.dataset.reviewStatus === review.review_status);
    });

    if (!review) {
        currentReview.innerHTML = '<div class="review-card empty-state">No saved review</div>';
        note.value = '';
        return;
    }

    note.value = review.note || '';
    currentReview.innerHTML = `
        <div class="review-card">
            <b>${escapeHtml(review.review_status)}</b><br>
            review_id: ${formatValue(review.id)}<br>
            updated: ${formatValue(review.updated_at || review.created_at)}
        </div>
    `;
}

function renderManualBoxesList() {
    const container = document.getElementById('manualBoxesList');
    const baseStatusText = multiSelectMode
        ? 'Select Many is active: drag a rectangle around boxes; press M to turn it off.'
            : annotationMode === 'draw'
                ? 'Smart Draw is active: drag empty area to draw; click existing boxes to edit without turning it off.'
            : selectedBoxLocked && selectedBoxIndex >= 0
                ? `A${selectedBoxIndex + 1} is locked. Click another box to unlock and switch selection.`
                : 'Click a box to edit; if boxes overlap, the smallest box under the cursor is selected. Drag empty area to pan. Press M for rectangle selection.';

    const unsavedText = annotationsDirty ? ' Unsaved changes will auto-save when you move to another frame.' : '';

    document.getElementById('editorStatus').textContent =
        manualBoxes.length === 0 && annotationMode !== 'draw'
            ? `No annotation boxes. This can be saved as an intentionally empty annotation.${unsavedText}`
            : `${baseStatusText}${unsavedText}`;

    if (manualBoxes.length === 0) {
        container.innerHTML = '<div class="manual-box-item empty-state">No annotation boxes</div>';
        updateAnnotationButtons();
        return;
    }

    container.innerHTML = '';

    manualBoxes.forEach((box, index) => {
        const div = document.createElement('div');
        div.className = 'manual-box-item';

        if (index === selectedBoxIndex) {
            div.classList.add('selected');
        }

        if (selectedBoxIndexes.has(index)) {
            div.classList.add('multi-selected');
        }

        div.onclick = function (event) {
            if (event.shiftKey) {
                toggleBoxMultiSelection(index);
                selectedBoxIndex = index;
                renderManualBoxesList();
                redrawCanvas();
                return;
            }

            if (selectedBoxLocked && index !== selectedBoxIndex) {
                return;
            }

            selectedBoxIndexes.clear();
            selectedBoxIndexes.add(index);
            selectedBoxIndex = index;
            annotationMode = 'select';
            draftBox = null;
            dragState = null;
            renderManualBoxesList();
            redrawCanvas();
        };

        const confText = box.confidence === null || box.confidence === undefined
            ? 'manual'
            : formatConfidence(box.confidence);

        div.innerHTML = `
            <b>Annotation Box ${index + 1}</b><br>
            class: product<br>
            label: product<br>
            confidence: ${confText}<br>
            source: ${escapeHtml(box.box_source || 'manual')}
        `;

        container.appendChild(div);
    });

    updateAnnotationButtons();
}

async function navigateToIndex(targetIndex) {
    if (frames.length === 0) return false;

    const nextIndex = clamp(targetIndex, 0, frames.length - 1);

    if (nextIndex === currentIndex) {
        return true;
    }

    const canLeave = await autoSaveCurrentAnnotationsIfDirty();

    if (!canLeave) {
        pausePlayback();
        return false;
    }

    currentIndex = nextIndex;
    renderCurrentFrame();

    if (isPlaying && currentIndex === frames.length - 1) {
        pausePlayback();
    }

    return true;
}

function stepFrame(step) {
    navigateToIndex(currentIndex + step);
}

function frameHasBoxes(frameItem) {
    if (!frameItem) {
        return false;
    }

    if (frameItem.annotations_saved) {
        return (frameItem.annotations || []).length > 0;
    }

    return Boolean(
        frameItem.prediction &&
        frameItem.prediction.boxes &&
        frameItem.prediction.boxes.length > 0
    );
}

function goPreviousBoxedFrame() {
    for (let index = currentIndex - 1; index >= 0; index -= 1) {
        if (frameHasBoxes(frames[index])) {
            navigateToIndex(index);
            return;
        }
    }
}

function goNextBoxedFrame() {
    for (let index = currentIndex + 1; index < frames.length; index += 1) {
        if (frameHasBoxes(frames[index])) {
            navigateToIndex(index);
            return;
        }
    }
}

function goFirst() {
    navigateToIndex(0);
}

function goLast() {
    navigateToIndex(frames.length - 1).then(() => pausePlayback());
}

function jumpToIndex() {
    if (frames.length === 0) return;

    const value = parseInt(document.getElementById('jumpInput').value, 10);

    if (isNaN(value)) return;

    navigateToIndex(value - 1);
}

function togglePlay() {
    if (isPlaying) {
        pausePlayback();
    } else {
        startPlayback();
    }
}

function startPlayback() {
    if (frames.length === 0) {
        return;
    }

    if (currentIndex >= frames.length - 1) {
        currentIndex = 0;
        renderCurrentFrame();
    }

    pausePlayback(false);
    isPlaying = true;
    updatePlayButton();

    const delayMs = 1000 / getPlaybackFps();
    playbackTimer = window.setInterval(async () => {
        if (playbackAdvancing) {
            return;
        }

        if (currentIndex >= frames.length - 1) {
            pausePlayback();
            return;
        }

        playbackAdvancing = true;
        await navigateToIndex(currentIndex + 1);
        playbackAdvancing = false;
    }, delayMs);
}

function pausePlayback(updateButton = true) {
    if (playbackTimer !== null) {
        window.clearInterval(playbackTimer);
        playbackTimer = null;
    }

    isPlaying = false;

    if (updateButton) {
        updatePlayButton();
    }
}

function handleSpeedChange() {
    if (isPlaying) {
        startPlayback();
    }
}

function getPlaybackFps() {
    return Number(document.getElementById('playbackSpeed').value || 8);
}

function updatePlayButton() {
    const button = document.getElementById('playButton');
    button.textContent = isPlaying ? 'Pause' : 'Play';
}

function updateControlState() {
    const hasFrames = frames.length > 0;

    navButtonIds.forEach(id => {
        document.getElementById(id).disabled = !hasFrames;
    });

    annotationButtonIds.forEach(id => {
        document.getElementById(id).disabled = !hasFrames;
    });

    document.querySelectorAll('.review-buttons button').forEach(button => {
        button.disabled = !hasFrames;
    });

    document.getElementById('jumpInput').disabled = !hasFrames;
    document.getElementById('reviewNote').disabled = !hasFrames;
    document.getElementById('classIdInput').disabled = !hasFrames;
    document.getElementById('labelNameInput').disabled = !hasFrames;
    document.getElementById('showModelBoxesInput').disabled = !hasFrames;
    updatePlayButton();
    updateAnnotationButtons();
}


async function downloadExportZip() {
    const canLeave = await autoSaveCurrentAnnotationsIfDirty();

    if (!canLeave) {
        return;
    }

    const orderCode = document.getElementById('orderFilter').value;
    const camera = document.getElementById('cameraFilter').value;

    if (orderCode === 'all' || camera === 'all') {
        alert('Select one Order and one Camera before exporting.');
        return;
    }

    const params = new URLSearchParams({
        order_code: orderCode,
        camera: camera,
        include_unreviewed_high_confidence: 'false'
    });

    document.getElementById('statusText').textContent = 'Creating export ZIP... Please wait.';
    window.location.href = `/api/export-zip?${params.toString()}`;
}

function updateAnnotationButtons() {
    const addButton = document.getElementById('addBoxButton');
    const lockButton = document.getElementById('lockBoxButton');
    const multiSelectButton = document.getElementById('multiSelectButton');
    const deleteButton = document.getElementById('deleteBoxButton');
    const hasSelectedBox = selectedBoxIndex >= 0 && selectedBoxIndex < manualBoxes.length;
    const selectedCount = selectedBoxIndexes.size;

    addButton.classList.toggle('active-tool', annotationMode === 'draw');
    addButton.textContent = annotationMode === 'draw' ? 'Draw Box: On' : 'Draw Box';

    lockButton.disabled = frames.length === 0 || !hasSelectedBox;
    lockButton.classList.toggle('active-tool', selectedBoxLocked);
    lockButton.textContent = selectedBoxLocked ? 'Unlock Selected' : 'Lock Selected';

    multiSelectButton.disabled = frames.length === 0;
    multiSelectButton.classList.toggle('active-tool', multiSelectMode);
    multiSelectButton.textContent = multiSelectMode ? 'Select Many: On' : 'Select Many';

    deleteButton.disabled = frames.length === 0 || (!hasSelectedBox && selectedCount === 0);
    deleteButton.textContent = selectedCount > 1 ? `Delete ${selectedCount} Boxes` : 'Delete Selected Box';
}

function getAnnotationSnapshot() {
    return {
        boxes: cloneBoxes(manualBoxes),
        selectedBoxIndex,
        selectedBoxIndexes: Array.from(selectedBoxIndexes),
        selectedBoxLocked,
        multiSelectMode,
        annotationMode
    };
}

function restoreAnnotationSnapshot(snapshot) {
    manualBoxes = cloneBoxes(snapshot.boxes || []);
    selectedBoxIndex = snapshot.selectedBoxIndex ?? -1;
    selectedBoxIndexes = new Set(snapshot.selectedBoxIndexes || []);
    selectedBoxLocked = Boolean(snapshot.selectedBoxLocked);
    multiSelectMode = Boolean(snapshot.multiSelectMode);
    annotationMode = snapshot.annotationMode === 'draw' ? 'draw' : 'select';
    draftBox = null;
    dragState = null;
    markAnnotationsDirty();
    renderManualBoxesList();
    renderInfo(frames[currentIndex]);
    redrawCanvas();
}

function pushUndoState() {
    undoStack.push(getAnnotationSnapshot());

    if (undoStack.length > MAX_UNDO_STATES) {
        undoStack.shift();
    }

    redoStack = [];
}

function undoAnnotationChange() {
    if (undoStack.length === 0) {
        return;
    }

    redoStack.push(getAnnotationSnapshot());
    const snapshot = undoStack.pop();
    restoreAnnotationSnapshot(snapshot);
}

function redoAnnotationChange() {
    if (redoStack.length === 0) {
        return;
    }

    undoStack.push(getAnnotationSnapshot());
    const snapshot = redoStack.pop();
    restoreAnnotationSnapshot(snapshot);
}

function copySelectedBox() {
    if (selectedBoxIndex < 0 || selectedBoxIndex >= manualBoxes.length) {
        return;
    }

    boxClipboard = cloneBoxes([manualBoxes[selectedBoxIndex]])[0];
    document.getElementById('editorStatus').textContent = `Copied A${selectedBoxIndex + 1}. Press Ctrl+V to paste.`;
}

function pasteCopiedBox() {
    if (!boxClipboard || frames.length === 0) {
        return;
    }

    pushUndoState();
    const pasted = cloneBoxes([boxClipboard])[0];
    pasted.class_id = 0;
    pasted.label_name = 'product';
    pasted.box_source = 'manual';
    pasted.confidence = null;
    pasted.points = (pasted.points || []).map(point => [
        Number(point[0]),
        Number(point[1])
    ]);

    manualBoxes.push(pasted);
    selectedBoxIndex = manualBoxes.length - 1;
    selectedBoxIndexes.clear();
    selectedBoxIndexes.add(selectedBoxIndex);
    selectedBoxLocked = false;
    markAnnotationsDirty();
    renderManualBoxesList();
    renderInfo(frames[currentIndex]);
    redrawCanvas();
}

function markAnnotationsDirty() {
    if (frames.length === 0) {
        return;
    }

    annotationsDirty = true;
    const item = frames[currentIndex];
    item.annotations = cloneBoxes(manualBoxes);
    item.annotations_saved = true;
}

async function autoSaveCurrentAnnotationsIfDirty() {
    if (!annotationsDirty || frames.length === 0) {
        return true;
    }

    return await saveAnnotations({ silent: true, keepMode: true });
}

async function saveReview(reviewStatus) {
    if (frames.length === 0) return;

    const item = frames[currentIndex];
    const note = document.getElementById('reviewNote').value;

    const res = await fetch('/api/save-review', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            queue_item_id: item.queue_item_id,
            review_status: reviewStatus,
            note: note
        })
    });

    const data = await res.json();

    if (data.success) {
        item.queue_status = data.queue_status;
        item.review = data.review;
        renderInfo(item);
        renderReview(item);
        stepFrame(1);
    } else {
        alert('Failed to save review');
    }
}

async function resetReview() {
    if (frames.length === 0) return;

    const item = frames[currentIndex];
    const res = await fetch('/api/reset-review', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            queue_item_id: item.queue_item_id
        })
    });

    const data = await res.json();

    if (data.success) {
        item.queue_status = data.queue_status;
        item.review = null;
        document.getElementById('reviewNote').value = '';
        renderInfo(item);
        renderReview(item);
    } else {
        alert('Failed to reset review');
    }
}

function copyModelBoxes() {
    if (frames.length === 0) return;

    pushUndoState();
    const item = frames[currentIndex];
    manualBoxes = getModelBoxesAsEditable(item);

    selectedBoxIndex = manualBoxes.length > 0 ? 0 : -1;
    selectedBoxIndexes.clear();
    if (selectedBoxIndex >= 0) {
        selectedBoxIndexes.add(selectedBoxIndex);
    }
    annotationMode = 'select';
    draftBox = null;
    dragState = null;
    selectedBoxLocked = false;
    markAnnotationsDirty();
    renderManualBoxesList();
    renderInfo(item);
    redrawCanvas();
}

function startAddBox() {
    if (frames.length === 0) return;

    annotationMode = annotationMode === 'draw' ? 'select' : 'draw';
    draftBox = null;
    dragState = null;
    selectedBoxLocked = false;

    if (annotationMode === 'draw') {
        selectedBoxIndex = -1;
    }

    pausePlayback();
    renderManualBoxesList();
    redrawCanvas();
}

function toggleSelectedBoxLock() {
    if (selectedBoxIndex < 0 || selectedBoxIndex >= manualBoxes.length) {
        selectedBoxLocked = false;
        updateAnnotationButtons();
        return;
    }

    selectedBoxLocked = !selectedBoxLocked;
    annotationMode = 'select';
    draftBox = null;
    dragState = null;
    renderManualBoxesList();
    redrawCanvas();
}

function deleteSelectedBox() {
    const indexesToDelete = getSelectedIndexesForDelete();

    if (indexesToDelete.length === 0) {
        return;
    }

    pushUndoState();

    indexesToDelete
        .sort((a, b) => b - a)
        .forEach(index => {
            manualBoxes.splice(index, 1);
        });

    selectedBoxIndexes.clear();
    selectedBoxIndex = manualBoxes.length > 0
        ? Math.min(indexesToDelete[indexesToDelete.length - 1] || 0, manualBoxes.length - 1)
        : -1;

    if (selectedBoxIndex >= 0) {
        selectedBoxIndexes.add(selectedBoxIndex);
    }

    selectedBoxLocked = false;
    draftBox = null;
    dragState = null;
    selectionRect = null;

    markAnnotationsDirty();
    renderManualBoxesList();
    renderInfo(frames[currentIndex]);
    redrawCanvas();
}

function clearManualBoxes() {
    pushUndoState();
    manualBoxes = [];
    selectedBoxIndex = -1;
    selectedBoxIndexes.clear();
    annotationMode = 'select';
    selectedBoxLocked = false;
    multiSelectMode = false;
    draftBox = null;
    dragState = null;
    markAnnotationsDirty();
    renderManualBoxesList();
    renderInfo(frames[currentIndex]);
    redrawCanvas();
}

async function saveAnnotations(options = {}) {
    if (frames.length === 0) return false;

    const silent = Boolean(options.silent);
    const keepMode = Boolean(options.keepMode);
    const item = frames[currentIndex];
    const previousMode = annotationMode;
    const previousSelectedIndex = selectedBoxIndex;
    const previousLockState = selectedBoxLocked;

    const res = await fetch('/api/save-annotations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            frame_id: item.frame_id,
            queue_item_id: item.queue_item_id,
            boxes: manualBoxes.map(box => ({
                class_id: 0,
                label_name: 'product',
                points: box.points,
                box_source: box.box_source || 'manual',
                confidence: box.confidence ?? null
            }))
        })
    });

    const data = await res.json();

    if (!data.success) {
        alert('Failed to save annotation boxes');
        return false;
    }

    item.annotations = cloneBoxes(data.boxes || []);
    item.annotations_saved = true;
    manualBoxes = cloneBoxes(item.annotations);
    annotationsDirty = false;

    if (keepMode) {
        annotationMode = previousMode;
        selectedBoxIndex = manualBoxes.length > 0
            ? clamp(previousSelectedIndex, 0, manualBoxes.length - 1)
            : -1;
        selectedBoxIndexes.clear();
        if (selectedBoxIndex >= 0) {
            selectedBoxIndexes.add(selectedBoxIndex);
        }
        selectedBoxLocked = previousLockState && selectedBoxIndex >= 0;
    } else {
        selectedBoxIndex = manualBoxes.length > 0
            ? clamp(previousSelectedIndex, 0, manualBoxes.length - 1)
            : -1;
        selectedBoxIndexes.clear();
        if (selectedBoxIndex >= 0) {
            selectedBoxIndexes.add(selectedBoxIndex);
        }
        selectedBoxLocked = false;
    }

    draftBox = null;
    dragState = null;
    renderManualBoxesList();
    renderInfo(item);
    redrawCanvas();

    if (!silent) {
        document.getElementById('editorStatus').textContent = `Saved ${manualBoxes.length} product box${manualBoxes.length === 1 ? '' : 'es'}`;
    }

    return true;
}

function getCanvasPoint(event) {
    const canvas = document.getElementById('imageCanvas');
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return [
        clamp((event.clientX - rect.left) * scaleX, 0, canvas.width),
        clamp((event.clientY - rect.top) * scaleY, 0, canvas.height)
    ];
}

function findBoxAtPoint(point) {
    const hits = [];

    manualBoxes.forEach((box, index) => {
        const points = box.points || [];

        if (pointInPolygon(point, points)) {
            hits.push({
                index,
                area: polygonArea(points)
            });
        }
    });

    if (hits.length === 0) {
        return -1;
    }

    hits.sort((a, b) => {
        if (a.area !== b.area) {
            return a.area - b.area;
        }

        return b.index - a.index;
    });

    return hits[0].index;
}

function findEditHandle(point) {
    if (selectedBoxIndex < 0 || selectedBoxIndex >= manualBoxes.length) {
        return null;
    }

    const points = manualBoxes[selectedBoxIndex].points || [];

    if (points.length !== 4) {
        return null;
    }

    const rotateHandles = getRotateHandlePoints(points);

    for (let index = 0; index < rotateHandles.length; index += 1) {
        if (pointDistance(point, rotateHandles[index]) <= HANDLE_RADIUS + 8) {
            return {
                type: 'rotate',
                boxIndex: selectedBoxIndex,
                handleIndex: index
            };
        }
    }

    const sideHandles = getSideHandlePoints(points);

    for (let index = 0; index < sideHandles.length; index += 1) {
        if (pointDistance(point, sideHandles[index]) <= HANDLE_RADIUS + 8) {
            return {
                type: 'side-resize',
                boxIndex: selectedBoxIndex,
                sideIndex: index
            };
        }
    }

    for (let index = 0; index < points.length; index += 1) {
        if (pointDistance(point, points[index]) <= HANDLE_RADIUS + 8) {
            return {
                type: 'resize',
                boxIndex: selectedBoxIndex,
                cornerIndex: index
            };
        }
    }

    return null;
}

function pointDistance(a, b) {
    const dx = Number(a[0]) - Number(b[0]);
    const dy = Number(a[1]) - Number(b[1]);
    return Math.sqrt(dx * dx + dy * dy);
}

function pointInPolygon(point, polygon) {
    if (!polygon || polygon.length < 3) {
        return false;
    }

    const x = point[0];
    const y = point[1];
    let inside = false;

    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = Number(polygon[i][0]);
        const yi = Number(polygon[i][1]);
        const xj = Number(polygon[j][0]);
        const yj = Number(polygon[j][1]);
        const intersects = ((yi > y) !== (yj > y)) &&
            (x < (xj - xi) * (y - yi) / ((yj - yi) || 0.000001) + xi);

        if (intersects) {
            inside = !inside;
        }
    }

    return inside;
}

function polygonArea(polygon) {
    if (!polygon || polygon.length < 3) {
        return Number.POSITIVE_INFINITY;
    }

    let sum = 0;

    for (let index = 0; index < polygon.length; index += 1) {
        const nextIndex = (index + 1) % polygon.length;
        const x1 = Number(polygon[index][0]);
        const y1 = Number(polygon[index][1]);
        const x2 = Number(polygon[nextIndex][0]);
        const y2 = Number(polygon[nextIndex][1]);

        sum += x1 * y2 - x2 * y1;
    }

    return Math.abs(sum) / 2;
}

function midpoint(a, b) {
    return [
        (Number(a[0]) + Number(b[0])) / 2,
        (Number(a[1]) + Number(b[1])) / 2
    ];
}

function dot(a, b) {
    return Number(a[0]) * Number(b[0]) + Number(a[1]) * Number(b[1]);
}

function getBoxGeometry(points) {
    if (!points || points.length !== 4) {
        return null;
    }

    const cleanPoints = points.map(point => [Number(point[0]), Number(point[1])]);
    const center = [
        cleanPoints.reduce((sum, point) => sum + point[0], 0) / 4,
        cleanPoints.reduce((sum, point) => sum + point[1], 0) / 4
    ];
    const width = pointDistance(cleanPoints[0], cleanPoints[1]);
    const height = pointDistance(cleanPoints[1], cleanPoints[2]);
    const angle = Math.atan2(
        cleanPoints[1][1] - cleanPoints[0][1],
        cleanPoints[1][0] - cleanPoints[0][0]
    );

    return {
        center,
        width,
        height,
        angle
    };
}

function pointsFromGeometry(center, width, height, angle) {
    const halfWidth = Math.max(width, MIN_BOX_SIZE) / 2;
    const halfHeight = Math.max(height, MIN_BOX_SIZE) / 2;
    const ux = [Math.cos(angle), Math.sin(angle)];
    const uy = [-Math.sin(angle), Math.cos(angle)];

    return [
        [
            center[0] - ux[0] * halfWidth - uy[0] * halfHeight,
            center[1] - ux[1] * halfWidth - uy[1] * halfHeight
        ],
        [
            center[0] + ux[0] * halfWidth - uy[0] * halfHeight,
            center[1] + ux[1] * halfWidth - uy[1] * halfHeight
        ],
        [
            center[0] + ux[0] * halfWidth + uy[0] * halfHeight,
            center[1] + ux[1] * halfWidth + uy[1] * halfHeight
        ],
        [
            center[0] - ux[0] * halfWidth + uy[0] * halfHeight,
            center[1] - ux[1] * halfWidth + uy[1] * halfHeight
        ]
    ];
}

function getRotateHandlePoints(points) {
    const geometry = getBoxGeometry(points);

    if (!geometry) {
        return [[0, 0], [0, 0]];
    }

    const topCenter = midpoint(points[0], points[1]);
    const bottomCenter = midpoint(points[2], points[3]);

    const topVector = [topCenter[0] - geometry.center[0], topCenter[1] - geometry.center[1]];
    const bottomVector = [bottomCenter[0] - geometry.center[0], bottomCenter[1] - geometry.center[1]];
    const topLength = Math.max(Math.sqrt(topVector[0] * topVector[0] + topVector[1] * topVector[1]), 1);
    const bottomLength = Math.max(Math.sqrt(bottomVector[0] * bottomVector[0] + bottomVector[1] * bottomVector[1]), 1);

    return [
        [topCenter[0] + (topVector[0] / topLength) * ROTATE_HANDLE_OFFSET, topCenter[1] + (topVector[1] / topLength) * ROTATE_HANDLE_OFFSET],
        [bottomCenter[0] + (bottomVector[0] / bottomLength) * ROTATE_HANDLE_OFFSET, bottomCenter[1] + (bottomVector[1] / bottomLength) * ROTATE_HANDLE_OFFSET]
    ];
}

function getRotateHandlePoint(points) {
    return getRotateHandlePoints(points)[0];
}

function getSideHandlePoints(points) {
    if (!points || points.length !== 4) {
        return [];
    }

    return [
        midpoint(points[0], points[1]),
        midpoint(points[1], points[2]),
        midpoint(points[2], points[3]),
        midpoint(points[3], points[0])
    ];
}

function normalizeBoxPoints(points) {
    if (!points || points.length !== 4) {
        return [];
    }

    const cleanPoints = points.map(point => [Number(point[0]), Number(point[1])]);
    const center = [
        cleanPoints.reduce((sum, point) => sum + point[0], 0) / 4,
        cleanPoints.reduce((sum, point) => sum + point[1], 0) / 4
    ];

    const sorted = cleanPoints.slice().sort((a, b) => {
        const angleA = Math.atan2(a[1] - center[1], a[0] - center[0]);
        const angleB = Math.atan2(b[1] - center[1], b[0] - center[0]);
        return angleA - angleB;
    });

    let startIndex = 0;
    let bestScore = Infinity;

    sorted.forEach((point, index) => {
        const score = point[0] + point[1];
        if (score < bestScore) {
            bestScore = score;
            startIndex = index;
        }
    });

    return [0, 1, 2, 3].map(offset => sorted[(startIndex + offset) % 4]);
}

function rectFromPoints(a, b) {
    const x = Math.min(Number(a[0]), Number(b[0]));
    const y = Math.min(Number(a[1]), Number(b[1]));
    const width = Math.abs(Number(a[0]) - Number(b[0]));
    const height = Math.abs(Number(a[1]) - Number(b[1]));
    return { x, y, width, height };
}

function getBoxCenter(points) {
    return [
        points.reduce((sum, point) => sum + Number(point[0]), 0) / points.length,
        points.reduce((sum, point) => sum + Number(point[1]), 0) / points.length
    ];
}

function pointInRect(point, rect) {
    return point[0] >= rect.x && point[0] <= rect.x + rect.width && point[1] >= rect.y && point[1] <= rect.y + rect.height;
}

function isBoxSelected(index) {
    return selectedBoxIndexes.has(index) || index === selectedBoxIndex;
}

function toggleBoxMultiSelection(index) {
    if (selectedBoxIndexes.has(index)) {
        selectedBoxIndexes.delete(index);
    } else {
        selectedBoxIndexes.add(index);
    }
    selectedBoxIndex = index;
}

function getSelectedIndexesForDelete() {
    const indexes = Array.from(selectedBoxIndexes).filter(index => index >= 0 && index < manualBoxes.length);
    if (indexes.length > 0) return indexes;
    if (selectedBoxIndex >= 0 && selectedBoxIndex < manualBoxes.length) return [selectedBoxIndex];
    return [];
}

function toggleMultiSelectMode() {
    if (frames.length === 0) return;
    multiSelectMode = !multiSelectMode;
    annotationMode = multiSelectMode ? 'select' : annotationMode;
    selectedBoxLocked = false;
    selectionRect = null;
    dragState = null;
    draftBox = null;
    renderManualBoxesList();
    redrawCanvas();
}

function getConfidenceFilterValue() {
    const input = document.getElementById('confidenceFilter');
    if (!input) return 0.25;
    const value = parseFloat(input.value);
    if (isNaN(value)) return 0.25;
    return clamp(value, 0, 1);
}

function createBoxFromDrag(startPoint, endPoint) {
    const x1 = Number(startPoint[0]);
    const y1 = Number(startPoint[1]);
    const x2 = Number(endPoint[0]);
    const y2 = Number(endPoint[1]);
    const minX = Math.min(x1, x2);
    const maxX = Math.max(x1, x2);
    const minY = Math.min(y1, y2);
    const maxY = Math.max(y1, y2);

    return {
        class_id: 0,
        label_name: 'product',
        points: [
            [minX, minY],
            [maxX, minY],
            [maxX, maxY],
            [minX, maxY]
        ],
        box_source: 'manual',
        confidence: null
    };
}

function boxMeetsMinimumSize(box) {
    const geometry = getBoxGeometry(box.points || []);
    return geometry && geometry.width >= MIN_BOX_SIZE && geometry.height >= MIN_BOX_SIZE;
}

function moveSelectedBox(point) {
    const dx = point[0] - dragState.lastPoint[0];
    const dy = point[1] - dragState.lastPoint[1];
    const box = manualBoxes[dragState.boxIndex];

    box.points = box.points.map(boxPoint => [
        Number(boxPoint[0]) + dx,
        Number(boxPoint[1]) + dy
    ]);

    dragState.lastPoint = point;
}

function moveSelectedBoxes(point) {
    const dx = point[0] - dragState.lastPoint[0];
    const dy = point[1] - dragState.lastPoint[1];
    const indexes = (dragState.boxIndexes || [])
        .filter(index => index >= 0 && index < manualBoxes.length);

    indexes.forEach(index => {
        const box = manualBoxes[index];
        box.points = (box.points || []).map(boxPoint => [
            Number(boxPoint[0]) + dx,
            Number(boxPoint[1]) + dy
        ]);
    });

    dragState.lastPoint = point;
}

function isPointInsideAnySelectedBox(point) {
    return Array.from(selectedBoxIndexes).some(index => {
        if (index < 0 || index >= manualBoxes.length) {
            return false;
        }

        return pointInPolygon(point, manualBoxes[index].points || []);
    });
}

function resizeSelectedBox(point) {
    const box = manualBoxes[dragState.boxIndex];
    const basePoints = dragState.startPoints || box.points || [];
    const geometry = getBoxGeometry(basePoints);

    if (!geometry) {
        return;
    }

    const signs = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1]
    ];
    const cornerIndex = dragState.cornerIndex;
    const oppositePoint = basePoints[(cornerIndex + 2) % 4];
    const sx = signs[cornerIndex][0];
    const sy = signs[cornerIndex][1];
    const ux = [Math.cos(geometry.angle), Math.sin(geometry.angle)];
    const uy = [-Math.sin(geometry.angle), Math.cos(geometry.angle)];
    const vector = [
        point[0] - oppositePoint[0],
        point[1] - oppositePoint[1]
    ];
    const width = Math.max(dot(vector, ux) * sx, MIN_BOX_SIZE);
    const height = Math.max(dot(vector, uy) * sy, MIN_BOX_SIZE);
    const center = [
        oppositePoint[0] + (sx * ux[0] * width + sy * uy[0] * height) / 2,
        oppositePoint[1] + (sx * ux[1] * width + sy * uy[1] * height) / 2
    ];

    box.points = pointsFromGeometry(center, width, height, geometry.angle);
}

function resizeSideSelectedBox(point) {
    const box = manualBoxes[dragState.boxIndex];
    const basePoints = dragState.startPoints || box.points || [];
    const geometry = getBoxGeometry(basePoints);

    if (!geometry) {
        return;
    }

    const ux = [Math.cos(geometry.angle), Math.sin(geometry.angle)];
    const uy = [-Math.sin(geometry.angle), Math.cos(geometry.angle)];
    const sideIndex = dragState.sideIndex;
    let center = geometry.center;
    let width = geometry.width;
    let height = geometry.height;

    if (sideIndex === 0 || sideIndex === 2) {
        const sy = sideIndex === 0 ? -1 : 1;
        const oppositeCenter = sideIndex === 0
            ? midpoint(basePoints[2], basePoints[3])
            : midpoint(basePoints[0], basePoints[1]);
        const vector = [
            point[0] - oppositeCenter[0],
            point[1] - oppositeCenter[1]
        ];

        height = Math.max(dot(vector, uy) * sy, MIN_BOX_SIZE);
        center = [
            oppositeCenter[0] + (sy * uy[0] * height) / 2,
            oppositeCenter[1] + (sy * uy[1] * height) / 2
        ];
    } else {
        const sx = sideIndex === 3 ? -1 : 1;
        const oppositeCenter = sideIndex === 3
            ? midpoint(basePoints[1], basePoints[2])
            : midpoint(basePoints[3], basePoints[0]);
        const vector = [
            point[0] - oppositeCenter[0],
            point[1] - oppositeCenter[1]
        ];

        width = Math.max(dot(vector, ux) * sx, MIN_BOX_SIZE);
        center = [
            oppositeCenter[0] + (sx * ux[0] * width) / 2,
            oppositeCenter[1] + (sx * ux[1] * width) / 2
        ];
    }

    box.points = pointsFromGeometry(center, width, height, geometry.angle);
}

function rotateSelectedBox(point) {
    const box = manualBoxes[dragState.boxIndex];
    const basePoints = dragState.startPoints || box.points || [];
    const geometry = getBoxGeometry(basePoints);

    if (!geometry) {
        return;
    }

    const angleToHandle = Math.atan2(point[1] - geometry.center[1], point[0] - geometry.center[0]);
    const angle = dragState.handleIndex === 1 ? angleToHandle - Math.PI / 2 : angleToHandle + Math.PI / 2;

    box.points = pointsFromGeometry(geometry.center, geometry.width, geometry.height, angle);
}

function updateCanvasCursor(point) {
    const canvas = document.getElementById('imageCanvas');
    const handle = findEditHandle(point);

    if (handle && handle.type === 'rotate') {
        canvas.style.cursor = 'grab';
    } else if (handle && handle.type === 'resize') {
        canvas.style.cursor = 'nwse-resize';
    } else if (handle && handle.type === 'side-resize') {
        canvas.style.cursor = handle.sideIndex === 0 || handle.sideIndex === 2 ? 'ns-resize' : 'ew-resize';
    } else if (selectedBoxIndexes.size > 1 && isPointInsideAnySelectedBox(point)) {
        canvas.style.cursor = 'move';
    } else if (!selectedBoxLocked && findBoxAtPoint(point) >= 0) {
        canvas.style.cursor = 'move';
    } else if (selectedBoxLocked && selectedBoxIndex >= 0 && pointInPolygon(point, manualBoxes[selectedBoxIndex].points || [])) {
        canvas.style.cursor = 'move';
    } else if (multiSelectMode) {
        canvas.style.cursor = 'crosshair';
    } else if (annotationMode === 'draw' || window.event?.altKey) {
        canvas.style.cursor = 'crosshair';
    } else {
        canvas.style.cursor = 'grab';
    }
}

function handleCanvasMouseDown(event) {
    if (frames.length === 0 || !currentImage) return;

    if (event.button === 1 || event.button === 2) {
        startCanvasPan(event);
        return;
    }

    const point = getCanvasPoint(event);
    event.preventDefault();

    const handle = findEditHandle(point);
    const hitBoxIndex = findBoxAtPoint(point);

    if ((event.ctrlKey || event.metaKey) && hitBoxIndex >= 0) {
        selectedBoxIndex = hitBoxIndex;
        selectedBoxIndexes.clear();
        selectedBoxIndexes.add(hitBoxIndex);
        selectedBoxLocked = true;
        multiSelectMode = false;
        annotationMode = 'select';
        draftBox = null;
        dragState = null;
        renderManualBoxesList();
        redrawCanvas();
        return;
    }

    if (event.shiftKey && hitBoxIndex >= 0 && !handle) {
        toggleBoxMultiSelection(hitBoxIndex);
        selectedBoxLocked = false;
        multiSelectMode = false;
        annotationMode = 'select';
        draftBox = null;
        dragState = null;
        renderManualBoxesList();
        redrawCanvas();
        return;
    }

    if (handle) {
        pushUndoState();
        selectedBoxIndex = handle.boxIndex;
        selectedBoxIndexes.clear();
        selectedBoxIndexes.add(selectedBoxIndex);
        dragState = { ...handle, startPoints: cloneBoxes([{ points: manualBoxes[handle.boxIndex].points }])[0].points };
        renderManualBoxesList();
        redrawCanvas();
        return;
    }

    if (selectedBoxIndexes.size > 1 && hitBoxIndex >= 0 && selectedBoxIndexes.has(hitBoxIndex) && !selectedBoxLocked) {
        pushUndoState();
        selectedBoxIndex = hitBoxIndex;
        dragState = {
            type: 'multi-move',
            boxIndexes: Array.from(selectedBoxIndexes),
            lastPoint: point
        };
        renderManualBoxesList();
        redrawCanvas();
        return;
    }

    if (multiSelectMode) {
        if (hitBoxIndex >= 0 && !handle) {
            toggleBoxMultiSelection(hitBoxIndex);
            selectedBoxLocked = false;
            renderManualBoxesList();
            redrawCanvas();
            return;
        }

        selectionRect = { startPoint: point, endPoint: point };
        dragState = { type: 'multi-select', startPoint: point };
        redrawCanvas();
        return;
    }

    if (selectedBoxLocked && selectedBoxIndex >= 0) {
        if (hitBoxIndex >= 0 && hitBoxIndex !== selectedBoxIndex) {
            pushUndoState();
            selectedBoxIndex = hitBoxIndex;
            selectedBoxIndexes.clear();
            selectedBoxIndexes.add(selectedBoxIndex);
            selectedBoxLocked = false;
            dragState = { type: 'move', boxIndex: selectedBoxIndex, lastPoint: point };
        } else if (pointInPolygon(point, manualBoxes[selectedBoxIndex].points || [])) {
            pushUndoState();
            dragState = { type: 'move', boxIndex: selectedBoxIndex, lastPoint: point };
        } else {
            startCanvasPan(event);
        }
        renderManualBoxesList();
        redrawCanvas();
        return;
    }

    if (hitBoxIndex >= 0) {
        pushUndoState();
        selectedBoxIndex = hitBoxIndex;
        selectedBoxIndexes.clear();
        selectedBoxIndexes.add(selectedBoxIndex);
        dragState = { type: 'move', boxIndex: selectedBoxIndex, lastPoint: point };
        renderManualBoxesList();
        redrawCanvas();
        return;
    }

    if (event.altKey || annotationMode === 'draw') {
        pushUndoState();
        draftBox = createBoxFromDrag(point, point);
        dragState = { type: 'draw', startPoint: point };
        renderManualBoxesList();
        redrawCanvas();
        return;
    }

    startCanvasPan(event);
}

function handleCanvasMouseMove(event) {
    if (frames.length === 0 || !currentImage) {
        return;
    }

    if (canvasPanState) {
        return;
    }

    const point = getCanvasPoint(event);
    updateCanvasCursor(point);

    if (!dragState) {
        return;
    }

    event.preventDefault();

    if (dragState.type === 'multi-select') {
        selectionRect = { startPoint: dragState.startPoint, endPoint: point };
    } else if (dragState.type === 'draw') {
        draftBox = createBoxFromDrag(dragState.startPoint, point);
    } else if (dragState.type === 'move') {
        moveSelectedBox(point);
    } else if (dragState.type === 'multi-move') {
        moveSelectedBoxes(point);
    } else if (dragState.type === 'resize') {
        resizeSelectedBox(point);
    } else if (dragState.type === 'side-resize') {
        resizeSideSelectedBox(point);
    } else if (dragState.type === 'rotate') {
        rotateSelectedBox(point);
    }

    redrawCanvas();
}

function handleCanvasMouseUp() {
    let changed = false;

    if (dragState && dragState.type === 'multi-select' && selectionRect) {
        const rect = rectFromPoints(selectionRect.startPoint, selectionRect.endPoint);
        if (rect.width >= 3 && rect.height >= 3) {
            selectedBoxIndexes.clear();
            manualBoxes.forEach((box, index) => {
                const points = box.points || [];
                if (points.length !== 4) return;
                if (pointInRect(getBoxCenter(points), rect)) {
                    selectedBoxIndexes.add(index);
                }
            });
            selectedBoxIndex = selectedBoxIndexes.size > 0 ? Array.from(selectedBoxIndexes).sort((a, b) => a - b)[0] : -1;
        }
    } else if (dragState && dragState.type === 'draw' && draftBox) {
        if (boxMeetsMinimumSize(draftBox)) {
            manualBoxes.push(draftBox);
            selectedBoxIndex = manualBoxes.length - 1;
            selectedBoxIndexes.clear();
            selectedBoxIndexes.add(selectedBoxIndex);
            changed = true;
        }
    } else if (dragState && ['move', 'multi-move', 'resize', 'side-resize', 'rotate'].includes(dragState.type)) {
        changed = true;
    }

    draftBox = null;
    dragState = null;
    selectionRect = null;

    if (changed) {
        markAnnotationsDirty();
    }

    renderManualBoxesList();
    if (frames.length > 0) renderInfo(frames[currentIndex]);
    redrawCanvas();
}

function cloneBoxes(boxes) {
    return JSON.parse(JSON.stringify(boxes || []));
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function formatValue(value) {
    if (value === null || value === undefined) {
        return '';
    }

    return escapeHtml(String(value));
}

function formatConfidence(value) {
    const numberValue = Number(value || 0);
    return numberValue.toFixed(3);
}

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

document.addEventListener('keydown', function(event) {
    const activeTag = document.activeElement.tagName.toLowerCase();
    const isTyping = activeTag === 'input' || activeTag === 'textarea' || activeTag === 'select';
    const key = event.key.toLowerCase();
    const ctrlOrMeta = event.ctrlKey || event.metaKey;

    if (isTyping) {
        return;
    }

    if (ctrlOrMeta && key === 'c') {
        event.preventDefault();
        copySelectedBox();
        return;
    }

    if (ctrlOrMeta && key === 'v') {
        event.preventDefault();
        pasteCopiedBox();
        return;
    }

    if (ctrlOrMeta && key === 'z') {
        event.preventDefault();
        if (event.shiftKey) {
            redoAnnotationChange();
        } else {
            undoAnnotationChange();
        }
        return;
    }

    if (ctrlOrMeta && key === 'y') {
        event.preventDefault();
        redoAnnotationChange();
        return;
    }

    if (event.code === 'Space') {
        event.preventDefault();
        togglePlay();
        return;
    }

    if (key === 'c') {
        event.preventDefault();
        stepFrame(-10);
    } else if (key === 'd') {
        event.preventDefault();
        stepFrame(-1);
    } else if (key === 'f') {
        event.preventDefault();
        stepFrame(1);
    } else if (event.key === 'ArrowLeft') {
        event.preventDefault();
        goPreviousBoxedFrame();
    } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        goNextBoxedFrame();
    } else if (key === 'v') {
        event.preventDefault();
        stepFrame(10);
    } else if (key === 'm') {
        event.preventDefault();
        toggleMultiSelectMode();
    } else if (event.key === 'Home') {
        event.preventDefault();
        goFirst();
    } else if (event.key === 'End') {
        event.preventDefault();
        goLast();
    } else if (event.key === 'Delete') {
        event.preventDefault();
        deleteSelectedBox();
    } else if (event.key === 'Escape') {
        event.preventDefault();
        annotationMode = 'select';
        multiSelectMode = false;
        selectionRect = null;
        draftBox = null;
        dragState = null;
        selectedBoxLocked = false;
        renderManualBoxesList();
        redrawCanvas();
    }
});

async function init() {
    const canvas = document.getElementById('imageCanvas');
    const canvasPanel = getCanvasPanel();
    canvas.addEventListener('mousedown', handleCanvasMouseDown);
    canvas.addEventListener('mousemove', handleCanvasMouseMove);
    canvas.addEventListener('wheel', handleCanvasWheel, { passive: false });
    canvas.addEventListener('contextmenu', event => event.preventDefault());
    canvasPanel.addEventListener('mousedown', handleCanvasPanelMouseDown);
    canvasPanel.addEventListener('contextmenu', event => event.preventDefault());
    window.addEventListener('mousemove', handleCanvasPanMove);
    window.addEventListener('mouseup', handleCanvasMouseUp);
    window.addEventListener('mouseup', stopCanvasPan);
    window.addEventListener('resize', updateCanvasDisplaySize);

    updateControlState();
    clearCanvas();
    await loadFilters();
    await loadFrames();
}

init();
</script>
</body>
</html>
"""


@app.get("/api/filters")
def get_filters(db: Session = Depends(get_db)):
    order_rows = db.query(Frame.order_code).distinct().order_by(Frame.order_code).all()
    camera_rows = db.query(Frame.camera).distinct().order_by(Frame.camera).all()
    reason_rows = db.query(QueueItem.queue_reason).distinct().order_by(QueueItem.queue_reason).all()

    return {
        "order_codes": [row[0] for row in order_rows if row[0]],
        "cameras": [row[0] for row in camera_rows if row[0]],
        "queue_reasons": [row[0] for row in reason_rows if row[0]],
    }


@app.get("/api/frames")
def get_frames(
    queue_reason: str = "all",
    status: str = "all",
    order_code: str = "all",
    camera: str = "all",
    min_confidence: float = 0.0,
    db: Session = Depends(get_db),
):
    query = (
        db.query(QueueItem, Frame, Prediction)
        .join(Frame, QueueItem.frame_id == Frame.id)
        .outerjoin(Prediction, QueueItem.prediction_id == Prediction.id)
    )

    if queue_reason != "all":
        query = query.filter(QueueItem.queue_reason == queue_reason)

    if status != "all":
        query = query.filter(QueueItem.status == status)

    if order_code != "all":
        query = query.filter(Frame.order_code == order_code)

    if camera != "all":
        query = query.filter(Frame.camera == camera)

    rows = (
        query.order_by(
            Frame.order_code,
            Frame.camera,
            Frame.video_id,
            Frame.frame_number,
            Frame.id,
        )
        .all()
    )

    queue_item_ids = [queue_item.id for queue_item, _, _ in rows]
    frame_ids = [frame.id for _, frame, _ in rows]

    reviews_by_queue_item = {}

    if queue_item_ids:
        reviews = (
            db.query(Review)
            .filter(Review.queue_item_id.in_(queue_item_ids))
            .order_by(Review.queue_item_id, Review.id.desc())
            .all()
        )

        for review in reviews:
            if review.queue_item_id not in reviews_by_queue_item:
                reviews_by_queue_item[review.queue_item_id] = review

    annotations_by_frame = {}
    saved_annotation_frame_ids = set()

    if frame_ids:
        saved_annotation_frame_ids = {
            row[0]
            for row in (
                db.query(Annotation.frame_id)
                .filter(Annotation.frame_id.in_(frame_ids))
                .filter(Annotation.annotation_source == "manual_draw")
                .filter(Annotation.status == "saved")
                .distinct()
                .all()
            )
        }

        annotation_rows = (
            db.query(Annotation, AnnotationBox)
            .join(AnnotationBox, AnnotationBox.annotation_id == Annotation.id)
            .filter(Annotation.frame_id.in_(frame_ids))
            .filter(Annotation.annotation_source == "manual_draw")
            .filter(Annotation.status == "saved")
            .order_by(Annotation.frame_id, AnnotationBox.id)
            .all()
        )

        for annotation, box in annotation_rows:
            annotations_by_frame.setdefault(annotation.frame_id, []).append(
                {
                    "annotation_id": annotation.id,
                    "annotation_box_id": box.id,
                    "class_id": box.class_id,
                    "label_name": box.label_name,
                    "box_type": box.box_type,
                    "points": box.points_json,
                    "normalized_points": box.normalized_points_json,
                    "confidence": box.confidence,
                    "box_source": box.box_source,
                }
            )

    data = []

    for queue_item, frame, prediction in rows:
        boxes = []
        prediction_id = None
        prediction_status = "missing_prediction"
        max_confidence = 0.0
        box_count = 0

        if prediction:
            raw_boxes = prediction.boxes_json or []
            boxes = [
                box for box in raw_boxes
                if float(box.get("confidence") or 0.0) >= min_confidence
            ]
            prediction_id = prediction.id
            prediction_status = prediction.prediction_status
            max_confidence = max([float(box.get("confidence") or 0.0) for box in boxes], default=0.0)
            box_count = len(boxes)

        data.append(
            {
                "queue_item_id": queue_item.id,
                "queue_type": queue_item.queue_type,
                "queue_reason": queue_item.queue_reason,
                "queue_status": queue_item.status,
                "decision": queue_item.decision,
                "warning": queue_item.warning,

                "frame_id": frame.id,
                "video_id": frame.video_id,
                "order_code": frame.order_code,
                "camera": frame.camera,
                "frame_number": frame.frame_number,
                "frame_path": frame.frame_path,
                "timestamp": frame.timestamp,
                "width": frame.width,
                "height": frame.height,

                "prediction": {
                    "id": prediction_id,
                    "status": prediction_status,
                    "max_confidence": max_confidence,
                    "box_count": box_count,
                    "boxes": boxes,
                },
                "review": review_to_dict(reviews_by_queue_item.get(queue_item.id)),
                "annotations": annotations_by_frame.get(frame.id, []),
                "annotations_saved": frame.id in saved_annotation_frame_ids,
            }
        )

    return data


@app.get("/api/image/{frame_id}")
def get_image(frame_id: int, db: Session = Depends(get_db)):
    frame = db.query(Frame).filter(Frame.id == frame_id).first()

    if not frame:
        raise HTTPException(status_code=404, detail="Frame not found")

    image_path = resolve_path(frame.frame_path)

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {image_path}")

    return FileResponse(str(image_path))



def safe_export_part(value: str) -> str:
    text = str(value or "all")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "all"


def cvat_export_name(order_code: str, camera: str, timestamp: str) -> str:
    task_parts = [safe_export_part(order_code)]

    if camera != "all":
        task_parts.append(safe_export_part(camera))

    task_name = "_".join(task_parts)

    return (
        f"task_{task_name}_annotations_{timestamp}_"
        "ultralytics yolo oriented bounding boxes 1.0"
    )


@app.get("/api/export-zip")
def export_zip(
    order_code: str = "all",
    camera: str = "all",
    include_unreviewed_high_confidence: bool = False,
):
    if order_code == "all" or camera == "all":
        raise HTTPException(
            status_code=400,
            detail="Select one Order and one Camera before exporting.",
        )

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    export_name = cvat_export_name(order_code, camera, timestamp)
    output_dir = Path("exports") / "web_downloads" / export_name

    args = SimpleNamespace(
        output_dir=str(output_dir),
        order_code=order_code,
        camera=camera,
        include_empty_labels=False,
        include_unreviewed_high_confidence=include_unreviewed_high_confidence,
        zip=True,
    )

    result = export_dataset(args)
    zip_path_text = result.get("zip_path")

    if not zip_path_text:
        raise HTTPException(status_code=500, detail="Export ZIP was not created")

    zip_path = Path(zip_path_text)

    if not zip_path.exists():
        raise HTTPException(status_code=500, detail=f"ZIP file not found: {zip_path}")

    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=f"{export_name}.zip",
    )

@app.get("/api/annotations/{frame_id}")
def get_annotations(frame_id: int, db: Session = Depends(get_db)):
    frame = db.query(Frame).filter(Frame.id == frame_id).first()

    if not frame:
        raise HTTPException(status_code=404, detail="Frame not found")

    annotations_saved = (
        db.query(Annotation)
        .filter(Annotation.frame_id == frame_id)
        .filter(Annotation.annotation_source == "manual_draw")
        .filter(Annotation.status == "saved")
        .first()
        is not None
    )

    rows = (
        db.query(Annotation, AnnotationBox)
        .join(AnnotationBox, AnnotationBox.annotation_id == Annotation.id)
        .filter(Annotation.frame_id == frame_id)
        .filter(Annotation.annotation_source == "manual_draw")
        .filter(Annotation.status == "saved")
        .order_by(AnnotationBox.id)
        .all()
    )

    boxes = []

    for annotation, box in rows:
        boxes.append(
            {
                "annotation_id": annotation.id,
                "annotation_box_id": box.id,
                "class_id": box.class_id,
                "label_name": box.label_name,
                "box_type": box.box_type,
                "points": box.points_json,
                "normalized_points": box.normalized_points_json,
                "confidence": box.confidence,
                "box_source": box.box_source,
            }
        )

    return {
        "frame_id": frame_id,
        "annotations_saved": annotations_saved,
        "boxes": boxes,
    }


@app.post("/api/save-review")
def save_review(request: ReviewRequest, db: Session = Depends(get_db)):
    queue_item = db.query(QueueItem).filter(QueueItem.id == request.queue_item_id).first()

    if not queue_item:
        raise HTTPException(status_code=404, detail="Queue item not found")

    review = (
        db.query(Review)
        .filter(Review.queue_item_id == queue_item.id)
        .order_by(Review.id.desc())
        .first()
    )

    if review:
        review.review_status = request.review_status
        review.note = request.note
        review.reviewed_by = "local_user"
    else:
        review = Review(
            queue_item_id=queue_item.id,
            frame_id=queue_item.frame_id,
            prediction_id=queue_item.prediction_id,
            review_status=request.review_status,
            note=request.note,
            reviewed_by="local_user",
        )
        db.add(review)

    queue_item.status = "completed"
    db.commit()
    db.refresh(review)

    return {
        "success": True,
        "queue_item_id": queue_item.id,
        "queue_status": queue_item.status,
        "review": review_to_dict(review),
    }


@app.post("/api/reset-review")
def reset_review(request: ReviewResetRequest, db: Session = Depends(get_db)):
    queue_item = db.query(QueueItem).filter(QueueItem.id == request.queue_item_id).first()

    if not queue_item:
        raise HTTPException(status_code=404, detail="Queue item not found")

    db.query(Review).filter(Review.queue_item_id == queue_item.id).delete()
    queue_item.status = get_reset_status(queue_item)
    db.commit()

    return {
        "success": True,
        "queue_item_id": queue_item.id,
        "queue_status": queue_item.status,
    }


@app.post("/api/save-annotations")
def save_annotations(request: SaveAnnotationsRequest, db: Session = Depends(get_db)):
    frame = db.query(Frame).filter(Frame.id == request.frame_id).first()

    if not frame:
        raise HTTPException(status_code=404, detail="Frame not found")

    queue_item = None

    if request.queue_item_id is not None:
        queue_item = db.query(QueueItem).filter(QueueItem.id == request.queue_item_id).first()

        if not queue_item:
            raise HTTPException(status_code=404, detail="Queue item not found")

    old_annotations_query = (
        db.query(Annotation)
        .filter(Annotation.frame_id == frame.id)
        .filter(Annotation.annotation_source == "manual_draw")
    )

    if queue_item:
        old_annotations_query = old_annotations_query.filter(Annotation.queue_item_id == queue_item.id)

    old_annotations = old_annotations_query.all()

    for annotation in old_annotations:
        db.query(AnnotationBox).filter(AnnotationBox.annotation_id == annotation.id).delete()
        db.delete(annotation)

    saved_boxes = []
    annotation = Annotation(
        frame_id=frame.id,
        queue_item_id=queue_item.id if queue_item else None,
        annotation_source="manual_draw",
        label_name="product",
        status="saved",
        created_by="local_user",
    )

    db.add(annotation)
    db.flush()

    for box_request in request.boxes:
        box_data = model_to_dict(box_request)
        points = box_data.get("points") or []

        if len(points) != 4:
            continue

        clean_points = [
            [round(float(point[0]), 2), round(float(point[1]), 2)]
            for point in points
        ]

        annotation_box = AnnotationBox(
            annotation_id=annotation.id,
            class_id=0,
            label_name="product",
            box_type="obb",
            points_json=clean_points,
            normalized_points_json=normalize_points(clean_points, frame.width, frame.height),
            confidence=box_data.get("confidence"),
            box_source=box_data.get("box_source") or "manual",
        )

        db.add(annotation_box)
        db.flush()

        saved_boxes.append(
            {
                "annotation_id": annotation.id,
                "annotation_box_id": annotation_box.id,
                "class_id": annotation_box.class_id,
                "label_name": annotation_box.label_name,
                "box_type": annotation_box.box_type,
                "points": annotation_box.points_json,
                "normalized_points": annotation_box.normalized_points_json,
                "confidence": annotation_box.confidence,
                "box_source": annotation_box.box_source,
            }
        )

    db.commit()

    return {
        "success": True,
        "frame_id": frame.id,
        "queue_item_id": queue_item.id if queue_item else None,
        "annotations_saved": True,
        "box_count": len(saved_boxes),
        "boxes": saved_boxes,
    }
