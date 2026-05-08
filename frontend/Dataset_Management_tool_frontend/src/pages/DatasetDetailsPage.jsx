import { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { FiDownload, FiImage, FiLayers, FiAlertCircle, FiCheckCircle } from 'react-icons/fi';
import { Chart as ChartJS, ArcElement, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { Bar, Pie } from 'react-chartjs-2';
import { datasetApi } from '../api/datasetApi';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorMessage from '../components/Common/ErrorMessage';
import './DatasetDetailsPage.css';

// Register Chart.js components
ChartJS.register(ArcElement, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const ISSUE_LABELS = {
    missing_label: 'Missing Label',
    invalid_bbox: 'Invalid BBox',
    blurry_image: 'Blurred',
    object_too_small: 'Too Small/Far',
};

const clamp01 = (value) => Math.min(1, Math.max(0, value));

const yoloToCorners = (x, y, w, h) => ({
    left: clamp01(x - (w / 2)),
    top: clamp01(y - (h / 2)),
    right: clamp01(x + (w / 2)),
    bottom: clamp01(y + (h / 2)),
});

const cornersToYolo = (left, top, right, bottom) => {
    const l = clamp01(Math.min(left, right));
    const t = clamp01(Math.min(top, bottom));
    const r = clamp01(Math.max(left, right));
    const b = clamp01(Math.max(top, bottom));
    return {
        x: (l + r) / 2,
        y: (t + b) / 2,
        w: r - l,
        h: b - t,
    };
};

function DatasetDetailsPage() {
    const { id } = useParams();
    const [dataset, setDataset] = useState(null);
    const [images, setImages] = useState([]);
    const [failedImageIds, setFailedImageIds] = useState(new Set());
    const [issuesByImageId, setIssuesByImageId] = useState({});
    const [selectedImageId, setSelectedImageId] = useState(null);
    const [editingLabels, setEditingLabels] = useState([]);
    const [activeLabelIndex, setActiveLabelIndex] = useState(null);
    const [drawMode, setDrawMode] = useState(false);
    const [drawClassId, setDrawClassId] = useState('0');
    const [dragStartPoint, setDragStartPoint] = useState(null);
    const [draftBox, setDraftBox] = useState(null);
    const [interaction, setInteraction] = useState(null);
    const [savingLabels, setSavingLabels] = useState(false);
    const [notice, setNotice] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('overview');
    const editorCanvasRef = useRef(null);

    useEffect(() => {
        fetchDatasetDetails();
    }, [id]);

    useEffect(() => {
        const onKeyDown = (event) => {
            if ((event.key === 'Delete' || event.key === 'Backspace') && activeLabelIndex !== null) {
                event.preventDefault();
                removeLabelRow(activeLabelIndex);
            }
        };

        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [activeLabelIndex]);

    const fetchDatasetDetails = async () => {
        try {
            setLoading(true);
            setError(null);
            setNotice('');

            const [datasetData, imagesData, issuesData] = await Promise.all([
                datasetApi.getDataset(id),
                datasetApi.getDatasetImages(id, { page: 1, limit: 12 }),
                datasetApi.getDatasetImageIssues(id, { page: 1, limit: 12 }),
            ]);

            setDataset(datasetData);
            setImages(imagesData.images);
            setFailedImageIds(new Set());
            setSelectedImageId(imagesData.images[0]?.id || null);
            setEditingLabels(mapLabelsForEditor(imagesData.images[0]?.labels || []));
            if (datasetData.class_distribution?.length > 0) {
                setDrawClassId(`${datasetData.class_distribution[0].class_id}`);
            }
            setActiveLabelIndex(null);
            setDragStartPoint(null);
            setDraftBox(null);
            setInteraction(null);
            setIssuesByImageId(indexIssuesByImageId(issuesData.flagged_images || []));
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to load dataset details');
        } finally {
            setLoading(false);
        }
    };

    const handleDownload = async () => {
        try {
            const response = await datasetApi.getDownloadUrl(id);
            window.open(response.download_url, '_blank');
        } catch (err) {
            setError('Failed to get download URL');
        }
    };

    const mapLabelsForEditor = (labels) => labels.map((label) => ({
        class_id: `${label.class_id ?? ''}`,
        x: `${label.bbox?.yolo?.[0] ?? ''}`,
        y: `${label.bbox?.yolo?.[1] ?? ''}`,
        w: `${label.bbox?.yolo?.[2] ?? ''}`,
        h: `${label.bbox?.yolo?.[3] ?? ''}`,
    }));

    const indexIssuesByImageId = (flaggedImages) => {
        const map = {};
        flaggedImages.forEach((item) => {
            map[item.id] = item;
        });
        return map;
    };

    const getImageSrc = (url) => {
        if (!url) return '';
        return url.startsWith('/api') ? url : `/api${url}`;
    };

    const handleImageError = (imageId) => {
        setFailedImageIds((prev) => new Set(prev).add(imageId));
    };

    const selectedImage = images.find((img) => img.id === selectedImageId) || null;
    const selectedImageIssues = selectedImage ? issuesByImageId[selectedImage.id] : null;
    const classIdOptions = (dataset?.class_distribution || []).map((item) => `${item.class_id}`);
    const editorBoxes = editingLabels
        .map((row, index) => {
            const valid = getValidYoloFromRow(row);
            if (!valid) return null;
            const corners = yoloToCorners(valid.x, valid.y, valid.w, valid.h);
            return { ...corners, classId: valid.classId, index };
        })
        .filter(Boolean);

    const selectImageForReview = (image) => {
        setSelectedImageId(image.id);
        setEditingLabels(mapLabelsForEditor(image.labels || []));
        setActiveLabelIndex(null);
        setDragStartPoint(null);
        setDraftBox(null);
        setInteraction(null);
        setNotice('');
        setError(null);
    };

    const addLabelRow = () => {
        setEditingLabels((prev) => {
            const next = [...prev, { class_id: '', x: '', y: '', w: '', h: '' }];
            setActiveLabelIndex(next.length - 1);
            return next;
        });
    };

    const removeLabelRow = (rowIndex) => {
        setEditingLabels((prev) => prev.filter((_, idx) => idx !== rowIndex));
        setActiveLabelIndex((prev) => {
            if (prev === null) return null;
            if (prev === rowIndex) return null;
            if (prev > rowIndex) return prev - 1;
            return prev;
        });
    };

    const updateLabelField = (rowIndex, key, value) => {
        setEditingLabels((prev) => prev.map((row, idx) => (
            idx === rowIndex ? { ...row, [key]: value } : row
        )));
    };

    const updateLabelFromCorners = (rowIndex, corners) => {
        const yolo = cornersToYolo(corners.left, corners.top, corners.right, corners.bottom);
        setEditingLabels((prev) => prev.map((row, idx) => (
            idx === rowIndex
                ? {
                    ...row,
                    x: yolo.x.toFixed(6),
                    y: yolo.y.toFixed(6),
                    w: yolo.w.toFixed(6),
                    h: yolo.h.toFixed(6),
                }
                : row
        )));
    };

    function getValidYoloFromRow(row) {
        const classId = `${row.class_id}`.trim();
        const values = [row.x, row.y, row.w, row.h].map((v) => Number.parseFloat(v));
        const allFinite = values.every((v) => Number.isFinite(v));
        if (!classId || !allFinite) return null;
        return {
            classId,
            x: clamp01(values[0]),
            y: clamp01(values[1]),
            w: clamp01(values[2]),
            h: clamp01(values[3]),
        };
    }

    const getCanvasPoint = (event) => {
        if (!editorCanvasRef.current) return null;
        const rect = editorCanvasRef.current.getBoundingClientRect();
        if (!rect.width || !rect.height) return null;
        const px = clamp01((event.clientX - rect.left) / rect.width);
        const py = clamp01((event.clientY - rect.top) / rect.height);
        return { x: px, y: py };
    };

    const handleCanvasMouseDown = (event) => {
        if (!drawMode || event.button !== 0) return;
        const point = getCanvasPoint(event);
        if (!point) return;
        setDragStartPoint(point);
        setDraftBox({ left: point.x, top: point.y, right: point.x, bottom: point.y });
    };

    const startMoveBox = (event, box) => {
        if (drawMode || event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();
        const point = getCanvasPoint(event);
        if (!point) return;
        setActiveLabelIndex(box.index);
        setInteraction({
            type: 'move',
            rowIndex: box.index,
            startPoint: point,
            startBox: box,
        });
    };

    const startResizeBox = (event, box, handle) => {
        if (drawMode || event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();
        const point = getCanvasPoint(event);
        if (!point) return;
        setActiveLabelIndex(box.index);
        setInteraction({
            type: 'resize',
            rowIndex: box.index,
            startPoint: point,
            startBox: box,
            handle,
        });
    };

    const handleCanvasMouseMove = (event) => {
        const point = getCanvasPoint(event);
        if (!point) return;

        if (drawMode && dragStartPoint) {
            setDraftBox({
                left: dragStartPoint.x,
                top: dragStartPoint.y,
                right: point.x,
                bottom: point.y,
            });
            return;
        }

        if (!interaction) return;

        if (interaction.type === 'move') {
            const { startPoint, startBox, rowIndex } = interaction;
            const dx = point.x - startPoint.x;
            const dy = point.y - startPoint.y;

            const width = startBox.right - startBox.left;
            const height = startBox.bottom - startBox.top;
            let left = startBox.left + dx;
            let top = startBox.top + dy;

            left = clamp01(left);
            top = clamp01(top);
            if (left + width > 1) left = 1 - width;
            if (top + height > 1) top = 1 - height;

            const nextCorners = {
                left,
                top,
                right: left + width,
                bottom: top + height,
            };
            updateLabelFromCorners(rowIndex, nextCorners);
            return;
        }

        if (interaction.type === 'resize') {
            const { startBox, rowIndex, handle } = interaction;
            let left = startBox.left;
            let top = startBox.top;
            let right = startBox.right;
            let bottom = startBox.bottom;

            if (handle.includes('n')) top = point.y;
            if (handle.includes('s')) bottom = point.y;
            if (handle.includes('w')) left = point.x;
            if (handle.includes('e')) right = point.x;

            const yolo = cornersToYolo(left, top, right, bottom);
            if (yolo.w < 0.01 || yolo.h < 0.01) return;

            updateLabelFromCorners(rowIndex, yoloToCorners(yolo.x, yolo.y, yolo.w, yolo.h));
        }
    };

    const handleCanvasMouseUp = (event) => {
        if (drawMode && dragStartPoint && draftBox) {
            const point = getCanvasPoint(event);
            const completed = point
                ? { left: dragStartPoint.x, top: dragStartPoint.y, right: point.x, bottom: point.y }
                : draftBox;
            const yolo = cornersToYolo(completed.left, completed.top, completed.right, completed.bottom);

            setDragStartPoint(null);
            setDraftBox(null);

            if (yolo.w < 0.01 || yolo.h < 0.01) return;

            setEditingLabels((prev) => {
                const next = [
                    ...prev,
                    {
                        class_id: `${drawClassId || '0'}`,
                        x: yolo.x.toFixed(6),
                        y: yolo.y.toFixed(6),
                        w: yolo.w.toFixed(6),
                        h: yolo.h.toFixed(6),
                    },
                ];
                setActiveLabelIndex(next.length - 1);
                return next;
            });
            return;
        }

        if (interaction) {
            setInteraction(null);
        }
    };

    const parseLabelRows = () => {
        const parsed = [];

        for (const row of editingLabels) {
            const hasAnyValue = Object.values(row).some((v) => `${v}`.trim() !== '');
            if (!hasAnyValue) continue;

            const classId = `${row.class_id}`.trim();
            const values = [row.x, row.y, row.w, row.h].map((v) => Number.parseFloat(v));
            const allFinite = values.every((v) => Number.isFinite(v));
            if (!classId || !allFinite) {
                throw new Error('Each label must have class_id and numeric x, y, w, h values.');
            }

            parsed.push({
                class_id: classId,
                bbox: { yolo: values },
            });
        }

        return parsed;
    };

    const refreshIssues = async () => {
        const issuesData = await datasetApi.getDatasetImageIssues(id, { page: 1, limit: 12 });
        setIssuesByImageId(indexIssuesByImageId(issuesData.flagged_images || []));
    };

    const handleSaveLabels = async () => {
        if (!selectedImage) return;

        try {
            setSavingLabels(true);
            setError(null);
            setNotice('');

            const payload = parseLabelRows();
            await datasetApi.updateImageLabels(id, selectedImage.id, payload);

            setImages((prev) => prev.map((img) => (
                img.id === selectedImage.id
                    ? { ...img, has_label: payload.length > 0, labels: payload }
                    : img
            )));
            setNotice('Labels updated successfully.');
            await refreshIssues();
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to update labels');
        } finally {
            setSavingLabels(false);
        }
    };

    if (loading) {
        return <LoadingSpinner message="Loading dataset details..." />;
    }

    if (error && !dataset) {
        return (
            <div className="dataset-details-page">
                <ErrorMessage message={error} />
            </div>
        );
    }

    // Prepare chart data
    const classDistributionData = {
        labels: dataset.class_distribution.map(cd => `Class ${cd.class_id}`),
        datasets: [{
            label: 'Object Count',
            data: dataset.class_distribution.map(cd => cd.object_count),
            backgroundColor: [
                'rgba(102, 126, 234, 0.8)',
                'rgba(118, 75, 162, 0.8)',
                'rgba(240, 147, 251, 0.8)',
                'rgba(245, 87, 108, 0.8)',
                'rgba(79, 172, 254, 0.8)',
                'rgba(0, 242, 254, 0.8)',
                'rgba(67, 233, 123, 0.8)',
                'rgba(56, 249, 215, 0.8)',
            ],
            borderColor: 'rgba(255, 255, 255, 0.2)',
            borderWidth: 2,
        }]
    };

    const validation = dataset.validation || {};
    const issueBreakdownData = {
        labels: ['Missing Labels', 'Orphan Labels', 'Empty Labels', 'Corrupted Images'],
        datasets: [{
            label: 'Count',
            data: [
                validation.missing_labels || 0,
                validation.orphan_labels || 0,
                validation.empty_labels || 0,
                validation.corrupted_images || 0,
            ],
            backgroundColor: [
                'rgba(245, 158, 11, 0.75)',
                'rgba(239, 68, 68, 0.75)',
                'rgba(251, 113, 133, 0.75)',
                'rgba(107, 114, 128, 0.75)',
            ],
            borderColor: 'rgba(255, 255, 255, 0.35)',
            borderWidth: 1,
        }]
    };

    const totalImages = dataset.total_images || 0;
    const corruptedImages = validation.corrupted_images || 0;
    const missingLabels = validation.missing_labels || 0;
    const cleanLabeled = Math.max(0, totalImages - corruptedImages - missingLabels);
    const coverageData = {
        labels: ['Clean Labeled', 'Missing Labels', 'Corrupted Images'],
        datasets: [{
            data: [cleanLabeled, missingLabels, corruptedImages],
            backgroundColor: [
                'rgba(16, 185, 129, 0.8)',
                'rgba(245, 158, 11, 0.8)',
                'rgba(107, 114, 128, 0.8)',
            ],
            borderColor: 'rgba(255, 255, 255, 0.35)',
            borderWidth: 1,
        }]
    };

    const compositionData = {
        labels: ['Total Images', 'Total Labels', 'Total Objects'],
        datasets: [{
            label: 'Dataset Composition',
            data: [dataset.total_images, dataset.total_labels, dataset.total_objects],
            backgroundColor: [
                'rgba(59, 130, 246, 0.75)',
                'rgba(99, 102, 241, 0.75)',
                'rgba(168, 85, 247, 0.75)',
            ],
            borderColor: 'rgba(255, 255, 255, 0.35)',
            borderWidth: 1,
        }]
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: '#22365f',
                }
            }
        },
        scales: {
            y: {
                ticks: { color: '#39517c' },
                grid: { color: 'rgba(34, 54, 95, 0.1)' }
            },
            x: {
                ticks: { color: '#39517c' },
                grid: { color: 'rgba(34, 54, 95, 0.1)' }
            }
        }
    };

    return (
        <div className="dataset-details-page fade-in">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Dataset Details</h1>
                    <p className="page-subtitle">ID: {id}</p>
                </div>
                <div className="header-actions">
                    <Link to="/augment" state={{ datasetId: id }} className="btn btn-secondary">
                        Augment
                    </Link>
                    <button onClick={handleDownload} className="btn btn-primary">
                        <FiDownload /> Download
                    </button>
                </div>
            </div>

            {error && <ErrorMessage message={error} onClose={() => setError(null)} />}
            {notice && <div className="notice-message">{notice}</div>}

            <div className="tabs">
                <button
                    className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
                    onClick={() => setActiveTab('overview')}
                >
                    Overview
                </button>
                <button
                    className={`tab ${activeTab === 'images' ? 'active' : ''}`}
                    onClick={() => setActiveTab('images')}
                >
                    Images
                </button>
                <button
                    className={`tab ${activeTab === 'validation' ? 'active' : ''}`}
                    onClick={() => setActiveTab('validation')}
                >
                    Validation
                </button>
            </div>

            {activeTab === 'overview' && (
                <div className="tab-content">
                    <div className="stats-grid grid grid-4">
                        <div className="stat-card card">
                            <FiImage className="stat-icon" />
                            <div>
                                <div className="stat-value">{dataset.total_images}</div>
                                <div className="stat-label">Total Images</div>
                            </div>
                        </div>
                        <div className="stat-card card">
                            <FiLayers className="stat-icon" />
                            <div>
                                <div className="stat-value">{dataset.total_classes}</div>
                                <div className="stat-label">Total Classes</div>
                            </div>
                        </div>
                        <div className="stat-card card">
                            <div className="stat-icon">🎯</div>
                            <div>
                                <div className="stat-value">{dataset.total_objects}</div>
                                <div className="stat-label">Total Objects</div>
                            </div>
                        </div>
                        <div className="stat-card card">
                            <div className="stat-icon">📊</div>
                            <div>
                                <div className="stat-value">{dataset.avg_objects_per_image.toFixed(2)}</div>
                                <div className="stat-label">Avg Objects/Image</div>
                            </div>
                        </div>
                    </div>

                    <div className="charts-grid">
                        <div className="chart-card card">
                            <h3>Class Distribution (Bar Chart)</h3>
                            <div className="chart-container">
                                <Bar data={classDistributionData} options={chartOptions} />
                            </div>
                        </div>
                        <div className="chart-card card">
                            <h3>Class Distribution (Pie Chart)</h3>
                            <div className="chart-container">
                                <Pie data={classDistributionData} options={{ ...chartOptions, scales: undefined }} />
                            </div>
                        </div>
                        <div className="chart-card card">
                            <h3>Validation Issues (Bar Chart)</h3>
                            <div className="chart-container">
                                <Bar data={issueBreakdownData} options={chartOptions} />
                            </div>
                        </div>
                        <div className="chart-card card">
                            <h3>Label Coverage (Pie Chart)</h3>
                            <div className="chart-container">
                                <Pie data={coverageData} options={{ ...chartOptions, scales: undefined }} />
                            </div>
                        </div>
                        <div className="chart-card card chart-card-wide">
                            <h3>Dataset Composition</h3>
                            <div className="chart-container">
                                <Bar data={compositionData} options={chartOptions} />
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {activeTab === 'images' && (
                <div className="tab-content">
                    <div className="images-grid grid grid-4">
                        {images.map((image) => (
                            <button
                                key={image.id}
                                type="button"
                                className={`image-card card image-select-btn ${selectedImageId === image.id ? 'selected' : ''}`}
                                onClick={() => selectImageForReview(image)}
                            >
                                {image.url && !failedImageIds.has(image.id) ? (
                                    <img
                                        className="image-preview"
                                        src={getImageSrc(image.url)}
                                        alt={image.file_name}
                                        loading="lazy"
                                        onError={() => handleImageError(image.id)}
                                    />
                                ) : (
                                    <div className="image-placeholder">
                                        <FiImage />
                                        <p className="image-name">{image.file_name}</p>
                                    </div>
                                )}
                                <div className="image-info">
                                    {image.has_label ? (
                                        <span className="badge badge-success">
                                            <FiCheckCircle /> Labeled
                                        </span>
                                    ) : (
                                        <span className="badge badge-warning">
                                            <FiAlertCircle /> No Label
                                        </span>
                                    )}
                                    {issuesByImageId[image.id]?.issues?.length > 0 && (
                                        <div className="issue-badges">
                                            {issuesByImageId[image.id].issues.map((issueCode) => (
                                                <span key={issueCode} className="badge badge-danger">
                                                    {ISSUE_LABELS[issueCode] || issueCode}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </button>
                        ))}
                    </div>

                    {selectedImage && (
                        <div className="label-editor card">
                            <div className="label-editor-header">
                                <div>
                                    <h3>Review & Correct Labels</h3>
                                    <p>{selectedImage.file_name}</p>
                                </div>
                                <button
                                    type="button"
                                    className="btn btn-primary"
                                    onClick={handleSaveLabels}
                                    disabled={savingLabels}
                                >
                                    {savingLabels ? 'Saving...' : 'Save Labels'}
                                </button>
                            </div>

                            <div className="label-editor-grid">
                                <div className="editor-image-wrap">
                                    <div className="editor-toolbar">
                                        <label className="tool-label" htmlFor="draw-class-select">Class</label>
                                        <select
                                            id="draw-class-select"
                                            value={drawClassId}
                                            onChange={(e) => setDrawClassId(e.target.value)}
                                        >
                                            {classIdOptions.length > 0 ? (
                                                classIdOptions.map((classId) => (
                                                    <option key={classId} value={classId}>
                                                        {classId}
                                                    </option>
                                                ))
                                            ) : (
                                                <option value="0">0</option>
                                            )}
                                        </select>
                                        <button
                                            type="button"
                                            className={`btn ${drawMode ? 'btn-primary' : 'btn-secondary'}`}
                                            onClick={() => {
                                                setDrawMode((prev) => !prev);
                                                setDragStartPoint(null);
                                                setDraftBox(null);
                                                setInteraction(null);
                                            }}
                                        >
                                            {drawMode ? 'Drawing On' : 'Draw Box'}
                                        </button>
                                    </div>

                                    <div
                                        ref={editorCanvasRef}
                                        className={`editor-canvas-wrap ${drawMode ? 'draw-mode' : ''}`}
                                        onMouseDown={handleCanvasMouseDown}
                                        onMouseMove={handleCanvasMouseMove}
                                        onMouseUp={handleCanvasMouseUp}
                                        onMouseLeave={handleCanvasMouseUp}
                                    >
                                        {selectedImage.url && !failedImageIds.has(selectedImage.id) ? (
                                            <img
                                                className="editor-image"
                                                src={getImageSrc(selectedImage.url)}
                                                alt={selectedImage.file_name}
                                                onError={() => handleImageError(selectedImage.id)}
                                            />
                                        ) : (
                                            <div className="image-placeholder editor-placeholder">
                                                <FiImage />
                                                <p className="image-name">{selectedImage.file_name}</p>
                                            </div>
                                        )}

                                        <div className="editor-overlay">
                                            {editorBoxes.map((box) => (
                                                <button
                                                    key={`editor-box-${box.index}`}
                                                    type="button"
                                                    className={`editor-box ${activeLabelIndex === box.index ? 'active' : ''}`}
                                                    style={{
                                                        left: `${box.left * 100}%`,
                                                        top: `${box.top * 100}%`,
                                                        width: `${(box.right - box.left) * 100}%`,
                                                        height: `${(box.bottom - box.top) * 100}%`,
                                                    }}
                                                    onMouseDown={(e) => startMoveBox(e, box)}
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        setActiveLabelIndex(box.index);
                                                    }}
                                                >
                                                    <span className="editor-box-label">{box.classId}</span>
                                                    <span
                                                        className="editor-handle nw"
                                                        onMouseDown={(e) => startResizeBox(e, box, 'nw')}
                                                    />
                                                    <span
                                                        className="editor-handle ne"
                                                        onMouseDown={(e) => startResizeBox(e, box, 'ne')}
                                                    />
                                                    <span
                                                        className="editor-handle sw"
                                                        onMouseDown={(e) => startResizeBox(e, box, 'sw')}
                                                    />
                                                    <span
                                                        className="editor-handle se"
                                                        onMouseDown={(e) => startResizeBox(e, box, 'se')}
                                                    />
                                                </button>
                                            ))}

                                            {draftBox && (
                                                <div
                                                    className="editor-box draft"
                                                    style={{
                                                        left: `${Math.min(draftBox.left, draftBox.right) * 100}%`,
                                                        top: `${Math.min(draftBox.top, draftBox.bottom) * 100}%`,
                                                        width: `${Math.abs(draftBox.right - draftBox.left) * 100}%`,
                                                        height: `${Math.abs(draftBox.bottom - draftBox.top) * 100}%`,
                                                    }}
                                                />
                                            )}
                                        </div>
                                    </div>

                                    <p className="draw-help">
                                        {drawMode
                                            ? 'Drag on image to create a new bounding box.'
                                            : 'Toggle "Draw Box" to add labels visually.'}
                                    </p>
                                    {selectedImageIssues?.issues?.length > 0 && (
                                        <div className="editor-issues">
                                            {selectedImageIssues.issues.map((issueCode) => (
                                                <span key={issueCode} className="badge badge-danger">
                                                    {ISSUE_LABELS[issueCode] || issueCode}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                    {selectedImageIssues?.blur_score !== undefined && selectedImageIssues?.blur_score !== null && (
                                        <p className="blur-score">
                                            Blur score: {selectedImageIssues.blur_score.toFixed(2)}
                                        </p>
                                    )}
                                </div>

                                <div className="editor-form">
                                    <div className="editor-actions">
                                        <button type="button" className="btn btn-secondary" onClick={addLabelRow}>
                                            Add Label
                                        </button>
                                    </div>

                                    <div className="label-table">
                                        <div className="label-row label-row-head">
                                            <span>Class</span>
                                            <span>X</span>
                                            <span>Y</span>
                                            <span>W</span>
                                            <span>H</span>
                                            <span>Action</span>
                                        </div>
                                        {editingLabels.map((row, rowIndex) => (
                                            <div
                                                key={`label-row-${rowIndex}`}
                                                className={`label-row ${activeLabelIndex === rowIndex ? 'active' : ''}`}
                                            >
                                                <input
                                                    value={row.class_id}
                                                    onChange={(e) => updateLabelField(rowIndex, 'class_id', e.target.value)}
                                                    onFocus={() => setActiveLabelIndex(rowIndex)}
                                                    placeholder="0"
                                                />
                                                <input
                                                    value={row.x}
                                                    onChange={(e) => updateLabelField(rowIndex, 'x', e.target.value)}
                                                    onFocus={() => setActiveLabelIndex(rowIndex)}
                                                    placeholder="0.50"
                                                />
                                                <input
                                                    value={row.y}
                                                    onChange={(e) => updateLabelField(rowIndex, 'y', e.target.value)}
                                                    onFocus={() => setActiveLabelIndex(rowIndex)}
                                                    placeholder="0.50"
                                                />
                                                <input
                                                    value={row.w}
                                                    onChange={(e) => updateLabelField(rowIndex, 'w', e.target.value)}
                                                    onFocus={() => setActiveLabelIndex(rowIndex)}
                                                    placeholder="0.20"
                                                />
                                                <input
                                                    value={row.h}
                                                    onChange={(e) => updateLabelField(rowIndex, 'h', e.target.value)}
                                                    onFocus={() => setActiveLabelIndex(rowIndex)}
                                                    placeholder="0.20"
                                                />
                                                <button
                                                    type="button"
                                                    className="btn btn-danger small-btn"
                                                    onClick={() => removeLabelRow(rowIndex)}
                                                >
                                                    Remove
                                                </button>
                                            </div>
                                        ))}
                                        {editingLabels.length === 0 && (
                                            <p className="empty-label-note">No labels. Click "Add Label" to create one.</p>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'validation' && dataset.validation && (
                <div className="tab-content">
                    <div className="validation-grid grid grid-2">
                        <div className="validation-card card">
                            <h3>Validation Summary</h3>
                            <div className="validation-stats">
                                <div className="validation-stat">
                                    <span className="label">Total Images:</span>
                                    <span className="value">{dataset.validation.total_images}</span>
                                </div>
                                <div className="validation-stat">
                                    <span className="label">Total Labels:</span>
                                    <span className="value">{dataset.validation.total_labels}</span>
                                </div>
                                <div className="validation-stat">
                                    <span className="label">Missing Labels:</span>
                                    <span className="value error">{dataset.validation.missing_labels}</span>
                                </div>
                                <div className="validation-stat">
                                    <span className="label">Orphan Labels:</span>
                                    <span className="value error">{dataset.validation.orphan_labels}</span>
                                </div>
                                <div className="validation-stat">
                                    <span className="label">Empty Labels:</span>
                                    <span className="value error">{dataset.validation.empty_labels}</span>
                                </div>
                                <div className="validation-stat">
                                    <span className="label">Corrupted Images:</span>
                                    <span className="value error">{dataset.validation.corrupted_images}</span>
                                </div>
                            </div>
                        </div>

                        <div className="validation-card card">
                            <h3>Dataset Info</h3>
                            <div className="validation-stats">
                                <div className="validation-stat">
                                    <span className="label">Format:</span>
                                    <span className="value">{dataset.format_type.toUpperCase()}</span>
                                </div>
                                <div className="validation-stat">
                                    <span className="label">Created:</span>
                                    <span className="value">{new Date(dataset.created_at).toLocaleString()}</span>
                                </div>
                                <div className="validation-stat">
                                    <span className="label">Missing Label Count:</span>
                                    <span className="value">{dataset.missing_label_count}</span>
                                </div>
                                <div className="validation-stat">
                                    <span className="label">Corrupted Image Count:</span>
                                    <span className="value">{dataset.corrupted_image_count}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default DatasetDetailsPage;
