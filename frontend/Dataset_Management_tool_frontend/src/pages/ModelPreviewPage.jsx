import { useEffect, useMemo, useRef, useState } from 'react';
import { FiCopy, FiUpload, FiCamera, FiLink, FiPlay } from 'react-icons/fi';
import { datasetApi } from '../api/datasetApi';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorMessage from '../components/Common/ErrorMessage';
import './ModelPreviewPage.css';

const clamp = (v, min, max) => Math.min(max, Math.max(min, v));

const yoloToCorners = (p) => ({
    l: clamp(p.x - (p.width / 2), 0, 1),
    t: clamp(p.y - (p.height / 2), 0, 1),
    r: clamp(p.x + (p.width / 2), 0, 1),
    b: clamp(p.y + (p.height / 2), 0, 1),
});

const iou = (a, b) => {
    const ac = yoloToCorners(a);
    const bc = yoloToCorners(b);
    const x1 = Math.max(ac.l, bc.l);
    const y1 = Math.max(ac.t, bc.t);
    const x2 = Math.min(ac.r, bc.r);
    const y2 = Math.min(ac.b, bc.b);
    const interW = Math.max(0, x2 - x1);
    const interH = Math.max(0, y2 - y1);
    const inter = interW * interH;
    if (!inter) return 0;
    const areaA = (ac.r - ac.l) * (ac.b - ac.t);
    const areaB = (bc.r - bc.l) * (bc.b - bc.t);
    return inter / Math.max(1e-9, (areaA + areaB - inter));
};

const nms = (predictions, overlapThreshold01) => {
    const sorted = [...predictions].sort((a, b) => b.confidence - a.confidence);
    const kept = [];
    for (const candidate of sorted) {
        const shouldDrop = kept.some((k) => iou(k, candidate) > overlapThreshold01);
        if (!shouldDrop) kept.push(candidate);
    }
    return kept;
};

function ModelPreviewPage() {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [notice, setNotice] = useState('');

    const [datasets, setDatasets] = useState([]);
    const [selectedDatasetId, setSelectedDatasetId] = useState('');
    const [datasetImages, setDatasetImages] = useState([]);
    const [selectedImageId, setSelectedImageId] = useState('');
    const [classIds, setClassIds] = useState([]);

    const [previewSrc, setPreviewSrc] = useState('');
    const [previewImageSize, setPreviewImageSize] = useState({ width: 0, height: 0 });
    const previewImageRef = useRef(null);
    const [rawPredictions, setRawPredictions] = useState([]);
    const [urlInput, setUrlInput] = useState('');
    const [confidenceThreshold, setConfidenceThreshold] = useState(50);
    const [overlapThreshold, setOverlapThreshold] = useState(50);
    const [opacityThreshold, setOpacityThreshold] = useState(75);
    const [labelMode, setLabelMode] = useState('draw_confidence');
    const [labelFormatMode, setLabelFormatMode] = useState('yolo');
    const [inferenceInfo, setInferenceInfo] = useState(null);

    const [webcamActive, setWebcamActive] = useState(false);
    const videoRef = useRef(null);
    const webcamStreamRef = useRef(null);
    const fileInputRef = useRef(null);
    const tempObjectUrlRef = useRef(null);
    const didInitialLoadRef = useRef(false);
    const skipDatasetEffectRef = useRef(false);

    const getImageSrc = (url) => (url?.startsWith('/api') ? url : `/api${url}`);

    const normalizeLabelCoords = (raw, forcedFormat, imageSize) => {
        const a = Number(raw?.[0]);
        const b = Number(raw?.[1]);
        const c = Number(raw?.[2]);
        const d = Number(raw?.[3]);
        if (![a, b, c, d].every(Number.isFinite)) return null;

        const isNormalized = [a, b, c, d].every((value) => value >= 0 && value <= 1);
        const hasImageSize = imageSize?.width > 0 && imageSize?.height > 0;
        const isAbsolute = hasImageSize && [a, b, c, d].some((value) => value > 1);

        const normalizeAbsoluteYolo = (x, y, w, h) => ({
            x: clamp(x / imageSize.width, 0, 1),
            y: clamp(y / imageSize.height, 0, 1),
            width: clamp(w / imageSize.width, 0, 1),
            height: clamp(h / imageSize.height, 0, 1),
        });

        const normalizedXyxy = () => {
            const x1 = clamp(Math.min(a, c), 0, 1);
            const y1 = clamp(Math.min(b, d), 0, 1);
            const x2 = clamp(Math.max(a, c), 0, 1);
            const y2 = clamp(Math.max(b, d), 0, 1);
            const width = Math.max(0, x2 - x1);
            const height = Math.max(0, y2 - y1);
            if (width <= 0 || height <= 0) return null;
            return { x: (x1 + x2) / 2, y: (y1 + y2) / 2, width, height };
        };

        const absoluteXyxy = () => {
            const x1 = clamp(Math.min(a, c) / imageSize.width, 0, 1);
            const y1 = clamp(Math.min(b, d) / imageSize.height, 0, 1);
            const x2 = clamp(Math.max(a, c) / imageSize.width, 0, 1);
            const y2 = clamp(Math.max(b, d) / imageSize.height, 0, 1);
            const width = Math.max(0, x2 - x1);
            const height = Math.max(0, y2 - y1);
            if (width <= 0 || height <= 0) return null;
            return { x: (x1 + x2) / 2, y: (y1 + y2) / 2, width, height };
        };

        const absoluteYolo = () => {
            const result = normalizeAbsoluteYolo(a, b, c, d);
            if (result.width <= 0 || result.height <= 0) return null;
            return result;
        };

        const normalizedYolo = () => {
            const width = clamp(c, 0, 1);
            const height = clamp(d, 0, 1);
            if (width <= 0 || height <= 0) return null;
            return { x: clamp(a, 0, 1), y: clamp(b, 0, 1), width, height };
        };

        const detectFormat = () => {
            if (forcedFormat !== 'auto') return forcedFormat;

            if (isNormalized) {
                const yoloCandidate = normalizedYolo();
                const xyxyCandidate = normalizedXyxy();
                if (yoloCandidate && !xyxyCandidate) return 'yolo';
                if (!yoloCandidate && xyxyCandidate) return 'xyxy';
                return 'yolo';
            }

            if (isAbsolute && hasImageSize) {
                const yoloCandidate = absoluteYolo();
                const xyxyCandidate = absoluteXyxy();
                if (yoloCandidate && !xyxyCandidate) return 'yolo';
                if (!yoloCandidate && xyxyCandidate) return 'xyxy';
                if (yoloCandidate && xyxyCandidate) {
                    const yoloArea = yoloCandidate.width * yoloCandidate.height;
                    const xyxyArea = xyxyCandidate.width * xyxyCandidate.height;
                    if (yoloArea === 0) return 'xyxy';
                    if (xyxyArea === 0) return 'yolo';
                    if (yoloArea * 4 < xyxyArea) return 'yolo';
                    if (xyxyArea * 4 < yoloArea) return 'xyxy';
                    return 'yolo';
                }
                return yoloCandidate ? 'yolo' : 'xyxy';
            }

            return 'yolo';
        };

        const actualFormat = detectFormat();

        if (actualFormat === 'xyxy') {
            const result = isNormalized ? normalizedXyxy() : (isAbsolute && hasImageSize ? absoluteXyxy() : null);
            if (result) return result;
            if (forcedFormat === 'auto') {
                return isNormalized ? normalizedYolo() : (isAbsolute && hasImageSize ? absoluteYolo() : null);
            }
            return null;
        }

        if (actualFormat === 'yolo') {
            const result = isNormalized ? normalizedYolo() : (isAbsolute && hasImageSize ? absoluteYolo() : null);
            if (result) return result;
            if (forcedFormat === 'auto') {
                return isNormalized ? normalizedXyxy() : (isAbsolute && hasImageSize ? absoluteXyxy() : null);
            }
            return null;
        }

        return null;
    };

    const toPredictions = (img) => {
        const labels = img.labels || [];
        const resolvedFormat = labelFormatMode;
        return labels
            .map((lbl, index) => {
                const yolo = lbl.bbox?.yolo || [];
                const classId = `${lbl.class_id ?? '0'}`;
                const norm = normalizeLabelCoords(yolo, resolvedFormat, previewImageSize);
                if (!norm) return null;
                return {
                    x: norm.x,
                    y: norm.y,
                    width: norm.width,
                    height: norm.height,
                    class: `class_${classId}`,
                    class_id: classId,
                    confidence: 1.0,
                    detection_id: `${img.id}-${index}`,
                };
            })
            .filter(Boolean);
    };

    // run inference against backend using the most recent model for the
    // currently selected dataset.  returns [] on failure.
    // returns null if inference could not be run (model missing / error)
    // returns [] if inference ran successfully but no detections were found
    const runInference = async ({ imageUrl, file }) => {
        if (!selectedDatasetId) return null;
        try {
            const res = await datasetApi.predict({
                datasetId: selectedDatasetId,
                imageUrl,
                file,
            });
            if (res.inference_time_ms) {
                setInferenceInfo(`Inference ${res.inference_time_ms.toFixed(1)} ms`);
            } else {
                setInferenceInfo(null);
            }
            return res.predictions || [];
        } catch (err) {
            console.error('inference error', err);
            setNotice('Inference failed, showing ground truth labels');
            setInferenceInfo(null);
            return null;
        }
    };

    const loadDatasets = async () => {
        const list = await datasetApi.listDatasets({ page: 1, limit: 100, sort_by: 'created_at', order: 'desc' });
        setDatasets(list.datasets || []);
        const firstId = list.datasets?.[0]?.id || '';
        setSelectedDatasetId(firstId);
        return firstId;
    };

    const loadDatasetImages = async (datasetId) => {
        if (!datasetId) {
            setDatasetImages([]);
            setSelectedImageId('');
            return;
        }
        const [imagesRes, datasetRes] = await Promise.all([
            datasetApi.getDatasetImages(datasetId, { page: 1, limit: 20 }),
            datasetApi.getDataset(datasetId),
        ]);
        const imgs = imagesRes.images || [];
        setDatasetImages(imgs);
        setClassIds((datasetRes.class_distribution || []).map((c) => `${c.class_id}`));
        if (imgs[0]) {
            await selectSampleImage(imgs[0]);
        } else {
            setSelectedImageId('');
            setPreviewSrc('');
            setRawPredictions([]);
        }
    };

    const selectSampleImage = async (img) => {
        setSelectedImageId(img.id);
        const src = getImageSrc(img.url);
        setPreviewImageSize({ width: 0, height: 0 });
        setPreviewSrc(src);
        setNotice('');
        const preds = await runInference({ imageUrl: src });
        if (preds === null) {
            // inference failed or no model, show ground truth
            setRawPredictions(toPredictions(img));
        } else if (preds.length) {
            setRawPredictions(preds);
        } else {
            // no detections were produced
            setRawPredictions([]);
        }
    };

    useEffect(() => {
        if (didInitialLoadRef.current) return;
        didInitialLoadRef.current = true;

        const run = async () => {
            try {
                setLoading(true);
                setError('');
                const firstId = await loadDatasets();
                skipDatasetEffectRef.current = true;
                await loadDatasetImages(firstId);
            } catch (e) {
                setError(e.response?.data?.detail || 'Failed to load model preview data');
            } finally {
                setLoading(false);
            }
        };
        run();
    }, []);

    useEffect(() => {
        if (skipDatasetEffectRef.current) {
            skipDatasetEffectRef.current = false;
            return;
        }
        if (!selectedDatasetId) return;
        loadDatasetImages(selectedDatasetId).catch((e) => {
            setError(e.response?.data?.detail || 'Failed to load dataset images');
        });
    }, [selectedDatasetId]);

    useEffect(() => {
        // whenever the selected image or label formatting changes we
        // attempt to re-run inference (if a model exists); fall back to
        // ground truth in case of failure.
        if (!selectedImageId) return;
        const selected = datasetImages.find((img) => img.id === selectedImageId);
        if (!selected) return;
        const update = async () => {
            const src = getImageSrc(selected.url);
            setPreviewSrc(src);
            setNotice('Running inference...');
            const preds = await runInference({ imageUrl: src });
            if (preds === null) {
                setRawPredictions(toPredictions(selected));
            } else if (preds.length) {
                setRawPredictions(preds);
            } else {
                setRawPredictions([]);
            }
            setNotice('');
        };
        update();
    }, [labelFormatMode, selectedImageId, datasetImages, previewImageSize]);

    useEffect(() => () => {
        if (tempObjectUrlRef.current) URL.revokeObjectURL(tempObjectUrlRef.current);
        if (webcamStreamRef.current) {
            webcamStreamRef.current.getTracks().forEach((t) => t.stop());
            webcamStreamRef.current = null;
        }
    }, []);

    const filteredPredictions = useMemo(() => {
        const conf01 = confidenceThreshold / 100;
        const overlap01 = overlapThreshold / 100;
        const confFiltered = rawPredictions.filter((p) => p.confidence >= conf01);
        return nms(confFiltered, overlap01);
    }, [rawPredictions, confidenceThreshold, overlapThreshold]);

    const predictionsJson = useMemo(() => JSON.stringify({ predictions: filteredPredictions }, null, 2), [filteredPredictions]);

    const handleCopyJson = async () => {
        try {
            await navigator.clipboard.writeText(predictionsJson);
            setNotice('Predictions JSON copied');
        } catch {
            setNotice('Copy failed');
        }
    };

    const handleUploadFile = async (event) => {
        const file = event.target.files?.[0];
        if (!file) return;
        if (tempObjectUrlRef.current) URL.revokeObjectURL(tempObjectUrlRef.current);
        const objectUrl = URL.createObjectURL(file);
        tempObjectUrlRef.current = objectUrl;
        setPreviewImageSize({ width: 0, height: 0 });
        setPreviewSrc(objectUrl);
        setNotice('Running inference...');
        setRawPredictions([]);
        if (selectedDatasetId) {
            const preds = await runInference({ file });
            if (preds === null) {
                setNotice('Inference failed');
                setRawPredictions([]);
            } else {
                setRawPredictions(preds);
                setNotice('');
            }
        }
    };

    const handleApplyUrl = async () => {
        if (!urlInput.trim()) return;
        setPreviewImageSize({ width: 0, height: 0 });
        setPreviewSrc(urlInput.trim());
        setNotice('Running inference...');
        setRawPredictions([]);
        if (selectedDatasetId) {
            const preds = await runInference({ imageUrl: urlInput.trim() });
            if (preds === null) {
                setNotice('Inference failed');
            } else {
                setRawPredictions(preds);
                setNotice('');
            }
        }
    };

    const startWebcam = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            webcamStreamRef.current = stream;
            setWebcamActive(true);
            requestAnimationFrame(() => {
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.play().catch(() => {});
                }
            });
            setNotice('');
        } catch {
            setError('Unable to access webcam');
        }
    };

    const captureWebcam = () => {
        if (!videoRef.current) return;
        const video = videoRef.current;
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 1280;
        canvas.height = video.videoHeight || 720;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        setPreviewSrc(canvas.toDataURL('image/png'));
        setRawPredictions([]);
        setNotice('Webcam frame captured. Running inference using trained model.');
    };

    const stopWebcam = () => {
        if (webcamStreamRef.current) {
            webcamStreamRef.current.getTracks().forEach((t) => t.stop());
            webcamStreamRef.current = null;
        }
        setWebcamActive(false);
    };

    if (loading) return <LoadingSpinner message="Loading model preview..." />;

    return (
        <div className="model-preview-page fade-in">
            <div className="model-preview-header">
                <h1>Preview Model</h1>
                <p>Roboflow-style interactive preview that automatically runs your latest trained model. Select a dataset sample or upload an image to see detection boxes, set confidence/overlap thresholds, and view JSON output.</p>
            </div>

            {error && <ErrorMessage message={error} onClose={() => setError('')} />}
            {notice && <div className="model-preview-notice">{notice}</div>}
            {inferenceInfo && <div className="model-preview-info">{inferenceInfo}</div>}

            <div className="model-preview-layout">
                <aside className="model-panel model-panel-left">
                    <div className="panel-block">
                        <label>Dataset</label>
                        <select value={selectedDatasetId} onChange={(e) => setSelectedDatasetId(e.target.value)}>
                            {datasets.map((d) => (
                                <option key={d.id} value={d.id}>{d.id.slice(0, 8)}... ({d.total_images} images)</option>
                            ))}
                        </select>
                    </div>

                    <div className="panel-block">
                        <h3>Samples from Dataset</h3>
                        <div className="sample-grid">
                            {datasetImages.slice(0, 8).map((img) => (
                                <button
                                    type="button"
                                    key={img.id}
                                    className={`sample-thumb ${selectedImageId === img.id ? 'active' : ''}`}
                                    onClick={() => selectSampleImage(img)}
                                >
                                    <img src={getImageSrc(img.url)} alt={img.file_name} loading="lazy" />
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="panel-block">
                        <h3>Upload Image</h3>
                        <button type="button" className="btn btn-secondary wide-btn" onClick={() => fileInputRef.current?.click()}>
                            <FiUpload /> Select File
                        </button>
                        <input ref={fileInputRef} type="file" accept="image/*" onChange={handleUploadFile} hidden />
                    </div>

                    <div className="panel-block">
                        <h3>Image URL</h3>
                        <div className="url-row">
                            <input
                                placeholder="https://..."
                                value={urlInput}
                                onChange={(e) => setUrlInput(e.target.value)}
                            />
                            <button type="button" className="btn btn-secondary" onClick={handleApplyUrl}>
                                <FiLink />
                            </button>
                        </div>
                    </div>

                    <div className="panel-block">
                        <h3>Webcam</h3>
                        <div className="webcam-actions">
                            {!webcamActive ? (
                                <button type="button" className="btn btn-secondary wide-btn" onClick={startWebcam}>
                                    <FiCamera /> Start Webcam
                                </button>
                            ) : (
                                <>
                                    <button type="button" className="btn btn-primary wide-btn" onClick={captureWebcam}>
                                        <FiPlay /> Capture Frame
                                    </button>
                                    <button type="button" className="btn btn-secondary wide-btn" onClick={stopWebcam}>
                                        Stop Webcam
                                    </button>
                                </>
                            )}
                        </div>
                    </div>
                </aside>

                <section className="model-canvas-panel">
                    <div className="model-canvas-wrap">
                        {webcamActive ? (
                            <video ref={videoRef} className="preview-image" muted playsInline />
                        ) : previewSrc ? (
                            <div className="preview-stage">
                                <div className="preview-frame">
                                    <img
                                        ref={previewImageRef}
                                        src={previewSrc}
                                        alt="preview"
                                        className="preview-image"
                                        onLoad={(event) => {
                                            const img = event.target;
                                            setPreviewImageSize({ width: img.naturalWidth, height: img.naturalHeight });
                                        }}
                                    />

                                    {!webcamActive && filteredPredictions.map((pred) => (
                                        <div
                                            key={pred.detection_id}
                                            className="pred-box"
                                            style={{
                                                left: `${(pred.x - (pred.width / 2)) * 100}%`,
                                                top: `${(pred.y - (pred.height / 2)) * 100}%`,
                                                width: `${pred.width * 100}%`,
                                                height: `${pred.height * 100}%`,
                                                backgroundColor: `rgba(120, 80, 255, ${opacityThreshold / 300})`,
                                            }}
                                        >
                                            {labelMode !== 'hidden' && (
                                                <span className="pred-label">
                                                    {labelMode === 'class_only'
                                                        ? pred.class
                                                        : `${pred.class} ${(pred.confidence * 100).toFixed(0)}%`}
                                                </span>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ) : (
                            <div className="preview-empty">Select a sample or upload an image</div>
                        )}
                    </div>
                    <div className="pred-count">{filteredPredictions.length} object detected</div>
                </section>

                <aside className="model-panel model-panel-right">
                    <div className="panel-block">
                        <h3>Controls</h3>
                        <label>Confidence Threshold: {confidenceThreshold}%</label>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            value={confidenceThreshold}
                            onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                        />

                        <label>Overlap Threshold: {overlapThreshold}%</label>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            value={overlapThreshold}
                            onChange={(e) => setOverlapThreshold(Number(e.target.value))}
                        />

                        <label>Opacity Threshold: {opacityThreshold}%</label>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            value={opacityThreshold}
                            onChange={(e) => setOpacityThreshold(Number(e.target.value))}
                        />

                        <label>Label Display Mode</label>
                        <select value={labelMode} onChange={(e) => setLabelMode(e.target.value)}>
                            <option value="draw_confidence">Draw Confidence</option>
                            <option value="class_only">Class Only</option>
                            <option value="hidden">Hidden</option>
                        </select>

                        <label>Label Format</label>
                        <div className="label-format-constant">
                            <span>Uploaded (YOLO x,y,w,h)</span>
                        </div>
                        <small className="control-hint">Using normalized YOLO labels for uploaded dataset annotations.</small>
                    </div>

                    <div className="panel-block json-panel">
                        <div className="json-header">
                            <h3>Predictions JSON</h3>
                            <button type="button" className="btn btn-secondary" onClick={handleCopyJson}>
                                <FiCopy /> Copy
                            </button>
                        </div>
                        <pre>{predictionsJson}</pre>
                    </div>

                    <div className="panel-block">
                        <h3>Classes</h3>
                        <div className="class-chip-row">
                            {classIds.length ? classIds.map((cid) => <span key={cid} className="class-chip">Class {cid}</span>) : <span className="class-chip">No classes</span>}
                        </div>
                    </div>
                </aside>
            </div>
        </div>
    );
}

export default ModelPreviewPage;
