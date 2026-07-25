import { useEffect, useMemo, useState } from 'react';
import { FiPlay, FiRefreshCw, FiStopCircle } from 'react-icons/fi';
import { datasetApi } from '../api/datasetApi';
import API_BASE_URL from '../config';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorMessage from '../components/Common/ErrorMessage';
import './TrainingPage.css';

const ACTIVE_STATUSES = new Set(['queued', 'preparing', 'running', 'cancelling']);
const safeSlice = (value, n = 8) => (typeof value === 'string' ? value.slice(0, n) : 'unknown');
const formatDateTime = (value) => {
    if (!value) return '-';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? '-' : date.toLocaleString();
};
const normalizeApiError = (error, fallback) => {
    const detail = error?.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
        return detail
            .map((item) => item?.msg || JSON.stringify(item))
            .join(' | ');
    }
    if (detail && typeof detail === 'object') return JSON.stringify(detail);
    return fallback;
};
const formatMetricValue = (value) => {
    const asNumber = Number(value);
    if (Number.isFinite(asNumber)) return asNumber.toFixed(4);
    return String(value ?? '-');
};

const formatDuration = (seconds) => {
    if (!Number.isFinite(seconds) || seconds <= 0) return '0s';
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return [hrs ? `${hrs}h` : null, mins ? `${mins}m` : null, `${secs}s`].filter(Boolean).join(' ');
};

const getJobTrainingTime = (job) => {
    const metricTime = Number(job.metrics?.total_training_time);
    if (Number.isFinite(metricTime) && metricTime > 0) return metricTime;
    if (!job?.started_at) return 0;
    const start = Date.parse(job.started_at);
    const end = job.finished_at ? Date.parse(job.finished_at) : Date.now();
    if (Number.isNaN(start) || Number.isNaN(end)) return 0;
    return Math.max(0, (end - start) / 1000);
};

const getJobTrainingImages = (job) => {
    const metricImages = Number(job.metrics?.images_trained ?? job.metrics?.total_images);
    if (Number.isFinite(metricImages) && metricImages >= 0) return metricImages;
    return 0;
};

const boolFromSelect = (value) => value === 'true';

const getDownloadUrl = (absolutePath) => {
    if (!absolutePath) return null;
    const match = absolutePath.match(/training\/jobs\/.+/);
    if (match) {
        return `/api/storage/${match[0]}`;
    }
    return null;
};

function TrainingPage() {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [notice, setNotice] = useState('');
    const [starting, setStarting] = useState(false);

    const [datasets, setDatasets] = useState([]);
    const [jobs, setJobs] = useState([]);
    const [selectedJobId, setSelectedJobId] = useState('');
    const [stoppingJobId, setStoppingJobId] = useState('');
    const [deviceInfo, setDeviceInfo] = useState(null);

    const [form, setForm] = useState({
        dataset_id: '',
        model: 'yolov8n.pt',
        epochs: 50,
        batch_size: 16,
        image_size: 640,
        learning_rate: 0.01,
        device: 'auto',
        val_split: 0.2,
        test_split: 0.1,
        augmentation_enabled: false,
        experiment_name: 'dataset_training',
        run_name: '',
        patience: 100,
    });

    const selectedJob = useMemo(
        () => jobs.find((job) => job.job_id === selectedJobId) || jobs[0] || null,
        [jobs, selectedJobId],
    );

    const fetchDatasets = async () => {
        const data = await datasetApi.listDatasets({ page: 1, limit: 100, sort_by: 'created_at', order: 'desc' });
        const list = data.datasets || [];
        setDatasets(list);
        if (!form.dataset_id && list[0]?.id) {
            setForm((prev) => ({ ...prev, dataset_id: list[0].id }));
        }
    };

    const fetchJobs = async () => {
        const response = await datasetApi.listTrainingJobs();
        const list = response.jobs || [];
        setJobs(list);
        if (!selectedJobId && list[0]?.job_id) {
            setSelectedJobId(list[0].job_id);
        }
    };

    const fetchDevices = async () => {
        try {
            const response = await datasetApi.getAvailableDevices();
            setDeviceInfo(response);
            // Always set device to recommended (GPU 0 if available, CPU otherwise)
            setForm((prev) => ({ ...prev, device: response.recommended_device }));
        } catch (e) {
            // Silently handle device detection failure - default to GPU 0
            console.warn('Device detection failed:', e);
        }
    };

    useEffect(() => {
        const run = async () => {
            try {
                setLoading(true);
                setError('');
                await Promise.all([fetchDatasets(), fetchJobs(), fetchDevices()]);
            } catch (e) {
                setError(normalizeApiError(e, 'Failed to load training page'));
            } finally {
                setLoading(false);
            }
        };
        run();
    }, []);

    useEffect(() => {
        const hasActiveJob = jobs.some((job) => ACTIVE_STATUSES.has(job.status));
        if (!hasActiveJob) return undefined;

        const timer = setInterval(() => {
            fetchJobs().catch(() => { });
        }, 3000);
        return () => clearInterval(timer);
    }, [jobs]);

    const handleStartTraining = async () => {
        if (!form.dataset_id) {
            setError('Select a dataset first.');
            return;
        }
        if ((Number(form.val_split) + Number(form.test_split)) >= 0.9) {
            setError('Validation + test split must be less than 0.9');
            return;
        }
        // Validate numeric fields
        const numericFields = ['epochs', 'batch_size', 'image_size', 'learning_rate', 'val_split', 'test_split'];
        for (const field of numericFields) {
            if (isNaN(Number(form[field]))) {
                setError(`${field.replace('_', ' ')} must be a valid number.`);
                return;
            }
        }

        try {
            setStarting(true);
            setError('');
            setNotice('');
            const payload = {
                ...form,
                epochs: Number(form.epochs),
                batch_size: Number(form.batch_size),
                image_size: Number(form.image_size),
                learning_rate: Number(form.learning_rate),
                val_split: Number(form.val_split),
                test_split: Number(form.test_split),
                augmentation_enabled: Boolean(form.augmentation_enabled),
                run_name: form.run_name || null,
                patience: Number(form.patience),
            };
            const started = await datasetApi.startTraining(payload);
            setNotice('Training job started.');
            await fetchJobs();
            setSelectedJobId(started.job_id);
        } catch (e) {
            setError(normalizeApiError(e, 'Failed to start training job'));
        } finally {
            setStarting(false);
        }
    };

    const handleStopTraining = async (jobId) => {
        if (!jobId) return;
        try {
            setStoppingJobId(jobId);
            setError('');
            setNotice('Sending stop request...');
            await datasetApi.stopTraining(jobId);
            await fetchJobs();
            if (selectedJobId === jobId) {
                setSelectedJobId('');
            }
        } catch (e) {
            setError(normalizeApiError(e, 'Failed to stop training job'));
        } finally {
            setStoppingJobId('');
        }
    };

    const isTerminalJob = (status) => ['completed', 'failed', 'cancelled'].includes(status);

    const handleCancelJob = async (event, job) => {
        event.stopPropagation();
        if (!job) return;

        try {
            setStoppingJobId(job.job_id);
            setError('');
            setNotice('Removing job from dashboard...');

            if (isTerminalJob(job.status)) {
                await datasetApi.deleteTrainingJob(job.job_id);
            } else {
                await handleStopTraining(job.job_id);
            }

            setJobs((prevJobs) => prevJobs.filter((item) => item.job_id !== job.job_id));
            if (selectedJobId === job.job_id) {
                setSelectedJobId('');
            }
        } catch (e) {
            setError(normalizeApiError(e, 'Failed to remove training job'));
        } finally {
            setStoppingJobId('');
        }
    };

    return (
        <div className="training-page fade-in">
            <div className="training-header">
                <h1>Train Model</h1>
                <p>Train your uploaded dataset with YOLO and monitor job status live.</p>
            </div>

            {loading && <LoadingSpinner message="Loading training workspace..." />}
            {error && <ErrorMessage message={error} onClose={() => setError('')} />}
            {notice && <div className="training-notice">{notice}</div>}

            <div className="training-layout">
                <section className="training-panel">
                    <h3>Training Configuration</h3>

                    <label>Dataset</label>
                    {datasets.length > 0 ? (
                        <select
                            value={form.dataset_id}
                            onChange={(e) => setForm((prev) => ({ ...prev, dataset_id: e.target.value }))}
                        >
                            {datasets.map((dataset) => (
                                <option key={dataset.id} value={dataset.id}>
                                    {safeSlice(dataset.id)}... ({dataset.total_images ?? 0} images)
                                </option>
                            ))}
                        </select>
                    ) : (
                        <div className="training-empty">No datasets found. Upload a dataset first.</div>
                    )}

                    <label>Base Model</label>
                    <select
                        value={form.model}
                        onChange={(e) => setForm((prev) => ({ ...prev, model: e.target.value }))}
                    >
                        <option value="yolov8n.pt">YOLOv8 Nano (Fast)</option>
                        <option value="yolov8s.pt">YOLOv8 Small</option>
                        <option value="yolov8m.pt">YOLOv8 Medium</option>
                        <option value="yolo11n.pt">YOLO11 Nano (Fast)</option>
                        <option value="yolo11s.pt">YOLO11 Small</option>
                        <option value="yolo11m.pt">YOLO11 Medium</option>
                        <option value="yolo26s.pt">YOLO26 Small</option>
                        <option value="yolo26m.pt">YOLO26 Medium</option>
                        <option value="yolo26l.pt">YOLO26 Large</option>
                    </select>

                    <div className="training-grid-2">
                        <div>
                            <label>Epochs</label>
                            <input
                                type="number"
                                min="1"
                                max="1000"
                                value={form.epochs}
                                onChange={(e) => setForm((prev) => ({ ...prev, epochs: e.target.value }))}
                            />
                        </div>
                        <div>
                            <label>Batch Size</label>
                            <input
                                type="number"
                                min="1"
                                max="256"
                                value={form.batch_size}
                                onChange={(e) => setForm((prev) => ({ ...prev, batch_size: e.target.value }))}
                            />
                        </div>
                    </div>

                    <div className="training-grid-2">
                        <div>
                            <label>Image Size</label>
                            <input
                                type="number"
                                min="128"
                                max="2048"
                                step="32"
                                value={form.image_size}
                                onChange={(e) => setForm((prev) => ({ ...prev, image_size: e.target.value }))}
                            />
                        </div>
                        <div>
                            <label>
                                Device
                                {deviceInfo && (
                                    <div style={{ fontSize: '0.85em', color: '#28a745', marginTop: '4px', fontWeight: 'bold' }}>
                                        ✓ {deviceInfo.message}
                                    </div>
                                )}
                            </label>
                            <select
                                value={form.device}
                                onChange={(e) => setForm((prev) => ({ ...prev, device: e.target.value }))}
                            >
                                {deviceInfo?.devices?.map((gpu) => (
                                    <option key={gpu.device_id} value={gpu.device_id}>
                                        🚀 {gpu.device_name} ({gpu.total_memory_gb}GB)
                                    </option>
                                ))}
                                <option value="0">🚀 GPU 0 (Default)</option>
                                <option value="cpu" style={{ color: '#d9534f' }}>🐢 CPU (Slow)</option>
                            </select>
                        </div>
                    </div>

                    <div className="training-grid-2">
                        <div>
                            <label>Learning Rate</label>
                            <input
                                type="number"
                                min="0.000001"
                                max="1"
                                step="0.0001"
                                value={form.learning_rate}
                                onChange={(e) => setForm((prev) => ({ ...prev, learning_rate: e.target.value }))}
                            />
                        </div>
                        <div>
                            <label>Patience</label>
                            <input
                                type="number"
                                min="0"
                                max="1000"
                                value={form.patience}
                                onChange={(e) => setForm((prev) => ({ ...prev, patience: e.target.value }))}
                            />
                        </div>
                    </div>

                    <div className="training-grid-2">
                        <div>
                            <label>Validation Split</label>
                            <input
                                type="number"
                                min="0.05"
                                max="0.4"
                                step="0.01"
                                value={form.val_split}
                                onChange={(e) => setForm((prev) => ({ ...prev, val_split: e.target.value }))}
                            />
                        </div>
                        <div>
                            <label>Test Split</label>
                            <input
                                type="number"
                                min="0"
                                max="0.4"
                                step="0.01"
                                value={form.test_split}
                                onChange={(e) => setForm((prev) => ({ ...prev, test_split: e.target.value }))}
                            />
                        </div>
                    </div>

                    <h4>Augmentation</h4>
                    <label>Augmentation Enabled</label>
                    <select
                        value={String(form.augmentation_enabled)}
                        onChange={(e) => setForm((prev) => ({ ...prev, augmentation_enabled: boolFromSelect(e.target.value) }))}
                    >
                        <option value="false">No</option>
                        <option value="true">Yes</option>
                    </select>

                    <h4>MLflow</h4>
                    <label>Experiment Name</label>
                    <input
                        type="text"
                        value={form.experiment_name}
                        onChange={(e) => setForm((prev) => ({ ...prev, experiment_name: e.target.value }))}
                    />

                    <label>Run Name (Optional)</label>
                    <input
                        type="text"
                        value={form.run_name}
                        onChange={(e) => setForm((prev) => ({ ...prev, run_name: e.target.value }))}
                    />

                    <button type="button" className="btn btn-primary wide-btn" onClick={handleStartTraining} disabled={starting}>
                        <FiPlay /> {starting ? 'Starting...' : 'Start Training'}
                    </button>
                </section>

                <section className="training-panel">
                    <div className="training-panel-row">
                        <h3>Training Jobs</h3>
                        <button type="button" className="btn btn-secondary" onClick={() => fetchJobs()}>
                            <FiRefreshCw /> Refresh
                        </button>
                    </div>

                    {!jobs.length && <div className="training-empty">No jobs yet. Start your first training run.</div>}

                    <div className="job-list">
                        {jobs.map((job) => (
                            <div
                                key={job.job_id}
                                className={`job-item ${selectedJob?.job_id === job.job_id ? 'active' : ''}`}
                                onClick={() => setSelectedJobId(job.job_id)}
                                onKeyDown={(e) => e.key === 'Enter' && setSelectedJobId(job.job_id)}
                                role="button"
                                tabIndex={0}
                            >
                                <div className="job-item-top">
                                    <div className="job-item-meta">
                                        <span className={`job-status ${job.status}`}>{job.status}</span>
                                        <span>{formatDateTime(job.created_at)}</span>
                                    </div>
                                    <button
                                        type="button"
                                        className="job-item-cancel"
                                        onClick={(e) => handleCancelJob(e, job)}
                                        disabled={stoppingJobId === job.job_id}
                                    >
                                        <FiStopCircle />
                                        {stoppingJobId === job.job_id ? 'Cancelling...' : isTerminalJob(job.status) ? 'Remove' : 'Cancel'}
                                    </button>
                                </div>
                                <div className="job-item-id">{safeSlice(job.job_id)}...</div>
                                <div className="job-item-dataset">Dataset: {safeSlice(job.dataset_id)}...</div>
                            </div>
                        ))}
                    </div>
                </section>

                <section className="training-panel">
                    <h3>Job Details</h3>
                    {!selectedJob && <div className="training-empty">Select a job to see details.</div>}

                    {selectedJob && (
                        <div className="job-detail">
                            <div className="job-detail-action">
                                <p><strong>Status:</strong> <span className={`job-status ${selectedJob.status}`}>{selectedJob.status}</span></p>
                                {(selectedJob.status === 'queued' || selectedJob.status === 'preparing' || selectedJob.status === 'running' || selectedJob.status === 'cancelling') && (
                                    <button
                                        type="button"
                                        className="btn btn-secondary small-btn"
                                        onClick={() => handleStopTraining(selectedJob.job_id)}
                                        disabled={stoppingJobId === selectedJob.job_id}
                                    >
                                        <FiStopCircle /> {stoppingJobId === selectedJob.job_id ? 'Stopping...' : 'Stop'}
                                    </button>
                                )}
                            </div>
                            <p><strong>Dataset:</strong> {selectedJob.dataset_id || '-'}</p>
                            <p><strong>Model:</strong> {selectedJob.params?.model}</p>
                            <p><strong>Epochs:</strong> {selectedJob.params?.epochs}</p>
                            <p><strong>Elapsed:</strong> {formatDuration(getJobTrainingTime(selectedJob))}</p>
                            <p><strong>Images Trained:</strong> {getJobTrainingImages(selectedJob)}</p>

                            {selectedJob.metrics && (
                                <div className="metric-grid">
                                    {Object.entries(selectedJob.metrics).map(([key, value]) => (
                                        <div key={key} className="metric-card">
                                            <span>{key}</span>
                                            <strong>{formatMetricValue(value)}</strong>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {selectedJob.mlflow && (
                                <div className="artifact-block">
                                    <h4>MLflow</h4>
                                    <p><strong>Enabled:</strong> {String(selectedJob.mlflow.enabled)}</p>
                                    <p><strong>Experiment:</strong> {selectedJob.mlflow.experiment_name || '-'}</p>
                                    <p><strong>Run ID:</strong> {selectedJob.mlflow.run_id || '-'}</p>
                                    <p><strong>Tracking URI:</strong> {selectedJob.mlflow.tracking_uri || '-'}</p>
                                    <p><strong>Start Time:</strong> {selectedJob.mlflow.training_start_time || '-'}</p>
                                    <p><strong>End Time:</strong> {selectedJob.mlflow.training_end_time || '-'}</p>
                                </div>
                            )}

                            {selectedJob.artifacts && selectedJob.status === 'completed' && (
                                <div className="artifact-block">
                                    <a
                                        href={`${API_BASE_URL}/train/jobs/${selectedJob.job_id}/download`}
                                        download={`training_results_${selectedJob.job_id}.zip`}
                                        className="btn btn-primary"
                                        style={{ textDecoration: 'none' }}
                                    >
                                        Download Full Results
                                    </a>
                                </div>
                            )}

                            {selectedJob.error && (
                                <div className="job-error">
                                    <strong>Error:</strong> {selectedJob.error}
                                </div>
                            )}

                            <div className="log-block">
                                <h4>Logs</h4>
                                <pre>{(selectedJob.logs || []).join('\n') || 'No logs yet.'}</pre>
                            </div>
                        </div>
                    )}
                </section>
            </div>
        </div>
    );
}

export default TrainingPage;
