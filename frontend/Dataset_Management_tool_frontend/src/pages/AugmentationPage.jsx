import { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { FiZap, FiRotateCw, FiSun, FiDroplet } from 'react-icons/fi';
import { datasetApi } from '../api/datasetApi';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorMessage from '../components/Common/ErrorMessage';
import './AugmentationPage.css';

function AugmentationPage() {
    const location = useLocation();
    const navigate = useNavigate();
    const [datasets, setDatasets] = useState([]);
    const [selectedDataset, setSelectedDataset] = useState(location.state?.datasetId || '');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(false);

    const [formData, setFormData] = useState({
        count: 1,
        horizontal_flip: false,
        vertical_flip: false,
        rotation: 0,
        brightness: 0,
        contrast: 0,
        blur: 0,
        noise: 0,
        export_format: 'yolo',
    });

    const redirectTimer = useRef(null);

    useEffect(() => {
        fetchDatasets();
        return () => {
            if (redirectTimer.current) clearTimeout(redirectTimer.current);
        };
    }, []);

    const fetchDatasets = async () => {
        try {
            const response = await datasetApi.listDatasets({ page: 1, limit: 100 });
            setDatasets(response.datasets);
        } catch (err) {
            setError('Failed to load datasets');
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!selectedDataset) {
            setError('Please select a dataset');
            return;
        }

        try {
            setLoading(true);
            setError(null);
            setSuccess(false);

            await datasetApi.augmentDataset({
                dataset_id: selectedDataset,
                ...formData,
            });

            setSuccess(true);
            redirectTimer.current = setTimeout(() => {
                navigate(`/datasets/${selectedDataset}`);
            }, 2000);
        } catch (err) {
            setError(err.response?.data?.detail || 'Augmentation failed');
        } finally {
            setLoading(false);
        }
    };

    const handleChange = (field, value) => {
        setFormData(prev => ({ ...prev, [field]: value }));
    };

    if (success) {
        return (
            <div className="augmentation-page fade-in">
                <div className="success-card card glass">
                    <FiZap className="success-icon" />
                    <h2>Augmentation Successful!</h2>
                    <p className="text-secondary">Your dataset has been augmented</p>
                    <p className="text-secondary">Redirecting to dataset details...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="augmentation-page fade-in">
            <div className="page-header">
                <h1 className="page-title">Augment Dataset</h1>
                <p className="page-subtitle">Apply transformations to increase dataset size</p>
            </div>

            {error && <ErrorMessage message={error} onClose={() => setError(null)} />}

            <form onSubmit={handleSubmit} className="augmentation-form">
                <div className="form-section card">
                    <h3>Select Dataset</h3>
                    <select
                        value={selectedDataset}
                        onChange={(e) => setSelectedDataset(e.target.value)}
                        disabled={loading}
                        required
                    >
                        <option value="">Choose a dataset...</option>
                        {datasets.map((dataset) => (
                            <option key={dataset.id} value={dataset.id}>
                                {dataset.id.substring(0, 8)}... - {dataset.format_type.toUpperCase()} ({dataset.total_images} images)
                            </option>
                        ))}
                    </select>
                </div>

                <div className="form-section card">
                    <h3>Basic Settings</h3>
                    <div className="form-group">
                        <label>Augmentation Count (per image)</label>
                        <input
                            type="number"
                            min="1"
                            max="10"
                            value={formData.count}
                            onChange={(e) => handleChange('count', parseInt(e.target.value))}
                            disabled={loading}
                        />
                        <small className="text-secondary">Number of augmented versions to create per image</small>
                    </div>

                    <div className="form-group">
                        <label>Export Format</label>
                        <select
                            value={formData.export_format}
                            onChange={(e) => handleChange('export_format', e.target.value)}
                            disabled={loading}
                        >
                            <option value="yolo">YOLO</option>
                            <option value="coco">COCO</option>
                            <option value="pascal_voc">Pascal VOC</option>
                        </select>
                    </div>
                </div>

                <div className="form-section card">
                    <h3>Transformations</h3>

                    <div className="augmentation-options">
                        <div className="option-card">
                            <div className="option-header">
                                <FiRotateCw />
                                <h4>Flips</h4>
                            </div>
                            <div className="option-controls">
                                <label className="checkbox-label">
                                    <input
                                        type="checkbox"
                                        checked={formData.horizontal_flip}
                                        onChange={(e) => handleChange('horizontal_flip', e.target.checked)}
                                        disabled={loading}
                                    />
                                    <span>Horizontal Flip</span>
                                </label>
                                <label className="checkbox-label">
                                    <input
                                        type="checkbox"
                                        checked={formData.vertical_flip}
                                        onChange={(e) => handleChange('vertical_flip', e.target.checked)}
                                        disabled={loading}
                                    />
                                    <span>Vertical Flip</span>
                                </label>
                            </div>
                        </div>

                        <div className="option-card">
                            <div className="option-header">
                                <FiRotateCw />
                                <h4>Rotation</h4>
                            </div>
                            <div className="option-controls">
                                <input
                                    type="range"
                                    min="0"
                                    max="360"
                                    value={formData.rotation}
                                    onChange={(e) => handleChange('rotation', parseInt(e.target.value))}
                                    disabled={loading}
                                />
                                <span className="range-value">{formData.rotation}°</span>
                            </div>
                        </div>

                        <div className="option-card">
                            <div className="option-header">
                                <FiSun />
                                <h4>Brightness</h4>
                            </div>
                            <div className="option-controls">
                                <input
                                    type="range"
                                    min="-1"
                                    max="1"
                                    step="0.1"
                                    value={formData.brightness}
                                    onChange={(e) => handleChange('brightness', parseFloat(e.target.value))}
                                    disabled={loading}
                                />
                                <span className="range-value">{formData.brightness.toFixed(1)}</span>
                            </div>
                        </div>

                        <div className="option-card">
                            <div className="option-header">
                                <FiSun />
                                <h4>Contrast</h4>
                            </div>
                            <div className="option-controls">
                                <input
                                    type="range"
                                    min="-1"
                                    max="1"
                                    step="0.1"
                                    value={formData.contrast}
                                    onChange={(e) => handleChange('contrast', parseFloat(e.target.value))}
                                    disabled={loading}
                                />
                                <span className="range-value">{formData.contrast.toFixed(1)}</span>
                            </div>
                        </div>

                        <div className="option-card">
                            <div className="option-header">
                                <FiDroplet />
                                <h4>Blur</h4>
                            </div>
                            <div className="option-controls">
                                <input
                                    type="range"
                                    min="0"
                                    max="10"
                                    value={formData.blur}
                                    onChange={(e) => handleChange('blur', parseInt(e.target.value))}
                                    disabled={loading}
                                />
                                <span className="range-value">{formData.blur}</span>
                            </div>
                        </div>

                        <div className="option-card">
                            <div className="option-header">
                                <FiDroplet />
                                <h4>Noise</h4>
                            </div>
                            <div className="option-controls">
                                <input
                                    type="range"
                                    min="0"
                                    max="1"
                                    step="0.1"
                                    value={formData.noise}
                                    onChange={(e) => handleChange('noise', parseFloat(e.target.value))}
                                    disabled={loading}
                                />
                                <span className="range-value">{formData.noise.toFixed(1)}</span>
                            </div>
                        </div>
                    </div>
                </div>

                <button
                    type="submit"
                    className="btn btn-primary btn-large"
                    disabled={loading || !selectedDataset}
                >
                    {loading ? (
                        <>
                            <div className="spinner spinner-small"></div>
                            Processing...
                        </>
                    ) : (
                        <>
                            <FiZap />
                            Apply Augmentation
                        </>
                    )}
                </button>
            </form>
        </div>
    );
}

export default AugmentationPage;
