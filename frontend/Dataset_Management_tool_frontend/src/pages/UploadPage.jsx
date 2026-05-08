import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FiUpload, FiFile, FiCheckCircle } from 'react-icons/fi';
import { datasetApi } from '../api/datasetApi';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorMessage from '../components/Common/ErrorMessage';
import './UploadPage.css';

function UploadPage() {
    const navigate = useNavigate();
    const [imagesFile, setImagesFile] = useState(null);
    const [labelsFile, setLabelsFile] = useState(null);
    const [formatType, setFormatType] = useState('yolo');
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!imagesFile || !labelsFile) {
            setError('Please select both images and labels files');
            return;
        }

        // Check file sizes (limit to 100GB each)
        const maxSize = 100 * 1024 * 1024 * 1024; // 100GB
        if (imagesFile.size > maxSize) {
            setError('Images ZIP file is too large. Maximum size is 100GB.');
            return;
        }
        if (labelsFile.size > maxSize) {
            setError('Labels ZIP file is too large. Maximum size is 100GB.');
            return;
        }

        // Check file types (allow ZIP files and files with .zip extension)
        const allowedTypes = ['application/zip', 'application/x-zip-compressed'];
        const isZipFile = (file) => {
            return allowedTypes.includes(file.type) || file.name.toLowerCase().endsWith('.zip');
        };
        if (!isZipFile(imagesFile)) {
            setError('Images file must be a ZIP file.');
            return;
        }
        if (!isZipFile(labelsFile)) {
            setError('Labels file must be a ZIP file.');
            return;
        }

        try {
            setUploading(true);
            setError(null);
            setUploadProgress(0);

            const formData = new FormData();
            formData.append('images_zip', imagesFile);
            formData.append('labels_zip', labelsFile);
            formData.append('format_type', formatType);

            const response = await datasetApi.uploadDataset(formData, (progress) => {
                setUploadProgress(progress);
            });

            setResult(response);
        } catch (err) {
            setError(err.response?.data?.detail || 'Upload failed');
        } finally {
            setUploading(false);
        }
    };

    if (result) {
        return (
            <div className="upload-page fade-in">
                <div className="success-card card glass">
                    <FiCheckCircle className="success-icon" />
                    <h2>Upload Successful!</h2>
                    <p className="text-secondary">Your dataset has been processed and validated</p>

                    <div className="result-stats">
                        <div className="result-stat">
                            <span className="stat-label">Total Images</span>
                            <span className="stat-value">{result.analysis_summary.total_images}</span>
                        </div>
                        <div className="result-stat">
                            <span className="stat-label">Total Classes</span>
                            <span className="stat-value">{result.analysis_summary.total_classes}</span>
                        </div>
                        <div className="result-stat">
                            <span className="stat-label">Total Objects</span>
                            <span className="stat-value">{result.analysis_summary.total_objects}</span>
                        </div>
                    </div>

                    <div className="result-actions">
                        <button
                            className="btn btn-primary"
                            onClick={() => navigate(`/datasets/${result.dataset_id}`)}
                        >
                            View Dataset
                        </button>
                        <button
                            className="btn btn-secondary"
                            onClick={() => {
                                setResult(null);
                                setImagesFile(null);
                                setLabelsFile(null);
                            }}
                        >
                            Upload Another
                        </button>
                        <a
                            href={result.download_url}
                            className="btn btn-secondary"
                            download
                        >
                            Download
                        </a>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="upload-page fade-in">
            <div className="page-header">
                <h1 className="page-title">Upload Dataset</h1>
                <p className="page-subtitle">Upload and validate your dataset</p>
            </div>

            {error && <ErrorMessage message={error} onClose={() => setError(null)} />}

            <div className="upload-container">
                <form onSubmit={handleSubmit} className="upload-form card">
                    <div className="form-group">
                        <label htmlFor="format-type">Dataset Format</label>
                        <select
                            id="format-type"
                            value={formatType}
                            onChange={(e) => setFormatType(e.target.value)}
                            disabled={uploading}
                        >
                            <option value="yolo">YOLO</option>
                            <option value="coco">COCO</option>
                            <option value="pascal_voc">Pascal VOC</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label htmlFor="images-file">Images ZIP File</label>
                        <div className="file-input-wrapper">
                            <input
                                type="file"
                                id="images-file"
                                accept=".zip"
                                onChange={(e) => setImagesFile(e.target.files[0])}
                                disabled={uploading}
                            />
                            <div className="file-input-display">
                                <FiFile />
                                <span>{imagesFile ? imagesFile.name : 'Choose images.zip file'}</span>
                            </div>
                        </div>
                    </div>

                    <div className="form-group">
                        <label htmlFor="labels-file">Labels ZIP File</label>
                        <div className="file-input-wrapper">
                            <input
                                type="file"
                                id="labels-file"
                                accept=".zip"
                                onChange={(e) => setLabelsFile(e.target.files[0])}
                                disabled={uploading}
                            />
                            <div className="file-input-display">
                                <FiFile />
                                <span>{labelsFile ? labelsFile.name : 'Choose labels.zip file'}</span>
                            </div>
                        </div>
                    </div>

                    {uploading && (
                        <div className="upload-progress">
                            <div className="progress-bar">
                                <div
                                    className="progress-fill"
                                    style={{ width: `${uploadProgress}%` }}
                                ></div>
                            </div>
                            <p className="progress-text">{uploadProgress}% uploaded</p>
                        </div>
                    )}

                    <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={uploading || !imagesFile || !labelsFile}
                    >
                        {uploading ? (
                            <>
                                <div className="spinner spinner-small"></div>
                                Uploading...
                            </>
                        ) : (
                            <>
                                <FiUpload />
                                Upload Dataset
                            </>
                        )}
                    </button>
                </form>

                <div className="upload-info card glass">
                    <h3>Upload Instructions</h3>
                    <ul>
                        <li>Prepare two ZIP files: one for images and one for labels</li>
                        <li>Supported formats: YOLO, COCO, Pascal VOC</li>
                        <li>Images should be in JPG, JPEG, or PNG format</li>
                        <li>Labels should match the selected format specification</li>
                        <li>Ensure image and label filenames match (except extension)</li>
                    </ul>
                </div>
            </div>
        </div>
    );
}

export default UploadPage;
