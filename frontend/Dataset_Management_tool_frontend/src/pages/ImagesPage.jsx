import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { FiImage, FiCheckCircle, FiAlertCircle } from 'react-icons/fi';
import { datasetApi } from '../api/datasetApi';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorMessage from '../components/Common/ErrorMessage';
import './ImagesPage.css';

function ImagesPage() {
    const [images, setImages] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);

    useEffect(() => {
        fetchImages();
    }, [page]);

    const fetchImages = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await datasetApi.listAllImages({ page, limit: 48 });
            setImages(response.images || []);
            setTotalPages(response.total_pages || 1);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to load images');
        } finally {
            setLoading(false);
        }
    };

    const getImageSrc = (url) => (url?.startsWith('/api') ? url : `/api${url}`);

    return (
        <div className="images-page fade-in">
            <div className="page-header">
                <div>
                    <h1 className="page-title">All Uploaded Images</h1>
                    <p className="page-subtitle">Browse all images across datasets</p>
                </div>
            </div>

            {error && <ErrorMessage message={error} onClose={() => setError(null)} />}

            {loading ? (
                <LoadingSpinner message="Loading images..." />
            ) : images.length === 0 ? (
                <div className="empty-state card">
                    <FiImage className="empty-icon" />
                    <h3>No images found</h3>
                    <p className="text-secondary">Upload a dataset to see images here</p>
                </div>
            ) : (
                <>
                    <div className="images-grid grid grid-4">
                        {images.map((img) => (
                            <Link key={img.id} to={`/datasets/${img.dataset_id}`} className="image-tile card">
                                <img src={getImageSrc(img.url)} alt={img.file_name} loading="lazy" />
                                <div className="tile-meta">
                                    <p className="file-name" title={img.file_name}>{img.file_name}</p>
                                    <p className="dataset-ref">Dataset: {img.dataset_id.slice(0, 8)}</p>
                                    <span className={`status ${img.has_label ? 'ok' : 'warn'}`}>
                                        {img.has_label ? <FiCheckCircle /> : <FiAlertCircle />}
                                        {img.has_label ? 'Labeled' : 'No Label'}
                                    </span>
                                </div>
                            </Link>
                        ))}
                    </div>

                    {totalPages > 1 && (
                        <div className="pagination">
                            <button
                                className="btn btn-secondary"
                                onClick={() => setPage((p) => Math.max(1, p - 1))}
                                disabled={page === 1}
                            >
                                Previous
                            </button>
                            <span className="page-info">Page {page} of {totalPages}</span>
                            <button
                                className="btn btn-secondary"
                                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                                disabled={page === totalPages}
                            >
                                Next
                            </button>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

export default ImagesPage;
