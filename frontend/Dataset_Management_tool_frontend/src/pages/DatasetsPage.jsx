import { useState, useEffect, useRef } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { FiImage, FiLayers, FiCalendar, FiFilter } from 'react-icons/fi';
import { datasetApi } from '../api/datasetApi';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorMessage from '../components/Common/ErrorMessage';
import './DatasetsPage.css';

function DatasetsPage() {
    const [searchParams] = useSearchParams();
    const [datasets, setDatasets] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [formatFilter, setFormatFilter] = useState('');
    const [sortBy, setSortBy] = useState(searchParams.get('sort_by') || 'created_at');
    const [order, setOrder] = useState(searchParams.get('order') || 'desc');

    useEffect(() => {
        const controller = new AbortController();
        fetchDatasets(controller.signal);
        return () => controller.abort();
    }, [page, formatFilter, sortBy, order]);

    const fetchDatasets = async (signal) => {
        try {
            setLoading(true);
            setError(null);

            const params = {
                page,
                limit: 12,
                sort_by: sortBy,
                order,
            };

            if (formatFilter) {
                params.format_type = formatFilter;
            }

            const response = await datasetApi.listDatasets(params, { signal });
            setDatasets(response.datasets || []);
            setTotalPages(response.total_pages || 1);
        } catch (err) {
            if (err.name === 'CanceledError' || err.name === 'AbortError') return;
            setError(err.response?.data?.detail || 'Failed to load datasets');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="datasets-page fade-in">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Datasets</h1>
                    <p className="page-subtitle">Manage and explore your datasets</p>
                </div>
                <Link to="/upload" className="btn btn-primary">
                    Upload Dataset
                </Link>
            </div>

            {error && <ErrorMessage message={error} onClose={() => setError(null)} />}

            <div className="filter-bar card glass">
                <div className="filter-group">
                    <FiFilter />
                    <label htmlFor="format-filter">Format:</label>
                    <select
                        id="format-filter"
                        value={formatFilter}
                        onChange={(e) => {
                            setFormatFilter(e.target.value);
                            setPage(1);
                        }}
                    >
                        <option value="">All Formats</option>
                        <option value="yolo">YOLO</option>
                        <option value="coco">COCO</option>
                        <option value="pascal_voc">Pascal VOC</option>
                    </select>
                </div>
                <div className="filter-group">
                    <label htmlFor="sort-by">Sort:</label>
                    <select
                        id="sort-by"
                        value={sortBy}
                        onChange={(e) => {
                            setSortBy(e.target.value);
                            setPage(1);
                        }}
                    >
                        <option value="created_at">Created Date</option>
                        <option value="total_images">Total Images</option>
                        <option value="total_classes">Total Classes</option>
                    </select>
                </div>
                <div className="filter-group">
                    <label htmlFor="sort-order">Order:</label>
                    <select
                        id="sort-order"
                        value={order}
                        onChange={(e) => {
                            setOrder(e.target.value);
                            setPage(1);
                        }}
                    >
                        <option value="desc">Descending</option>
                        <option value="asc">Ascending</option>
                    </select>
                </div>
            </div>

            {loading ? (
                <LoadingSpinner message="Loading datasets..." />
            ) : datasets.length === 0 ? (
                <div className="empty-state card">
                    <FiImage className="empty-icon" />
                    <h3>No datasets found</h3>
                    <p className="text-secondary">Upload your first dataset to get started</p>
                    <Link to="/upload" className="btn btn-primary mt-md">Upload Dataset</Link>
                </div>
            ) : (
                <>
                    <div className="datasets-grid grid grid-3">
                        {datasets.map((dataset) => (
                            <Link key={dataset.id} to={`/datasets/${dataset.id}`} className="dataset-card card">
                                <div className="dataset-header">
                                    <span className="dataset-format">{dataset.format_type.toUpperCase()}</span>
                                    <div className="dataset-date">
                                        <FiCalendar />
                                        <span>{new Date(dataset.created_at).toLocaleDateString()}</span>
                                    </div>
                                </div>

                                <div className="dataset-id">ID: {dataset.id.substring(0, 8)}...</div>

                                <div className="dataset-stats-grid">
                                    <div className="dataset-stat">
                                        <FiImage className="stat-icon" />
                                        <div>
                                            <div className="stat-value">{dataset.total_images}</div>
                                            <div className="stat-label">Images</div>
                                        </div>
                                    </div>

                                    <div className="dataset-stat">
                                        <FiLayers className="stat-icon" />
                                        <div>
                                            <div className="stat-value">{dataset.total_classes}</div>
                                            <div className="stat-label">Classes</div>
                                        </div>
                                    </div>

                                    <div className="dataset-stat">
                                        <div className="stat-icon">🎯</div>
                                        <div>
                                            <div className="stat-value">{dataset.total_objects}</div>
                                            <div className="stat-label">Objects</div>
                                        </div>
                                    </div>
                                </div>
                            </Link>
                        ))}
                    </div>

                    {totalPages > 1 && (
                        <div className="pagination">
                            <button
                                className="btn btn-secondary"
                                onClick={() => setPage(p => Math.max(1, p - 1))}
                                disabled={page === 1}
                            >
                                Previous
                            </button>
                            <span className="page-info">
                                Page {page} of {totalPages}
                            </span>
                            <button
                                className="btn btn-secondary"
                                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
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

export default DatasetsPage;
