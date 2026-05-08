import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FiDatabase, FiImage, FiLayers, FiTrendingUp, FiTrash2, FiClock } from 'react-icons/fi';
import { datasetApi } from '../api/datasetApi';
import LoadingSpinner from '../components/Common/LoadingSpinner';
import ErrorMessage from '../components/Common/ErrorMessage';
import './HomePage.css';

function HomePage() {
    const [stats, setStats] = useState(null);
    const [recentDatasets, setRecentDatasets] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [deletingIds, setDeletingIds] = useState(new Set());

    useEffect(() => {
        fetchDashboardData();
    }, []);

    const getTrainingTimeSeconds = (job) => {
        const metricTime = Number(job.metrics?.total_training_time);
        if (Number.isFinite(metricTime) && metricTime > 0) return metricTime;
        if (!job.started_at) return 0;

        const start = Date.parse(job.started_at);
        const end = job.finished_at ? Date.parse(job.finished_at) : Date.now();
        if (Number.isNaN(start) || Number.isNaN(end)) return 0;
        return Math.max(0, (end - start) / 1000);
    };

    const getTrainingImages = (job) => {
        const metricImages = Number(job.metrics?.images_trained ?? job.metrics?.total_images);
        if (Number.isFinite(metricImages) && metricImages >= 0) return metricImages;
        return 0;
    };

    const fetchDashboardData = async () => {
        try {
            setLoading(true);
            setError(null);

            // Fetch recent datasets
            const datasetsResponse = await datasetApi.listDatasets({ page: 1, limit: 5, sort_by: 'created_at', order: 'desc' });
            setRecentDatasets(datasetsResponse.datasets);

            // Calculate aggregate stats
            const totalDatasets = datasetsResponse.total;
            const totalImages = datasetsResponse.datasets.reduce((sum, d) => sum + d.total_images, 0);
            const totalClasses = datasetsResponse.datasets.reduce((sum, d) => sum + d.total_classes, 0);
            const totalObjects = datasetsResponse.datasets.reduce((sum, d) => sum + d.total_objects, 0);

            const trainingResponse = await datasetApi.listTrainingJobs();
            const trainingJobs = trainingResponse.jobs || [];
            const totalTrainingTimeSeconds = trainingJobs.reduce((sum, job) => sum + getTrainingTimeSeconds(job), 0);
            const totalTrainingImages = trainingJobs.reduce((sum, job) => sum + getTrainingImages(job), 0);

            setStats({
                totalDatasets,
                totalImages,
                totalClasses,
                totalObjects,
                totalTrainingTimeSeconds,
                totalTrainingImages,
            });
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to load dashboard data');
        } finally {
            setLoading(false);
        }
    };

    const handleDeleteDataset = async (event, datasetId) => {
        event.preventDefault();
        event.stopPropagation();

        const confirmed = window.confirm('Delete this dataset permanently? This action cannot be undone.');
        if (!confirmed) {
            return;
        }

        try {
            setError(null);
            setDeletingIds((prev) => {
                const next = new Set(prev);
                next.add(datasetId);
                return next;
            });
            await datasetApi.deleteDataset(datasetId);
            await fetchDashboardData();
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to delete dataset');
        } finally {
            setDeletingIds((prev) => {
                const next = new Set(prev);
                next.delete(datasetId);
                return next;
            });
        }
    };

    const formatDuration = (seconds) => {
        if (!Number.isFinite(seconds) || seconds <= 0) return '0s';
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return [hrs ? `${hrs}h` : null, mins ? `${mins}m` : null, `${secs}s`].filter(Boolean).join(' ');
    };

    if (loading) {
        return <LoadingSpinner message="Loading dashboard..." />;
    }

    return (
        <div className="home-page fade-in">
            <div className="page-header">
                <h1 className="page-title">Dashboard</h1>
                <p className="page-subtitle">Welcome to your Dataset Management Hub</p>
            </div>

            {error && <ErrorMessage message={error} onClose={() => setError(null)} />}

            <div className="stats-grid grid grid-4">
                <Link to="/datasets?sort_by=created_at&order=desc" className="stat-card-link">
                <div className="stat-card card glass">
                    <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
                        <FiDatabase />
                    </div>
                    <div className="stat-content">
                        <h3 className="stat-value">{stats?.totalDatasets || 0}</h3>
                        <p className="stat-label">Total Datasets</p>
                    </div>
                </div>
                </Link>

                <Link to="/images" className="stat-card-link">
                <div className="stat-card card glass">
                    <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' }}>
                        <FiImage />
                    </div>
                    <div className="stat-content">
                        <h3 className="stat-value">{stats?.totalImages || 0}</h3>
                        <p className="stat-label">Total Images</p>
                    </div>
                </div>
                </Link>

                <Link to="/datasets?sort_by=total_classes&order=desc" className="stat-card-link">
                <div className="stat-card card glass">
                    <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' }}>
                        <FiLayers />
                    </div>
                    <div className="stat-content">
                        <h3 className="stat-value">{stats?.totalClasses || 0}</h3>
                        <p className="stat-label">Total Classes</p>
                    </div>
                </div>
                </Link>

                <Link to="/datasets?sort_by=created_at&order=desc" className="stat-card-link">
                <div className="stat-card card glass">
                    <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)' }}>
                        <FiTrendingUp />
                    </div>
                    <div className="stat-content">
                        <h3 className="stat-value">{stats?.totalObjects || 0}</h3>
                        <p className="stat-label">Total Objects</p>
                    </div>
                </div>
                </Link>
            </div>

            <div className="stats-grid grid grid-4">
                <div className="stat-card card glass">
                    <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%)' }}>
                        <FiClock />
                    </div>
                    <div className="stat-content">
                        <h3 className="stat-value">{formatDuration(stats?.totalTrainingTimeSeconds || 0)}</h3>
                        <p className="stat-label">Total Training Time</p>
                    </div>
                </div>
                <div className="stat-card card glass">
                    <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #56ccf2 0%, #2f80ed 100%)' }}>
                        <FiImage />
                    </div>
                    <div className="stat-content">
                        <h3 className="stat-value">{stats?.totalTrainingImages || 0}</h3>
                        <p className="stat-label">Total Trained Images</p>
                    </div>
                </div>
            </div>

            <div className="recent-section">
                <div className="section-header">
                    <h2 className="section-title">Recent Datasets</h2>
                    <Link to="/datasets" className="btn btn-secondary">View All</Link>
                </div>

                {recentDatasets.length === 0 ? (
                    <div className="empty-state card">
                        <FiDatabase className="empty-icon" />
                        <h3>No datasets yet</h3>
                        <p className="text-secondary">Upload your first dataset to get started</p>
                        <Link to="/upload" className="btn btn-primary mt-md">Upload Dataset</Link>
                    </div>
                ) : (
                    <div className="datasets-grid grid grid-3">
                        {recentDatasets.map((dataset) => (
                            <Link key={dataset.id} to={`/datasets/${dataset.id}`} className="dataset-card card">
                                <button
                                    type="button"
                                    className="dataset-delete-btn"
                                    onClick={(event) => handleDeleteDataset(event, dataset.id)}
                                    disabled={deletingIds.has(dataset.id)}
                                    aria-label={`Delete dataset ${dataset.id}`}
                                >
                                    <FiTrash2 />
                                </button>
                                <div className="dataset-title-block">
                                    <h3 className="dataset-title">Dataset {dataset.id.slice(0, 8)}</h3>
                                    <p className="dataset-id">{dataset.id}</p>
                                </div>
                                <div className="dataset-header">
                                    <span className="dataset-format">{dataset.format_type.toUpperCase()}</span>
                                    <span className="dataset-date">{new Date(dataset.created_at).toLocaleDateString()}</span>
                                </div>
                                <div className="dataset-stats">
                                    <div className="dataset-stat">
                                        <FiImage />
                                        <span>{dataset.total_images} images</span>
                                    </div>
                                    <div className="dataset-stat">
                                        <FiLayers />
                                        <span>{dataset.total_classes} classes</span>
                                    </div>
                                </div>
                            </Link>
                        ))}
                    </div>
                )}
            </div>

            <div className="quick-actions">
                <h2 className="section-title">Quick Actions</h2>
                <div className="actions-grid grid grid-3">
                    <Link to="/upload" className="action-card card">
                        <FiDatabase className="action-icon" />
                        <h3>Upload Dataset</h3>
                        <p className="text-secondary">Upload and validate a new dataset</p>
                    </Link>

                    <Link to="/datasets" className="action-card card">
                        <FiImage className="action-icon" />
                        <h3>Browse Datasets</h3>
                        <p className="text-secondary">View and manage your datasets</p>
                    </Link>

                    <Link to="/augment" className="action-card card">
                        <FiLayers className="action-icon" />
                        <h3>Augment Data</h3>
                        <p className="text-secondary">Apply augmentations to datasets</p>
                    </Link>
                </div>
            </div>
        </div>
    );
}

export default HomePage;
