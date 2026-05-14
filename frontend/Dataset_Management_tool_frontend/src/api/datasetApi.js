import axios from 'axios';
import API_BASE_URL from '../config';

const api = axios.create({
    baseURL: API_BASE_URL,
    // Removed default Content-Type to allow per-request override
});

// Dataset API
export const datasetApi = {
    // List all datasets with pagination
    listDatasets: async (params = {}) => {
        const response = await api.get('/datasets', { params });
        return response.data;
    },

    // Get dataset details
    getDataset: async (datasetId) => {
        const response = await api.get(`/datasets/${datasetId}`);
        return response.data;
    },

    // Delete dataset
    deleteDataset: async (datasetId) => {
        const response = await api.delete(`/datasets/${datasetId}`);
        return response.data;
    },

    // Get dataset images
    getDatasetImages: async (datasetId, params = {}) => {
        const response = await api.get(`/datasets/${datasetId}/images`, { params });
        return response.data;
    },

    // List all images across datasets
    listAllImages: async (params = {}) => {
        const response = await api.get('/images', { params });
        return response.data;
    },

    // Get flagged image/label issues for review
    getDatasetImageIssues: async (datasetId, params = {}) => {
        const response = await api.get(`/datasets/${datasetId}/images/issues`, { params });
        return response.data;
    },

    // Get dataset statistics
    getDatasetStatistics: async (datasetId) => {
        const response = await api.get(`/datasets/${datasetId}/statistics`);
        return response.data;
    },

    // Upload dataset
    uploadDataset: async (formData, onProgress) => {
        const response = await api.post('/upload-dataset', formData, {
            onUploadProgress: (progressEvent) => {
                if (onProgress) {
                    const percentCompleted = Math.round(
                        (progressEvent.loaded * 100) / progressEvent.total
                    );
                    onProgress(percentCompleted);
                }
            },
        });
        return response.data;
    },

    // Apply augmentation
    augmentDataset: async (augmentationData) => {
        const response = await api.post('/augment', augmentationData);
        return response.data;
    },

    // Get download URL
    getDownloadUrl: async (datasetId) => {
        const response = await api.get(`/download/${datasetId}`);
        return response.data;
    },

    // Update labels for a single image
    updateImageLabels: async (datasetId, imageId, labels) => {
        const response = await api.put(`/datasets/${datasetId}/images/${imageId}/labels`, labels);
        return response.data;
    },

    // Get available compute devices (GPU/CPU)
    getAvailableDevices: async () => {
        const response = await api.get('/train/devices');
        return response.data;
    },

    // Start a training job
    startTraining: async (payload) => {
        const response = await api.post('/train/start', payload, {
            headers: {
                'Content-Type': 'application/json',
            },
        });
        return response.data;
    },

    // List all training jobs
    listTrainingJobs: async () => {
        const response = await api.get('/train/jobs');
        return response.data;
    },

    // Get single training job details
    getTrainingJob: async (jobId) => {
        const response = await api.get(`/train/jobs/${jobId}`);
        return response.data;
    },

    // Stop a running or queued training job
    stopTraining: async (jobId) => {
        const response = await api.post(`/train/jobs/${jobId}/stop`);
        return response.data;
    },

    // Delete a training job from the dashboard
    deleteTrainingJob: async (jobId) => {
        const response = await api.delete(`/train/jobs/${jobId}`);
        return response.data;
    },

    // Run inference using trained model for a dataset
    // supply either imageUrl or file
    predict: async ({ datasetId, imageUrl, file }) => {
        const form = new FormData();
        form.append('dataset_id', datasetId);
        if (imageUrl) {
            form.append('image_url', imageUrl);
        }
        if (file) {
            form.append('image_file', file);
        }
        const response = await api.post('/train/predict', form, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },
};

export default api;
