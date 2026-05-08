import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout/Layout';
import HomePage from './pages/HomePage';
import DatasetsPage from './pages/DatasetsPage';
import DatasetDetailsPage from './pages/DatasetDetailsPage';
import UploadPage from './pages/UploadPage';
import AugmentationPage from './pages/AugmentationPage';
import ModelPreviewPage from './pages/ModelPreviewPage';
import ImagesPage from './pages/ImagesPage';
import TrainingPage from './pages/TrainingPage';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/datasets/:id" element={<DatasetDetailsPage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/augment" element={<AugmentationPage />} />
          <Route path="/train" element={<TrainingPage />} />
          <Route path="/models" element={<ModelPreviewPage />} />
          <Route path="/images" element={<ImagesPage />} />
          <Route path="*" element={<HomePage />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
