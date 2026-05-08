# Dataset Management Frontend

A modern React frontend for the Dataset Management System with glassmorphism design, charts, and comprehensive dataset visualization.

## Features

- 📊 **Dashboard**: Overview with statistics and recent datasets
- 📁 **Dataset Management**: Browse, filter, and view dataset details
- 📤 **Upload**: Upload and validate datasets (YOLO, COCO, Pascal VOC)
- 🎨 **Augmentation**: Apply transformations to increase dataset size
- 📈 **Visualizations**: Charts for class distribution and statistics
- 🖼️ **Image Gallery**: Browse dataset images with label status
- 🌙 **Dark Mode**: Premium dark theme with glassmorphism effects

## Tech Stack

- **React 18** - UI library
- **Vite** - Build tool and dev server
- **React Router** - Client-side routing
- **Chart.js** - Data visualization
- **Axios** - HTTP client
- **React Icons** - Icon library

## Getting Started

### Prerequisites

- Node.js 16+ and npm
- Backend server running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at `http://localhost:5173`

### Build for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
src/
├── api/              # API service layer
├── components/       # Reusable components
│   ├── Layout/      # Layout and navigation
│   └── Common/      # Common components (Loading, Error)
├── pages/           # Page components
│   ├── HomePage.jsx
│   ├── DatasetsPage.jsx
│   ├── DatasetDetailsPage.jsx
│   ├── UploadPage.jsx
│   └── AugmentationPage.jsx
├── App.jsx          # Main app component
├── main.jsx         # Entry point
├── index.css        # Global styles and design system
└── config.js        # Configuration
```

## API Endpoints Used

- `GET /datasets` - List all datasets
- `GET /datasets/{id}` - Get dataset details
- `GET /datasets/{id}/images` - Get dataset images
- `GET /datasets/{id}/statistics` - Get dataset statistics
- `POST /upload-dataset` - Upload new dataset
- `POST /augment` - Apply augmentations
- `GET /download/{id}` - Get download URL

## Design System

The application uses a comprehensive design system with:

- **Colors**: HSL-based color palette for easy customization
- **Typography**: Inter font family from Google Fonts
- **Spacing**: Consistent spacing scale
- **Components**: Reusable card, button, and form components
- **Effects**: Glassmorphism, gradients, and smooth animations

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

### Environment Variables

The API base URL is configured in `src/config.js`. Update it if your backend runs on a different port.

## License

MIT
