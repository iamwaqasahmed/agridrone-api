# Agridrone Frontend

A modern Next.js frontend for the Agridrone crop monitoring and NDVI analysis API.

## Features

- **Crop Monitoring Dashboard**: Real-time crop health monitoring with NDVI analysis
- **Image Upload & Analysis**: Upload satellite images for crop analysis
- **Weather Integration**: Real-time weather data for agricultural insights
- **Field Management**: Multiple field locations with detailed analytics
- **Responsive Design**: Modern UI with Tailwind CSS and Framer Motion

## Tech Stack

- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework
- **Framer Motion**: Smooth animations and transitions
- **Lucide React**: Beautiful icons
- **Axios**: HTTP client for API calls

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Navigate to the frontend directory:
```bash
cd agridrone_api/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Building for Production

```bash
npm run build
npm start
```

## API Integration

The frontend is designed to work with the Agridrone API running on port 8001. Make sure the API is running before testing the frontend.

### Environment Variables

Create a `.env.local` file in the frontend directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8001
```

## Project Structure

```
frontend/
├── app/
│   ├── globals.css          # Global styles with Tailwind
│   ├── layout.tsx           # Root layout component
│   └── page.tsx             # Main dashboard page
├── public/                  # Static assets
├── package.json             # Dependencies and scripts
├── tailwind.config.js       # Tailwind configuration
├── next.config.js           # Next.js configuration
└── tsconfig.json           # TypeScript configuration
```

## Features Overview

### Dashboard
- Field overview with location selection
- Weather conditions display
- Crop health indicators
- Real-time monitoring stats

### Image Analysis
- Upload satellite images
- NDVI score calculation
- Crop type identification
- Health status assessment
- Agricultural recommendations

### Monitoring
- Real-time camera feeds
- Live monitoring controls
- Alert management
- Performance metrics

## Customization

### Colors
The theme uses green colors for agricultural focus. Modify `tailwind.config.js` to change the color scheme.

### API Endpoints
Update the API calls in `page.tsx` to match your backend endpoints.

## Deployment

### Docker
```bash
docker build -t agridrone-frontend .
docker run -p 3000:3000 agridrone-frontend
```

### Vercel
1. Connect your repository to Vercel
2. Set environment variables
3. Deploy automatically

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the AI APIs portfolio. 