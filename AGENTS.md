# Fusion Starter

A production-ready full-stack React application template with integrated Express server, featuring React Router 6 SPA mode, TypeScript, Vitest, Zod and modern tooling.

While the starter comes with a express server, only create endpoint when strictly neccesary, for example to encapsulate logic that must leave in the server, such as private keys handling, or certain DB operations, db...

## Tech Stack

- **Frontend**: React 18 + React Router 6 (spa) + TypeScript + Vite + TailwindCSS 3
- **Backend**: Express server integrated with Vite dev server
- **Testing**: Vitest
- **UI**: Radix UI + TailwindCSS 3 + Lucide React icons

## Project Structure

```
client/                   # React SPA frontend
├── pages/                # Route components (Index.tsx = home)
├── components/ui/        # Pre-built UI component library
├── App.tsx                # App entry point and with SPA routing setup
└── global.css            # TailwindCSS 3 theming and global styles

server/                   # Express API backend
├── index.ts              # Main server setup (express config + routes)
└── routes/               # API handlers

shared/                   # Types used by both client & server
└── api.ts                # Example of how to share api interfaces
```

## Key Features

## SPA Routing System

The routing system is powered by React Router 6:

- `client/pages/Index.tsx` represents the home page.
- Routes are defined in `client/App.tsx` using the `react-router-dom` import
- Route files are located in the `client/pages/` directory

For example, routes can be defined with:

```typescript
import { BrowserRouter, Routes, Route } from "react-router-dom";

<Routes>
  <Route path="/" element={<Index />} />
  {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
  <Route path="*" element={<NotFound />} />
</Routes>;
```

### Styling System

- **Primary**: TailwindCSS 3 utility classes
- **Theme and design tokens**: Configure in `client/global.css` 
- **UI components**: Pre-built library in `client/components/ui/`
- **Utility**: `cn()` function combines `clsx` + `tailwind-merge` for conditional classes

```typescript
// cn utility usage
className={cn(
  "base-classes",
  { "conditional-class": condition },
  props.className  // User overrides
)}
```

### Express Server Integration

- **Development**: Single port (8080) for both frontend/backend
- **Hot reload**: Both client and server code
- **API endpoints**: Prefixed with `/api/`

#### Example API Routes
- `GET /api/ping` - Simple ping api
- `GET /api/demo` - Demo endpoint  

### Shared Types
Import consistent types in both client and server:
```typescript
import { DemoResponse } from '@shared/api';
```

Path aliases:
- `@shared/*` - Shared folder
- `@/*` - Client folder

## Development Commands

```bash
npm run dev        # Start dev server (client + server)
npm run build      # Production build
npm run start      # Start production server
npm run typecheck  # TypeScript validation
npm test          # Run Vitest tests
```

## Running the Application

This project consists of two main components: a React frontend and a Python FastAPI backend. Here's how to run both:

### Prerequisites

1. **Node.js and npm** (for frontend)
2. **Python 3.8+** (for backend)
3. **Git** (for cloning the repository)

### Frontend Setup

The frontend is a React application built with Vite:

```bash
# Install dependencies (if not already done)
npm install

# Start the development server
npm run dev
```

The frontend will be available at:
- **Local**: http://localhost:3000
- **Network**: http://192.168.31.251:3000 (accessible from other devices on your network)

### Backend Setup

The backend is a Python FastAPI application with machine learning capabilities:

```bash
# Navigate to the backend directory
cd stroke-lab-backend

# Activate the virtual environment
source venv/bin/activate

# Install dependencies (if not already done)
pip install -r requirements.txt

# Start the FastAPI server
python -m app.main
```

The backend will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Running Both Services

To run both frontend and backend simultaneously:

1. **Terminal 1** (Frontend):
   ```bash
   npm run dev
   ```

2. **Terminal 2** (Backend):
   ```bash
   cd stroke-lab-backend
   source venv/bin/activate
   python -m app.main
   ```

### Backend Features

The Python backend provides:

- **Image-to-Stroke Conversion**: Convert handwritten images to time-series stroke data
- **Mathematical Analysis**: Advanced mathematical invariant analysis including:
  - Affine differential geometry
  - Topological data analysis (TDA)
  - Path signature calculations
  - Spectral analysis with wavelets
  - Persistent homology
- **Machine Learning Training**: Train models on stroke invariant features
- **Real-time Analytics**: Real-time stroke analysis and feature extraction
- **Model Export**: Export trained models for deployment

### Key API Endpoints

- `GET /` - API information and status
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)
- `POST /api/dataset/import` - Import datasets for training
- `POST /api/train` - Start machine learning training
- `GET /api/train/{session_id}/status` - Monitor training progress
- `GET /api/models` - List trained models
- `GET /api/models/{model_id}/export` - Export trained models

### Database

The backend uses SQLite for data storage with the following tables:
- `stroke_data` - Processed stroke information
- `trained_models` - Trained machine learning models
- `training_sessions` - Training session metadata
- `dataset_imports` - Imported dataset information

### Troubleshooting

1. **Port Conflicts**: If ports 3000 or 8000 are in use, the services will automatically try alternative ports
2. **Python Dependencies**: Ensure all requirements are installed: `pip install -r requirements.txt`
3. **Virtual Environment**: Always activate the virtual environment before running the backend
4. **Database**: The SQLite database is automatically created on first run
5. **CORS**: The backend is configured to accept requests from the frontend development server

### Development Workflow

1. Start both services using the commands above
2. Open http://localhost:3000 in your browser for the frontend
3. Open http://localhost:8000/docs for API documentation
4. Make changes to frontend code - Vite will hot-reload automatically
5. Make changes to backend code - Uvicorn will restart automatically
6. Test API endpoints using the interactive documentation

## Adding Features

### Add new colors to the theme

Open `client/global.css` and `tailwind.config.ts` and add new tailwind colors.

### New API Route
1. **Optional**: Create a shared interface in `shared/api.ts`:
```typescript
export interface MyRouteResponse {
  message: string;
  // Add other response properties here
}
```

2. Create a new route handler in `server/routes/my-route.ts`:
```typescript
import { RequestHandler } from "express";
import { MyRouteResponse } from "@shared/api"; // Optional: for type safety

export const handleMyRoute: RequestHandler = (req, res) => {
  const response: MyRouteResponse = {
    message: 'Hello from my endpoint!'
  };
  res.json(response);
};
```

3. Register the route in `server/index.ts`:
```typescript
import { handleMyRoute } from "./routes/my-route";

// Add to the createServer function:
app.get("/api/my-endpoint", handleMyRoute);
```

4. Use in React components with type safety:
```typescript
import { MyRouteResponse } from '@shared/api'; // Optional: for type safety

const response = await fetch('/api/my-endpoint');
const data: MyRouteResponse = await response.json();
```

### New Page Route
1. Create component in `client/pages/MyPage.tsx`
2. Add route in `client/App.tsx`:
```typescript
<Route path="/my-page" element={<MyPage />} />
```

## Production Deployment

- **Standard**: `npm run build` + `npm start`
- **Binary**: Self-contained executables (Linux, macOS, Windows)
- **Cloud Deployment**: Use either Netlify or Vercel via their MCP integrations for easy deployment. Both providers work well with this starter template.

## Architecture Notes

- Single-port development with Vite + Express integration
- TypeScript throughout (client, server, shared)
- Full hot reload for rapid development
- Production-ready with multiple deployment options
- Comprehensive UI component library included
- Type-safe API communication via shared interfaces
