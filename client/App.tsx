import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "next-themes";
import StrokeAnalysisPage from "./pages/StrokeAnalysisPage";
import "./global.css";
import "./store/analysisProcessor"; // Initialize analysis processor

function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <BrowserRouter>
        <div className="min-h-screen bg-background text-foreground">
          <Routes>
            <Route path="/" element={<StrokeAnalysisPage />} />
            <Route path="*" element={<div className="flex items-center justify-center h-screen"><div className="text-center"><h1 className="text-2xl font-bold mb-4">Page Not Found</h1><p className="text-muted-foreground">Continue prompting to fill in additional page contents.</p></div></div>} />
          </Routes>
        </div>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
