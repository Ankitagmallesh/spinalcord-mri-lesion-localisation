import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import ArchitecturePage from "./pages/ArchitecturePage";
import TrainingCurvesPage from "./pages/TrainingCurvesPage";
import EvaluationPage from "./pages/EvaluationPage";
import HistoryPage from "./pages/HistoryPage";
import { Toaster } from "sonner";

function App() {
    return (
        <div className="App">
            <BrowserRouter>
                <Routes>
                    <Route path="/" element={<Dashboard />} />
                    <Route path="/architecture" element={<ArchitecturePage />} />
                    <Route path="/training" element={<TrainingCurvesPage />} />
                    <Route path="/metrics" element={<EvaluationPage />} />
                    <Route path="/history" element={<HistoryPage />} />
                </Routes>
            </BrowserRouter>
            <Toaster position="bottom-right" richColors />
        </div>
    );
}

export default App;

