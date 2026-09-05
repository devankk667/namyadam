import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { api } from './services/api';
import RootLayout from './layouts/RootLayout';
import DashboardPage from './pages/DashboardPage';
import DetectionDetailPage from './pages/DetectionDetailPage';
import AnalyticsPage from './pages/AnalyticsPage';
import AlertsPage from './pages/AlertsPage';
import ModelPage from './pages/ModelPage';
import PredictionDemoPage from './pages/PredictionDemoPage';

export default function App() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await api.getHealth();
        setHealth(res);
      } catch (err) {
        console.error("Health check error:", err);
      }
    }
    checkHealth();
  }, []);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<RootLayout health={health} />}>
          <Route index element={<DashboardPage />} />
          <Route path="detections/:id" element={<DetectionDetailPage />} />
          <Route path="analytics" element={<AnalyticsPage />} />
          <Route path="alerts" element={<AlertsPage />} />
          <Route path="model" element={<ModelPage />} />
          <Route path="predict" element={<PredictionDemoPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
