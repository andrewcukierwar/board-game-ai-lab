import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Connect4Page from "./pages/Connect4.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<h2>Home</h2>} />
      <Route path="/connect4" element={<Connect4Page />} />
    </Routes>
  </BrowserRouter>
);
