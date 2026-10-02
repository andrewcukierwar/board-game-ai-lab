import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Connect4Page from "./pages/Connect4.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<main style={{ padding: "20px" }}><h1>Board Game AI Lab</h1><p>Play Connect 4 against Random or Negamax.</p><Link to="/connect4">Play Connect 4</Link></main>} />
      <Route path="/connect4" element={<Connect4Page />} />
    </Routes>
  </BrowserRouter>
);
