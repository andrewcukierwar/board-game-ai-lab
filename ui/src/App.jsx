import { Route, Routes } from 'react-router-dom';
import AppShell from './components/AppShell.jsx';
import Home from './pages/Home.jsx';
import Connect4Page from './pages/Connect4.jsx';

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<Home />} />
        <Route path="/connect4" element={<Connect4Page />} />
      </Route>
    </Routes>
  );
}
