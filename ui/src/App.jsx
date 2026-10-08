import { Route, Routes } from 'react-router-dom';
import AppShell from './components/AppShell.jsx';
import Home from './pages/Home.jsx';
import MatchLabPage from './pages/MatchLab.jsx';
import SeasonLabPage from './pages/SeasonLab.jsx';
import TournamentLabPage from './pages/TournamentLab.jsx';
import Connect4Page from './pages/Connect4.jsx';

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<Home />} />
        <Route path="/connect4/match-lab" element={<MatchLabPage />} />
        <Route path="/connect4/season" element={<SeasonLabPage />} />
        <Route path="/connect4/tournament" element={<TournamentLabPage />} />
        <Route path="/connect4" element={<Connect4Page />} />
      </Route>
    </Routes>
  );
}
