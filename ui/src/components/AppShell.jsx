import { useEffect, useRef } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { REPOSITORY_URL } from './primitives.jsx';

export default function AppShell() {
  const { pathname, hash, key } = useLocation();
  const previousPath = useRef(pathname);

  useEffect(() => {
    const routeChanged = previousPath.current !== pathname;
    previousPath.current = pathname;
    const frame = requestAnimationFrame(() => {
      if (hash) {
        const section = document.getElementById(hash.slice(1));
        section?.scrollIntoView();
        section?.focus({ preventScroll: true });
      } else {
        window.scrollTo(0, 0);
        if (routeChanged) document.getElementById('main-content')?.focus({ preventScroll: true });
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [pathname, hash, key]);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <header className="site-header">
        <div className="site-container header-inner">
          <Link className="brand" to="/" aria-label="Board Game AI Lab home">
            <span className="brand-mark" aria-hidden="true"><i /><i /><i /><i /></span>
            <span>Board Game <strong>AI Lab</strong></span>
          </Link>
          <nav className="site-nav" aria-label="Main navigation">
            <NavLink end to="/connect4">Play</NavLink>
            <NavLink to="/connect4/match-lab">Match Lab</NavLink>
            <Link to="/#agents">Agents</Link>
            <Link to="/#research">Research</Link>
            <a href={REPOSITORY_URL}>GitHub <span aria-hidden="true">↗</span></a>
          </nav>
        </div>
      </header>
      <div id="main-content" className="site-content" tabIndex={-1}>
        <Outlet />
      </div>
      <footer className="site-footer">
        <div className="site-container footer-inner">
          <div><Link className="footer-brand" to="/">Board Game AI Lab</Link><p>An interactive laboratory for game-playing AI.</p></div>
          <div className="footer-links"><Link to="/connect4">Open the game →</Link><a href={REPOSITORY_URL}>Open GitHub repository ↗</a></div>
        </div>
      </footer>
    </div>
  );
}
