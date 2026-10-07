import { Link } from 'react-router-dom';

export const REPOSITORY_URL = 'https://github.com/andrewcukierwar/board-game-ai-lab';

export function ActionLink({ to, href, variant = 'primary', children }) {
  const className = `action-link action-link--${variant}`;
  const content = <>{children}<span aria-hidden="true">↗</span></>;
  return href ? <a href={href} className={className}>{content}</a>
    : <Link to={to} className={className}>{content}</Link>;
}

export function Badge({ children, variant = 'available' }) {
  return <span className={`badge badge--${variant}`}>{children}</span>;
}

export function SectionHeading({ number, eyebrow, title, children }) {
  return (
    <div className="section-heading">
      <p className="eyebrow"><span className="section-number">{number}</span>{eyebrow}</p>
      <h2>{title}</h2>
      {children && <p className="section-intro">{children}</p>}
    </div>
  );
}
