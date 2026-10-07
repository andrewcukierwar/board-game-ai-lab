import BoardIllustration from '../components/BoardIllustration.jsx';
import { ActionLink, Badge, REPOSITORY_URL, SectionHeading } from '../components/primitives.jsx';
import './home.css';

const agents = [
  { name: 'Random', label: 'The baseline', glyph: 'random', title: 'Choose a legal move.',
    description: 'Uniformly selects from the legal columns. A simple baseline for comparing more structured approaches.',
    detail: 'Uniform sampling · no lookahead' },
  { name: 'Negamax', label: 'Adversarial tree search', glyph: 'tree', title: 'Look ahead. Evaluate replies.',
    description: 'Explores moves and opposing replies to a limited depth, then uses a heuristic to evaluate the resulting positions.',
    detail: 'Depth-limited · heuristic evaluation' },
  { name: 'MCTS', label: 'Monte Carlo Tree Search', glyph: 'mcts', title: 'Simulate. Build evidence.',
    description: 'Uses UCT-style search and simulated rollouts to explore possible continuations within a bounded simulation budget.',
    detail: '100 / 400 / 800 simulation presets' },
];

function ApproachGlyph({ type }) {
  return (
    <svg className={`approach-glyph approach-glyph--${type}`} viewBox="0 0 180 60" aria-hidden="true">
      {type === 'tree' ? <>
        <path d="M90 10 L45 32 M90 10 L135 32 M45 32 L23 52 M45 32 L67 52 M135 32 L113 52 M135 32 L157 52" />
        {[[90, 10], [45, 32], [135, 32], [23, 52], [67, 52], [113, 52], [157, 52]].map(([cx, cy]) => <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r="4" />)}
      </> : [0, 1, 2, 3, 4, 5, 6].map(i => <rect key={i} x={i * 24 + 8} y={type === 'random' ? 25 : [40, 30, 17, 7, 22, 34, 42][i]} width="12" height={type === 'random' ? 24 : [9, 19, 32, 42, 27, 15, 7][i]} rx="3" />)}
    </svg>
  );
}

export default function Home() {
  return (
    <main className="home-page">
      <section className="hero site-container" aria-labelledby="hero-title">
        <div className="hero-copy">
          <p className="eyebrow"><span className="accent-dash" />Play · compare · understand</p>
          <h1 id="hero-title">Board Game<br /><span>AI Lab</span><span className="title-period">.</span></h1>
          <p className="hero-statement">Explore how game-playing AI chooses moves.</p>
          <p className="hero-description">Play Connect 4 against different AI approaches. Compare how they choose moves and inspect grounded analysis of the board.</p>
          <div className="action-row"><ActionLink to="/connect4">Play Connect 4</ActionLink><ActionLink to="/#agents" variant="secondary">Explore the agents</ActionLink></div>
          <p className="hero-note"><span className="piece-dot piece-dot--red" /><span className="piece-dot piece-dot--yellow" />Choose whether you or the AI moves first.</p>
        </div>
        <BoardIllustration />
      </section>

      <div className="lab-strip"><div className="site-container lab-strip-inner"><span>An interactive laboratory</span><span>3 playable agents</span><span>Grounded position analysis</span><span>Open-source research</span></div></div>

      <section id="agents" className="home-section site-container" aria-label="Public agents" tabIndex={-1}>
        <SectionHeading number="01" eyebrow="The public agents" title="Same game. Different approaches.">From a simple baseline to tree search and simulated rollouts, explore three ways to select a move.</SectionHeading>
        <div className="agent-grid">
          {agents.map(agent => <article key={agent.name} className="lab-card agent-card">
            <div className="card-topline"><p className="eyebrow">{agent.label}</p><Badge>Playable</Badge></div>
            <h3>{agent.name}</h3><ApproachGlyph type={agent.glyph} />
            <h4>{agent.title}</h4><p>{agent.description}</p><div className="card-detail">{agent.detail}</div>
          </article>)}
        </div>
      </section>

      <section className="match-lab-discovery site-container" aria-label="Match Lab">
        <div><p className="eyebrow">Competition Lab</p><h2>Watch strategies collide.</h2><p>Configure any two competitors and compare their play move by move.</p></div>
        <ActionLink to="/connect4/match-lab" variant="secondary">Open Match Lab</ActionLink>
      </section>

      <section className="match-lab-discovery site-container" aria-label="Tournament Lab">
        <div><p className="eyebrow">Tournament Lab</p><h2>Build a field. Crown a champion.</h2><p>Seed an AI bracket, run matchups sequentially, and replay every game.</p></div>
        <ActionLink to="/connect4/tournament" variant="secondary">Open Tournament Lab</ActionLink>
      </section>

      <section className="explain-section" aria-label="Explainability">
        <div className="site-container explain-layout">
          <div><SectionHeading number="02" eyebrow="Beyond the move" title="Make the position understandable.">A move is only the beginning. Explore verified board facts and their connection to Connect 4 strategy.</SectionHeading>
            <div className="method-note"><span className="eyebrow">Grounded, post-hoc analysis</span><p>Explanations analyze the board and a move’s consequences. They do not reveal the agent’s private reasoning or intent.</p></div>
          </div>
          <div className="analysis-list">
            {[
              ['01', 'The previous move', 'Examine the tactical consequences of the last move.'],
              ['02', 'The current position', 'Inspect available moves, immediate threats, and verified board facts.'],
              ['03', 'A hypothetical move', 'Ask “what if?” about a legal column without changing the live game.'],
            ].map(([number, title, text]) => <article className="analysis-item" key={number}><span className="analysis-number">{number}</span><div><h3>{title}</h3><p>{text}</p></div></article>)}
          </div>
        </div>
      </section>

      <section id="research" className="home-section site-container research-section" aria-label="Research" tabIndex={-1}>
        <div className="research-heading"><SectionHeading number="03" eyebrow="The research track" title="From search to self-play.">A broader exploration of how game-playing agents can learn, evaluate positions, and improve through experience.</SectionHeading><a className="text-link" href={REPOSITORY_URL}>Explore the repository <span aria-hidden="true">↗</span></a></div>
        <div className="research-track">
          <article><span className="track-index">01</span><h3>Classical search</h3><p>Baselines and depth-limited adversarial search.</p><Badge>Playable now</Badge></article>
          <article><span className="track-index">02</span><h3>Monte Carlo search</h3><p>Tree exploration guided by simulated rollouts.</p><Badge>Playable now</Badge></article>
          <article><span className="track-index">03</span><h3>DQN experiments</h3><p>Learning action values through reinforcement learning.</p><Badge variant="research">Experimental</Badge></article>
          <article><span className="track-index">04</span><h3>Neural MCTS</h3><p>Combining neural evaluation with tree search.</p><Badge variant="research">Experimental</Badge></article>
          <article><span className="track-index">05</span><h3>AlphaZero-style</h3><p>Research into learning through self-play.</p><Badge variant="research">Experimental</Badge></article>
        </div>
        <p className="research-boundary"><strong>Playable now:</strong> Random, Negamax, and MCTS. <strong>Research / experimental:</strong> learned-agent work, including DQN, neural MCTS, and AlphaZero-style self-play, is not available in the public game.</p>
      </section>

      <section className="final-cta site-container" aria-labelledby="cta-title"><div><p className="eyebrow">Your next move</p><h2 id="cta-title">Put an agent to the test.</h2><p>Pick an approach. Play a game. Explore the position.</p></div><div className="action-row"><ActionLink to="/connect4">Start a game</ActionLink><ActionLink href={REPOSITORY_URL} variant="secondary">View the source</ActionLink></div></section>
    </main>
  );
}
