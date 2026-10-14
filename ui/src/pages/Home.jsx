import axios from 'axios';
import { Link } from 'react-router-dom';
import { researchAgentEnabled, aiAgentTypes, VICTOR_RESEARCH } from '../connect4/researchAgent.js';
import { ALPHAZERO } from '../connect4/alphazero.js';
import { useAlphaZero } from '../connect4/useAlphaZero.js';
import { NEGAMAX_DEPTHS, MCTS_SIMULATIONS, TACTICAL_MCTS_SIMULATIONS } from '../connect4/competitorConfig.js';
import BoardIllustration from '../components/BoardIllustration.jsx';
import { ActionLink, Badge, REPOSITORY_URL, SectionHeading } from '../components/primitives.jsx';
import './home.css';

const homeHttp = axios.create({ baseURL: import.meta.env?.VITE_API_BASE, timeout: 90000 });
const range = values => `${values[0].toLocaleString('en-US')}–${values.at(-1).toLocaleString('en-US')}`;

const agentCards = {
  random: { name: 'Random', label: 'The baseline', glyph: 'random', title: 'Choose a legal move.',
    description: 'Uniformly selects from the legal columns. A simple baseline for comparing more structured approaches.',
    detail: 'Uniform sampling · no lookahead' },
  negamax: { name: 'Negamax', label: 'Adversarial tree search', glyph: 'tree', title: 'Look ahead. Evaluate replies.',
    description: 'Explores moves and opposing replies to a limited depth, then uses a heuristic to evaluate the resulting positions.',
    detail: `Depth presets ${NEGAMAX_DEPTHS.join(' / ')}` },
  mcts: { name: 'Classic MCTS', label: 'Monte Carlo Tree Search', glyph: 'mcts', title: 'Simulate. Build evidence.',
    description: 'Uses UCT tree search and random rollouts to estimate which moves win most often within a fixed simulation budget.',
    detail: `${range(MCTS_SIMULATIONS)} simulation presets` },
  mcts_tactical: { name: 'Tactical MCTS', label: 'Solver-enhanced search', glyph: 'tactical', title: 'Simulate, then prove.',
    description: 'Adds tactical rollouts and proven win/loss propagation to MCTS, and can stop early once a position is solved.',
    detail: `${range(TACTICAL_MCTS_SIMULATIONS)} max-simulation presets` },
  [ALPHAZERO]: { name: 'AlphaZero', label: 'Self-play neural network', glyph: 'neural', title: 'Learn, then search.',
    description: 'A policy/value network trained through self-play guides Monte Carlo Tree Search. Experimental; not a perfect solver.',
    detail: 'PyTorch CNN · 256 PUCT simulations' },
  [VICTOR_RESEARCH]: { name: 'Victor Research', label: 'Strategic + exact search', glyph: 'victor', title: 'Combine strategy with proof.',
    description: 'Hybrid strategic and exact search inspired by Victor Allis’s Connect 4 research. Experimental, publicly playable, and capable of losing.',
    detail: 'Bounded proof search · heuristic fallback' },
};

const alphaZeroSteps = [
  ['Self-play', 'The agent plays complete games against itself. Each position is saved with the moves its search preferred and the eventual result.'],
  ['Learning', 'A convolutional network with 325,896 parameters, trained in PyTorch, learns a policy (which columns look promising) and a value (who is likely to win).'],
  ['Search', 'Each move runs 256 PUCT simulations: the network’s predictions steer Monte Carlo Tree Search toward promising lines instead of random rollouts.'],
  ['Optional safeguard', 'A conventional one-ply check after search takes an immediate win and avoids handing one over. Measured effect across two trained networks: +1.30 points of match score, below the +2‑point threshold declared before the study.'],
];

function ApproachGlyph({ type }) {
  return (
    <svg className={`approach-glyph approach-glyph--${type}`} viewBox="0 0 180 60" aria-hidden="true">
      {type === 'victor' ? <>
        <path d="M32 16 L72 16 L108 44 L148 44 M32 44 L72 44 L108 16 L148 16" />
        {[32, 72, 108, 148].map((x, i) => <g key={x}><circle cx={x} cy={i < 2 ? 16 : 44} r="5" /><circle cx={x} cy={i < 2 ? 44 : 16} r="5" /></g>)}
        <path d="M83 29 L89 35 L100 23" />
      </> : type === 'tree' ? <>
        <path d="M90 10 L45 32 M90 10 L135 32 M45 32 L23 52 M45 32 L67 52 M135 32 L113 52 M135 32 L157 52" />
        {[[90, 10], [45, 32], [135, 32], [23, 52], [67, 52], [113, 52], [157, 52]].map(([cx, cy]) => <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r="4" />)}
      </> : type === 'neural' ? <>
        <path d={[...[12, 30, 48].flatMap(a => [8, 23, 38, 53].map(b => `M24 ${a} L90 ${b}`)), ...[8, 23, 38, 53].flatMap(a => [18, 42].map(b => `M90 ${a} L156 ${b}`))].join(' ')} />
        {[[24, 12], [24, 30], [24, 48], [90, 8], [90, 23], [90, 38], [90, 53], [156, 18], [156, 42]].map(([cx, cy]) => <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r="4.5" />)}
      </> : type === 'tactical' ? <>
        {[0, 1, 2, 3, 4].map(i => <rect key={i} x={i * 24 + 8} y={[34, 20, 7, 24, 38][i]} width="12" height={[15, 29, 42, 25, 11][i]} rx="3" />)}
        <circle cx="150" cy="30" r="17" /><path d="M141 30 L147 36 L159 23" />
      </> : [0, 1, 2, 3, 4, 5, 6].map(i => <rect key={i} x={i * 24 + 8} y={type === 'random' ? 25 : [40, 30, 17, 7, 22, 34, 42][i]} width="12" height={type === 'random' ? 24 : [9, 19, 32, 42, 27, 15, 7][i]} rx="3" />)}
    </svg>
  );
}

function alphaZeroBadge(alphaZero) {
  if (alphaZero.available) return <Badge>Playable · Experimental</Badge>;
  return <Badge variant="research">{alphaZero.checking ? 'Checking availability…' : 'Not available on this server'}</Badge>;
}

export default function Home({ http = homeHttp }) {
  const enabled = researchAgentEnabled();
  const alphaZero = useAlphaZero(http);
  // AlphaZero is always described; only backend discovery may call it playable.
  const agents = aiAgentTypes(enabled, true).map(type => ({ type, ...agentCards[type] }));
  const playable = aiAgentTypes(enabled, alphaZero.available).map(type => agentCards[type].name);
  return (
    <main className="home-page">
      <section className="hero site-container" aria-labelledby="hero-title">
        <div className="hero-copy">
          <p className="eyebrow"><span className="accent-dash" />Play · compare · understand</p>
          <h1 id="hero-title">Board Game<br /><span>AI Lab</span><span className="title-period">.</span></h1>
          <p className="hero-statement">Explore how game-playing AI chooses moves.</p>
          <p className="hero-description">Play Connect 4 against classical search and Monte Carlo agents{alphaZero.available ? ', and a neural network trained through self-play' : ''}. Compare how they choose moves and inspect grounded analysis of the board.</p>
          <div className="action-row"><ActionLink to="/connect4">Play Connect 4</ActionLink><ActionLink to="/#agents" variant="secondary">Explore the agents</ActionLink></div>
          <p className="hero-note"><span className="piece-dot piece-dot--red" /><span className="piece-dot piece-dot--yellow" />Choose whether you or the AI moves first.</p>
        </div>
        <BoardIllustration />
      </section>

      <div className="lab-strip"><div className="site-container lab-strip-inner"><span>An interactive laboratory</span><span>Search, simulation &amp; neural agents</span><span>Grounded position analysis</span><span>Open-source research</span></div></div>

      <section id="agents" className="home-section site-container" aria-label="Public agents" tabIndex={-1}>
        <SectionHeading number="01" eyebrow="The public agents" title="Same game. Different approaches.">From a random baseline to adversarial search, Monte Carlo simulation and a self-play neural network: compare how each approach selects a move.</SectionHeading>
        <div className="agent-grid">
          {agents.map(agent => <article key={agent.type} className="lab-card agent-card">
            <div className="card-topline"><p className="eyebrow">{agent.label}</p>{agent.type === ALPHAZERO ? alphaZeroBadge(alphaZero) : <Badge>{agent.type === VICTOR_RESEARCH ? 'Playable · Experimental' : 'Playable'}</Badge>}</div>
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
        <div><p className="eyebrow">Tournament Lab</p><h2>Build a field. Crown a champion.</h2><p>Single-elimination bracket competition. Enter yourself or watch the AI field compete.</p></div>
        <ActionLink to="/connect4/tournament" variant="secondary">Open Tournament Lab</ActionLink>
      </section>

      <section className="match-lab-discovery site-container" aria-label="Season Lab">
        <div><p className="eyebrow">Season Lab</p><h2>Compare a field over a full season.</h2><p>Balanced repeated matchups, standings, and ratings for comparative agent evaluation.</p></div>
        <ActionLink to="/connect4/season" variant="secondary">Open Season Lab</ActionLink>
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
        <div className="research-heading"><SectionHeading number="03" eyebrow="The research track" title="From strategy to search and self-play.">A broader exploration of how game-playing agents can learn, evaluate positions, and improve through experience.</SectionHeading><a className="text-link" href={REPOSITORY_URL}>Explore the repository <span aria-hidden="true">↗</span></a></div>
        <div className="research-track">
          <article><span className="track-index">01</span><h3>Classical search</h3><p>Baselines and depth-limited adversarial search.</p><Badge>Playable now</Badge></article>
          <article><span className="track-index">02</span><h3>Monte Carlo search</h3><p>Rollout-based UCT and solver-enhanced Tactical MCTS.</p><Badge>Playable now</Badge></article>
          <article><span className="track-index">03</span><h3>DQN experiments</h3><p>Learning action values through reinforcement learning.</p><Badge variant="research">Research only</Badge></article>
          <article><span className="track-index">04</span><h3>Neural MCTS &amp; AlphaZero</h3><p>Neural evaluation combined with tree search, trained through self-play.</p>{alphaZero.available ? <Badge>Playable · Experimental</Badge> : <Badge variant="research">Experimental</Badge>}</article>
        </div>

        <article className="alphazero-showcase lab-card" aria-labelledby="alphazero-title">
          <div className="alphazero-summary">
            <div className="card-topline"><p className="eyebrow">Self-play reinforcement learning</p>{alphaZeroBadge(alphaZero)}</div>
            <h3 id="alphazero-title">AlphaZero</h3>
            <p className="alphazero-lede">Train a neural network to evaluate positions, then use Monte Carlo Tree Search to turn those predictions into decisions.</p>
            <p className="alphazero-stack">PyTorch · Policy/value CNN · PUCT search · Self-play</p>
            <p>Seeded experiments compare the trained model against conventional search agents. It is not a perfect solver: in a controlled study from varied openings, the public model scored about 54% against depth-3 Negamax and about 36% against depth-5 Negamax. Every AlphaZero game export records the exact model and search settings.</p>
            <div className="action-row">
              {alphaZero.available && <Link className="text-link" to="/connect4">Play against AlphaZero <span aria-hidden="true">→</span></Link>}
              <a className="text-link" href={`${REPOSITORY_URL}/tree/main/games/connect4/alphazero_v2`}>AlphaZero implementation ↗</a>
              <a className="text-link" href={`${REPOSITORY_URL}/blob/main/docs/phase4d4k-a-alphazero-backend.md`}>Evaluation results ↗</a>
            </div>
          </div>
          <ol className="alphazero-steps" aria-label="How AlphaZero works">
            {alphaZeroSteps.map(([title, text], i) => <li key={title}><span className="analysis-number">{String(i + 1).padStart(2, '0')}</span><div><h4>{title}</h4><p>{text}</p></div></li>)}
          </ol>
        </article>

        <article className="victor-research-entry lab-card">
          <p className="eyebrow">Strategic reasoning + exact proofs</p><h3>Victor Research</h3>
          <p>Victor Allis’s original Connect 4 framework reasons about threats and how pairs of squares can secure them. This implementation includes all nine Allis rules, conditional strategic reasoning, and verified guarantees for narrower cases. It does not establish a complete executable non-loss theorem for arbitrary combinations of those rules.</p>
          <p>An exact opening book covers 1,722 selected positions, rather than every early position. Native bounded optimal-move proof search can establish exact choices when it finishes; heuristic fallbacks choose moves when proof is incomplete. The whole agent is not perfect play and can lose.</p>
          <p>Independent exact-oracle benchmarks check choices and proof bounds. Reported accuracy measures finite sampled distributions, not exhaustive Connect 4 coverage or a universal strength guarantee.</p>
          <div className="action-row"><a className="text-link" href={`${REPOSITORY_URL}/blob/main/docs/victor-nine-rule-implementation.md`}>Nine-rule framework ↗</a><a className="text-link" href={`${REPOSITORY_URL}/blob/main/docs/victor-astra-second-pass.md`}>Proof search and oracle audit ↗</a><a className="text-link" href={`${REPOSITORY_URL}/blob/main/docs/victor-opening-and-app-readiness.md`}>Opening book coverage ↗</a></div>
          <Badge variant="research">{enabled ? 'Experimental · publicly playable' : 'Experimental · public play disabled'}</Badge>
        </article>
        <p className="research-boundary"><strong>Playable now:</strong> {playable.join(', ')}. {!alphaZero.available && 'AlphaZero appears in game setup once the connected game server reports it available. '}<strong>Research only:</strong> DQN experiments and earlier neural MCTS training runs are not available in the public game.</p>
      </section>

      <section className="final-cta site-container" aria-labelledby="cta-title"><div><p className="eyebrow">Your next move</p><h2 id="cta-title">Put an agent to the test.</h2><p>Pick an approach. Play a game. Explore the position.</p></div><div className="action-row"><ActionLink to="/connect4">Start a game</ActionLink><ActionLink href={REPOSITORY_URL} variant="secondary">View the source</ActionLink></div></section>
    </main>
  );
}
