// A decorative, static position; deliberately independent of the gameplay DOM.
const pieces = [
  '.......',
  '.......',
  '...y...',
  '..ry...',
  '.yryr..',
  'ryrry.y',
];

export default function BoardIllustration() {
  return (
    <figure className="board-illustration">
      <div className="diagram-topline"><span className="eyebrow">Connect 4 / search space</span><span className="diagram-dot" /></div>
      <div className="illustration-board" aria-hidden="true">
        {pieces.flatMap((row, r) => [...row].map((piece, c) => (
          <span key={`${r}-${c}`} className={`illustration-slot illustration-slot--${piece === '.' ? 'empty' : piece}`} />
        )))}
      </div>
      <div className="diagram-process" aria-hidden="true"><span>Position</span><span>Search</span><span>Move</span></div>
      <figcaption>One board. Different ways to choose.<span>Illustrative position · no live game</span></figcaption>
    </figure>
  );
}
