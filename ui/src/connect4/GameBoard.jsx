import { humanTurn } from './useConnect4Game.js';

export default function GameBoard({ game, busy, uncertain, move, highlights = [] }) {
  const squareNames = new Set(highlights.map(square => square.name));
  return <div className="board-surface">
    <div className="board-topline"><span className="eyebrow">The playing field</span><span className="board-revision">{game ? `Move ${game.revision}` : 'Ready when you are'}</span></div>
    <div className="column-guide" aria-hidden="true">{[1, 2, 3, 4, 5, 6, 7].map(column => <span key={column}>{column}<i>↓</i></span>)}</div>
    <div id="game-board" role="group" aria-label="Connect 4 board" aria-busy={busy}>
      {game ? game.board.flatMap((row, rowIndex) => row.map((piece, column) => {
        const name = `${String.fromCharCode(97 + column)}${6 - rowIndex}`;
        const highlighted = squareNames.has(name);
        return <button key={`${game.game_id}:${rowIndex}:${column}`} type="button" className={`cell${highlighted ? ' explanation-square' : ''}`}
          disabled={busy || uncertain || game.gameOver || !humanTurn(game) || !game.legalMoves.includes(column)}
          aria-label={`Column ${column + 1}, row ${rowIndex + 1}: ${piece === 'X' ? 'red' : piece === 'O' ? 'yellow' : 'empty'}`}
          data-column={column} data-square={name}
          onClick={() => move(column)}>
          <span className={`circle ${piece === 'X' ? 'x' : piece === 'O' ? 'o' : 'empty'}`} />
          {highlighted && <span className="square-label" aria-hidden="true">{name}</span>}
        </button>;
      })) : Array.from({ length: 42 }, (_, index) => <span key={index} className="board-preview-slot" aria-hidden="true"><span className="circle empty" /></span>)}
    </div>
    <div className="board-legend"><span><i className="legend-piece legend-piece--red" />You · red</span><span><i className="legend-piece legend-piece--yellow" />AI · yellow</span><span>Four in a row wins</span></div>
  </div>;
}
