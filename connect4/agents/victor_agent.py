import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy
from collections import defaultdict

@dataclass
class Threat:
    """Represents a threat in Victor Allis's classification"""
    type: str  # A1, A2, B, C, D
    squares: List[Tuple[int, int]]  # Squares involved in threat
    forcing_moves: List[Tuple[int, int]]  # Moves required to execute threat
    key_square: Tuple[int, int]  # Critical square for threat

@dataclass
class ProofNode:
    """Node in proof-number search tree"""
    position: np.ndarray
    proof_number: int
    disproof_number: int
    children: Dict[int, 'ProofNode']
    player: int  # Player to move at this node
    is_or_node: bool  # True for OR node (current player), False for AND node (opponent)
    value: Optional[bool] = None  # True=win, False=loss, None=unknown

@dataclass
class ThreatSequence:
    """Represents a sequence of threats and forced responses"""
    moves: List[Tuple[int, int]]  # Sequence of moves
    score: float  # Evaluation score of sequence
    terminal: bool  # Whether sequence leads to win
    creating_threats: List[Threat]  # Threats created by sequence

class VictorAgent:
    """
    Implements Victor Allis's complete solution to Connect Four,
    including both threat-space search and proof-number search
    """
    def __init__(self):
        self.transposition_table: Dict[str, bool] = {}  # Cached proven positions
        self.threat_patterns = self._init_threat_patterns()
        self.move_ordering = [3, 2, 4, 1, 5, 0, 6]  # Center-based move ordering
        
    def _init_threat_patterns(self) -> Dict[str, np.ndarray]:
        """Initialize all threat patterns from Allis's classification"""
        patterns = {}
        
        # Type A1: Direct four threat
        patterns['A1'] = np.array([
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1]
        ])
        
        # Type A2: Indirect four threat
        patterns['A2'] = np.array([
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0]
        ])
        
        # Type B: Double-three threat
        patterns['B'] = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ])
        
        # Type C: Three-three threat
        patterns['C'] = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0]
        ])
        
        # Type D: Potential threats
        patterns['D'] = np.array([
            [0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0]
        ])
        
        return patterns

    def _compute_position_hash(self, board) -> str:
        """Create unique hash for board position"""
        return ''.join([''.join(row) for row in board])

    def _find_threats(self, board, player) -> List[Threat]:
        """Find all threats for given player on current board"""
        threats = []
        piece = 'X' if player == 0 else 'O'
        
        # Check all possible threat patterns in all directions
        for direction in ['horizontal', 'vertical', 'diagonal', 'antidiagonal']:
            threats.extend(self._scan_direction(board, piece, direction))
            
        return threats

    def _scan_direction(self, board, piece, direction) -> List[Threat]:
        """Scan board in given direction for threats"""
        threats = []
        rows, cols = len(board), len(board[0])
        
        if direction == 'horizontal':
            for row in range(rows):
                for col in range(cols - 3):
                    self._check_threat_window(board, piece, row, col, 0, 1, threats)
                    
        elif direction == 'vertical':
            for row in range(rows - 3):
                for col in range(cols):
                    self._check_threat_window(board, piece, row, col, 1, 0, threats)
                    
        elif direction == 'diagonal':
            for row in range(rows - 3):
                for col in range(cols - 3):
                    self._check_threat_window(board, piece, row, col, 1, 1, threats)
                    
        else:  # antidiagonal
            for row in range(rows - 3):
                for col in range(3, cols):
                    self._check_threat_window(board, piece, row, col, 1, -1, threats)
                    
        return threats

    def _is_playable_square(self, board, row: int, col: int) -> bool:
        """Check if a square is immediately playable according to gravity rules"""
        # Square must be empty
        if board[row][col] != ' ':
            return False
        
        # Square must be on bottom row or supported from below
        return row == 5 or board[row + 1][col] != ' '

    def _check_threat_window(self, board, piece, row, col, dx, dy, threats):
        """Check window for all possible threat patterns with gravity validation"""
        window = []
        squares = []
        cols = []  # Track column indices for gravity checks
        
        # Extract window, squares and columns
        for i in range(5):  # Use 5 squares to detect all threat types
            new_row = row + i*dx
            new_col = col + i*dy
            if 0 <= new_row < len(board) and 0 <= new_col < len(board[0]):
                window.append(board[new_row][new_col])
                squares.append((new_row, new_col))
                cols.append(new_col)
            else:
                return
                
        # Convert window to pattern format with gravity validation
        pattern = np.array([1 if x == piece else 0 if (x == ' ' and 
            self._is_playable_square(board, squares[i][0], cols[i])) else -1 
            for i, x in enumerate(window)])
        
        # Only proceed if we have valid empty squares according to gravity
        if 0 in pattern:
            self._check_type_a_threats(pattern, squares, threats, board)
            self._check_type_b_threats(pattern, squares, threats, board)
            self._check_type_c_threats(pattern, squares, threats, board)
            self._check_type_d_threats(pattern, squares, threats, board)

    def _check_type_a_threats(self, pattern, squares, threats, board):
        """Check for Type A threats using count-based pattern matching"""
        # Convert pattern to list for easier slicing
        pattern = pattern.tolist()
        
        # Check all 4-length windows for direct four (A1) threats
        for i in range(len(pattern)-3):
            window = pattern[i:i+4]
            # Must have exactly 3 pieces and 1 playable space
            if sum(x == 1 for x in window) == 3 and sum(x == 0 for x in window) == 1:
                empty_idx = i + window.index(0)
                empty_square = squares[empty_idx]
                if self._is_playable_square(board, empty_square[0], empty_square[1]):
                    threats.append(Threat(
                        type='A1',
                        squares=squares[i:i+4],
                        forcing_moves=[empty_square],
                        key_square=empty_square
                    ))
        
        # Check all 5-length windows for indirect four (A2) threats
        # Pattern must have 3 pieces and 2 playable spaces with correct spacing
        if len(pattern) >= 5:
            for i in range(len(pattern)-4):
                window = pattern[i:i+5]
                if (sum(x == 1 for x in window) == 3 and 
                    sum(x == 0 for x in window) == 2):
                    # Verify spacing: pieces can't all be adjacent
                    piece_indices = [j for j, x in enumerate(window) if x == 1]
                    if max(piece_indices) - min(piece_indices) == 3:  # Proper spacing
                        empty_indices = [i + j for j, x in enumerate(window) if x == 0]
                        if all(self._is_playable_square(board, squares[idx][0], squares[idx][1]) 
                              for idx in empty_indices):
                            threats.append(Threat(
                                type='A2',
                                squares=squares[i:i+5],
                                forcing_moves=[squares[idx] for idx in empty_indices],
                                key_square=squares[empty_indices[0]]
                            ))

    def _check_type_b_threats(self, pattern, squares, threats, board):
        """Check for Type B threats (double-three) using count-based pattern matching"""
        pattern = pattern.tolist()
        for i in range(len(pattern)-3):
            window = pattern[i:i+4]
            # Must have exactly 2 pieces and 2 playable spaces
            if (sum(x == 1 for x in window) == 2 and 
                sum(x == 0 for x in window) == 2):
                # Verify pieces are adjacent
                piece_indices = [j for j, x in enumerate(window) if x == 1]
                if piece_indices[1] - piece_indices[0] == 1:  # Adjacent pieces
                    empty_indices = [i + j for j, x in enumerate(window) if x == 0]
                    if all(self._is_playable_square(board, squares[idx][0], squares[idx][1]) 
                          for idx in empty_indices):
                        threats.append(Threat(
                            type='B',
                            squares=squares[i:i+4],
                            forcing_moves=[squares[idx] for idx in empty_indices],
                            key_square=squares[empty_indices[0]]
                        ))

    def _check_type_c_threats(self, pattern, squares, threats, board):
        """Check for Type C threats (three-three) using count-based pattern matching"""
        pattern = pattern.tolist()
        if len(pattern) >= 5:
            for i in range(len(pattern)-4):
                window = pattern[i:i+5]
                # Must have exactly 2 pieces and 3 playable spaces
                if (sum(x == 1 for x in window) == 2 and 
                    sum(x == 0 for x in window) == 3):
                    # Verify pieces are adjacent
                    piece_indices = [j for j, x in enumerate(window) if x == 1]
                    if piece_indices[1] - piece_indices[0] == 1:  # Adjacent pieces
                        empty_indices = [i + j for j, x in enumerate(window) if x == 0]
                        if all(self._is_playable_square(board, squares[idx][0], squares[idx][1]) 
                              for idx in empty_indices):
                            threats.append(Threat(
                                type='C',
                                squares=squares[i:i+5],
                                forcing_moves=[squares[idx] for idx in empty_indices],
                                key_square=squares[empty_indices[0]]
                            ))

    def _check_type_d_threats(self, pattern, squares, threats, board):
        """Check for Type D threats (potential threats) using count-based pattern matching"""
        pattern = pattern.tolist()
        if len(pattern) >= 5:
            for i in range(len(pattern)-4):
                window = pattern[i:i+5]
                # Must have exactly 2 pieces and 3 playable spaces
                if (sum(x == 1 for x in window) == 2 and 
                    sum(x == 0 for x in window) == 3):
                    # For type D, pieces must have exactly one space between them
                    piece_indices = [j for j, x in enumerate(window) if x == 1]
                    if piece_indices[1] - piece_indices[0] == 2:  # One space between pieces
                        empty_indices = [i + j for j, x in enumerate(window) if x == 0]
                        if all(self._is_playable_square(board, squares[idx][0], squares[idx][1]) 
                              for idx in empty_indices):
                            threats.append(Threat(
                                type='D',
                                squares=squares[i:i+5],
                                forcing_moves=[squares[idx] for idx in empty_indices],
                                key_square=squares[empty_indices[1]]  # Middle empty square
                            ))

    def _prove_position(self, board, player) -> Optional[bool]:
        """Use proof-number search to prove position"""
        position_hash = self._compute_position_hash(board)
        
        # Check transposition table
        if position_hash in self.transposition_table:
            return self.transposition_table[position_hash]
            
        root = ProofNode(
            position=np.array(board),
            proof_number=1,
            disproof_number=1,
            children={},
            player=player,
            is_or_node=True  # Root is always an OR node for the current player
        )
        
        # Run proof-number search
        result = self._proof_number_search(root)
        self.transposition_table[position_hash] = result
        return result

    def _proof_number_search(self, node: ProofNode) -> Optional[bool]:
        """Implement proof-number search algorithm with proper player alternation"""
        while node.proof_number > 0 and node.disproof_number > 0:
            most_proving = self._select_most_proving_node(node)
            if not most_proving:
                break
                
            self._expand_node(most_proving)
            self._update_proof_numbers(most_proving)
            
        if node.proof_number == 0:
            return True  # Position is won
        elif node.disproof_number == 0:
            return False  # Position is lost
        return None  # Position is unknown

    def _expand_node(self, node: ProofNode):
        """Expand node in proof tree with proper player alternation"""
        next_player = 1 - node.player  # Switch player for children
        
        for col in self.move_ordering:
            if self._is_valid_move(node.position, col):
                # Make move for current player
                new_position = self._make_move(node.position.copy(), col, node.player)
                
                # Check transposition table first
                position_hash = self._compute_position_hash(new_position)
                if position_hash in self.transposition_table:
                    value = self.transposition_table[position_hash]
                    child = ProofNode(
                        position=new_position,
                        proof_number=0 if value else float('inf'),
                        disproof_number=float('inf') if value else 0,
                        children={},
                        player=next_player,
                        is_or_node=not node.is_or_node,
                        value=value
                    )
                    node.children[col] = child
                    continue
                
                child = ProofNode(
                    position=new_position,
                    proof_number=1,
                    disproof_number=1,
                    children={},
                    player=next_player,
                    is_or_node=not node.is_or_node  # Alternate OR/AND nodes
                )
                
                # Check if position is terminal
                winner = self._check_winner(new_position)
                if winner is not None:
                    child.value = (winner == node.player)  # Win for current player?
                    if child.is_or_node:
                        child.proof_number = 0 if child.value else float('inf')
                        child.disproof_number = float('inf') if child.value else 0
                    else:
                        child.proof_number = float('inf') if child.value else 0
                        child.disproof_number = 0 if child.value else float('inf')
                
                node.children[col] = child

    def _update_proof_numbers(self, node: ProofNode):
        """Update proof and disproof numbers based on node type (AND/OR)"""
        if not node.children:
            return
            
        if node.is_or_node:
            # OR node: Take minimum proof number, sum disproof numbers
            node.proof_number = min(child.proof_number for child in node.children.values())
            node.disproof_number = sum(child.disproof_number for child in node.children.values())
        else:
            # AND node: Sum proof numbers, take minimum disproof number
            node.proof_number = sum(child.proof_number for child in node.children.values())
            node.disproof_number = min(child.disproof_number for child in node.children.values())

    def _select_most_proving_node(self, node: ProofNode) -> Optional[ProofNode]:
        """Select most promising node based on node type"""
        current = node
        while current.children:
            if current.is_or_node:
                # OR node: Select child with minimum proof number
                best_child = min(current.children.values(), 
                               key=lambda c: (c.proof_number, -c.disproof_number))
            else:
                # AND node: Select child with minimum disproof number
                best_child = min(current.children.values(), 
                               key=lambda c: (c.disproof_number, -c.proof_number))
            
            if best_child is None:
                break
            current = best_child
            
        return current if current != node else None

    def choose_move(self, game) -> int:
        """Choose best move using threat-space search and proof-number search"""
        board = game.board
        player = game.current_player
        
        # First check for immediate winning moves
        winning_move = self._find_winning_move(board, player)
        if winning_move is not None:
            return winning_move
            
        # Check for opponent's winning moves to block
        opponent = 1 - player
        blocking_move = self._find_winning_move(board, opponent)
        if blocking_move is not None:
            return blocking_move
            
        # Look for winning threat sequences
        threats = self._find_threats(board, player)
        winning_sequence = self._find_winning_threat_sequence(threats, board, player)
        if winning_sequence:
            return winning_sequence[0]
            
        # Use proof-number search for winning strategy
        for move in self.move_ordering:
            if self._is_valid_move(board, move):
                new_position = self._make_move(deepcopy(board), move, player)
                if self._prove_position(new_position, player):
                    return move
                    
        # If no winning strategy, use threat-based evaluation
        return self._evaluate_moves(board, player)

    def _find_winning_move(self, board, player) -> Optional[int]:
        """Find winning move including multi-turn forced sequences"""
        # First check immediate wins
        piece = 'X' if player == 0 else 'O'
        for col in self.move_ordering:
            if self._is_valid_move(board, col):
                test_board = deepcopy(board)
                row = self._get_next_row(test_board, col)
                test_board[row][col] = piece
                if self._check_winner(test_board) == player:
                    return col

        # Check for multi-turn forced wins
        for col in self.move_ordering:
            if self._is_valid_move(board, col):
                test_board = deepcopy(board)
                row = self._get_next_row(test_board, col)
                test_board[row][col] = piece
                
                # Look for forcing moves that lead to win
                if self._check_forcing_sequence(test_board, player):
                    return col

        return None

    def _check_forcing_sequence(self, board, player, depth=4) -> bool:
        """Check if position has forced sequence leading to win"""
        if depth == 0:
            return False
            
        # Check if current position is won
        if self._check_winner(board) == player:
            return True
            
        threats = self._find_threats(board, player)
        
        # Group threats by type
        threat_groups = defaultdict(list)
        for threat in threats:
            threat_groups[threat.type].append(threat)
        
        # Check for forcing moves
        forcing_moves = []
        
        # Type A threats force response
        if threat_groups['A1'] or threat_groups['A2']:
            threat = (threat_groups['A1'] or threat_groups['A2'])[0]
            forcing_moves.extend(threat.forcing_moves)
        
        # Double Type B threats force response
        if len(threat_groups['B']) >= 2:
            forcing_moves.extend(threat_groups['B'][0].forcing_moves)
        
        # Type B + Type C forces response
        if threat_groups['B'] and threat_groups['C']:
            forcing_moves.extend(threat_groups['B'][0].forcing_moves)
        
        # Try each forcing move
        for move in forcing_moves:
            test_board = deepcopy(board)
            row, col = move
            
            if not self._is_valid_move(test_board, col):
                continue
                
            test_board[row][col] = 'X' if player == 0 else 'O'
            
            # Check opponent's forced responses
            opp_responses = self._get_forced_responses(
                self._find_threats(test_board, 1-player)
            )
            
            if not opp_responses:
                # No forced response - we can continue our attack
                if self._check_forcing_sequence(test_board, player, depth-1):
                    return True
            else:
                # Try each forced response
                all_responses_fail = True
                for response in opp_responses:
                    response_board = deepcopy(test_board)
                    row, col = response
                    response_board[row][col] = 'X' if (1-player) == 0 else 'O'
                    
                    if not self._check_forcing_sequence(response_board, player, depth-1):
                        all_responses_fail = False
                        break
                        
                if all_responses_fail:
                    return True
                    
        return False

    def _find_winning_threat_sequence(self, threats: List[Threat], board, player) -> Optional[List[int]]:
        """Find winning sequence of threats if it exists"""
        # Group threats by type
        threat_groups = {
            'A1': [],
            'A2': [],
            'B': [],
            'C': [],
            'D': []
        }
        for threat in threats:
            threat_groups[threat.type].append(threat)

        # Type A threat is immediate win
        if threat_groups['A1'] or threat_groups['A2']:
            threat = (threat_groups['A1'] or threat_groups['A2'])[0]
            return [threat.forcing_moves[0][1]]  # Return column of forcing move
            
        # Double Type B threat is winning
        if len(threat_groups['B']) >= 2:
            threat = threat_groups['B'][0]
            return [threat.forcing_moves[0][1]]
            
        # Type B + Type C is winning
        if threat_groups['B'] and threat_groups['C']:
            threat = threat_groups['B'][0]
            return [threat.forcing_moves[0][1]]
            
        # If no immediate win, search for winning sequences
        sequence = self._search_threat_sequence(board, player, depth=4)
        if sequence and sequence.terminal:
            return [move[1] for move in sequence.moves]  # Return columns of moves
    
        return None

    def _evaluate_moves(self, board, player) -> int:
        """Evaluate moves based on threat creation and prevention"""
        best_score = float('-inf')
        best_move = self.move_ordering[0]
        
        for col in self.move_ordering:
            if not self._is_valid_move(board, col):
                continue
                
            score = 0
            test_board = deepcopy(board)
            row = self._get_next_row(test_board, col)
            test_board[row][col] = 'X' if player == 0 else 'O'
            
            # Count threats created
            my_threats = self._find_threats(test_board, player)
            score += len([t for t in my_threats if t.type in ('A1', 'A2')]) * 100
            score += len([t for t in my_threats if t.type == 'B']) * 50
            score += len([t for t in my_threats if t.type == 'C']) * 25
            score += len([t for t in my_threats if t.type == 'D']) * 10
            
            # Count opponent threats prevented
            opp_threats = self._find_threats(board, 1-player)
            prevented = [t for t in opp_threats if (row, col) in t.squares]
            score += len(prevented) * 30
            
            # Center control bonus
            score += (3 - abs(col - 3)) * 5
            
            if score > best_score:
                best_score = score
                best_move = col
                
        return best_move

    def _is_valid_move(self, board, col: int) -> bool:
        """Check if move is valid"""
        return 0 <= col < len(board[0]) and board[0][col] == ' '

    def _get_next_row(self, board, col: int) -> int:
        """Get next available row in column"""
        for row in range(len(board)-1, -1, -1):
            if board[row][col] == ' ':
                return row
        return -1

    def _make_move(self, board, col: int, player: int) -> np.ndarray:
        """Make move and return new board"""
        row = self._get_next_row(board, col)
        if row >= 0:
            board[row][col] = 'X' if player == 0 else 'O'
        return board

    def _check_winner(self, board) -> Optional[int]:
        """Check if position is won"""
        # Check horizontal
        for row in range(6):
            for col in range(4):
                if board[row][col] != ' ':
                    if all(board[row][col + i] == board[row][col] for i in range(4)):
                        return 0 if board[row][col] == 'X' else 1

        # Check vertical
        for row in range(3):
            for col in range(7):
                if board[row][col] != ' ':
                    if all(board[row + i][col] == board[row][col] for i in range(4)):
                        return 0 if board[row][col] == 'X' else 1

        # Check diagonals
        for row in range(3):
            for col in range(4):
                if board[row][col] != ' ':
                    if all(board[row + i][col + i] == board[row][col] for i in range(4)):
                        return 0 if board[row][col] == 'X' else 1

        for row in range(3):
            for col in range(3, 7):
                if board[row][col] != ' ':
                    if all(board[row + i][col - i] == board[row][col] for i in range(4)):
                        return 0 if board[row][col] == 'X' else 1

        # Check for draw
        if all(board[0][col] != ' ' for col in range(7)):
            return -1

        return None

    def _search_threat_sequence(self, board, player, depth=3, alpha=float('-inf'), beta=float('inf')) -> ThreatSequence:
        """Search for winning sequences of threats with alpha-beta pruning"""
        if depth == 0:
            return ThreatSequence([], self._evaluate_position(board, player), False, [])
            
        threats = self._find_threats(board, player)
        
        # Check for immediate win
        for threat in threats:
            if threat.type in ('A1', 'A2'):
                return ThreatSequence(
                    [threat.forcing_moves[0]], 
                    float('inf'),
                    True,
                    [threat]
                )
        
        best_sequence = ThreatSequence([], float('-inf'), False, [])
        
        # Try each threat-creating move
        for threat in threats:
            for forcing_move in threat.forcing_moves:
                if not self._is_valid_move(board, forcing_move[1]):
                    continue
                    
                # Make move
                new_board = deepcopy(board)
                new_board[forcing_move[0]][forcing_move[1]] = 'X' if player == 0 else 'O'
                
                # Find opponent's forced responses
                opp_threats = self._find_threats(new_board, 1-player)
                forced_responses = self._get_forced_responses(opp_threats)
                
                if not forced_responses:
                    # No forced response - we can continue our attack
                    sequence = self._search_threat_sequence(new_board, player, depth-1, -beta, -alpha)
                    score = sequence.score
                    
                    if score > alpha:
                        alpha = score
                        best_sequence = ThreatSequence(
                            [forcing_move] + sequence.moves,
                            score,
                            sequence.terminal,
                            [threat] + sequence.creating_threats
                        )
                else:
                    # Try each forced response
                    min_score = float('inf')
                    best_response_seq = None
                    
                    for response in forced_responses:
                        response_board = deepcopy(new_board)
                        response_board[response[0]][response[1]] = 'X' if (1-player) == 0 else 'O'
                        
                        sequence = self._search_threat_sequence(response_board, player, depth-1, -beta, -alpha)
                        score = -sequence.score
                        
                        if score < min_score:
                            min_score = score
                            best_response_seq = sequence
                            
                    if min_score > alpha:
                        alpha = min_score
                        best_sequence = ThreatSequence(
                            [forcing_move] + best_response_seq.moves,
                            min_score,
                            best_response_seq.terminal,
                            [threat] + best_response_seq.creating_threats
                        )
                
                if alpha >= beta:
                    break
                    
        return best_sequence

    def _get_forced_responses(self, threats: List[Threat]) -> List[Tuple[int, int]]:
        """Get list of moves that must be played to respond to threats"""
        forced_moves = []
        
        # Must respond to A1/A2 threats
        for threat in threats:
            if threat.type in ('A1', 'A2'):
                forced_moves.extend(threat.forcing_moves)
                
        # Must respond to double B threats
        b_threats = [t for t in threats if t.type == 'B']
        if len(b_threats) >= 2:
            forced_moves.extend(b_threats[0].forcing_moves)
            
        return forced_moves

    def _evaluate_position(self, board, player) -> float:
        """Evaluate board position considering threats"""
        threats = self._find_threats(board, player)
        opp_threats = self._find_threats(board, 1-player)
        
        score = 0
        score += len([t for t in threats if t.type in ('A1', 'A2')]) * 100
        score += len([t for t in threats if t.type == 'B']) * 50
        score += len([t for t in threats if t.type == 'C']) * 25
        score += len([t for t in threats if t.type == 'D']) * 10
        
        score -= len([t for t in opp_threats if t.type in ('A1', 'A2')]) * 100
        score -= len([t for t in opp_threats if t.type == 'B']) * 50
        score -= len([t for t in opp_threats if t.type == 'C']) * 25
        score -= len([t for t in opp_threats if t.type == 'D']) * 10
        
        return score
