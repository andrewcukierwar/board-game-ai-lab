# api/connect4/state.py
import uuid, threading

_games   = {}              # {game_id: GameState}
_lock    = threading.Lock()

def new_game(game_obj, p1, p2):
    gid = str(uuid.uuid4())
    with _lock:
        _games[gid] = dict(
            board   = game_obj,
            player1 = p1,
            player2 = p2,
        )
    return gid

def get_game(gid):
    with _lock:
        return _games.get(gid)

def delete_game(gid):
    with _lock:
        _games.pop(gid, None)
