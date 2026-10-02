"""Render supplies PORT; Compose retains port 8000."""
import os

bind = f"0.0.0.0:{int(os.environ.get('PORT', '8000'))}"
