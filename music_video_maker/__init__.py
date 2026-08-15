"""Turn a master audio track + lyrics + cast photos into a lip-synced music video.

Pipeline stages (see CLAUDE.md and the blueprint PDF):

1. Forced alignment      -- stable-ts ``model.align()`` maps audio to lyric text
2. Temporal slicing      -- pydub stems + deterministic prompt expansion
3. Asset staging         -- upload images/audio to ComfyUI
4. Execution             -- mutate the workflow graph, submit, monitor over WebSocket
5. Assembly              -- FFmpeg concat + mux the pristine master track
"""

__version__ = "0.1.0"
