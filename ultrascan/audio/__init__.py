"""Audio output path (DESIGN §3): L3 audification worker -> SPSC ring -> L0' output callback.

This package is the second of the two physically separated passes (DESIGN §1):
strict-FIFO audio. It shares nothing with the display path except the L1 ring,
where each side holds its own ``RingReader`` cursor.
"""
