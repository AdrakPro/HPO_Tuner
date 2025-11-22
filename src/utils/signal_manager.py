"""
Centralized signal management to prevent conflicts and ensure proper cleanup.
"""

import signal
import sys
from typing import List, Callable

from src.logger.logger import logger


class _SignalManager:
    _instance = None
    _cleanup_handlers: List[Callable] = []
    _original_handlers = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_SignalManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return

        self._original_handlers[signal.SIGINT] = signal.signal(
            signal.SIGINT, self.handle_signal
        )
        self._original_handlers[signal.SIGTERM] = signal.signal(
            signal.SIGTERM, self.handle_signal
        )
        self._initialized = True
        logger.info("Signal manager initialized")

    def register_cleanup_handler(self, handler: Callable):
        """Register cleanup handlers in order of execution."""
        self._cleanup_handlers.append(handler)

    def unregister_cleanup_handler(self, handler: Callable):
        if handler in self._cleanup_handlers:
            self._cleanup_handlers.remove(handler)

    def handle_signal(self, signum, frame):
        logger.info(
            f"Received signal {signum}, initiating graceful shutdown..."
        )

        # Execute cleanup handlers in reverse order (LIFO)
        for handler in reversed(self._cleanup_handlers):
            try:
                handler()
            except Exception as e:
                logger.error(f"Cleanup handler error: {e}")

        # Restore original handler and re-raise
        signal.signal(
            signum, self._original_handlers.get(signum, signal.SIG_DFL)
        )
        sys.exit(1 if signum in (signal.SIGINT, signal.SIGTERM) else 0)


signal_manager = _SignalManager()
