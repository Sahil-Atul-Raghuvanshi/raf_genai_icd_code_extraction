"""
Adaptive rate limiting for API calls.
Tracks call timestamps and only sleeps when approaching rate limits.
"""

import time
from typing import Optional


class AdaptiveRateLimiter:
    """
    Smart rate limiter that only sleeps when approaching API rate limits.
    
    Performance Optimization (Step 8):
    - Old: Fixed sleep every 5 calls (wastes time)
    - New: Only sleeps when actually approaching limit
    - Time saved: ~2s per batch (eliminates unnecessary waits)
    
    Example:
        limiter = AdaptiveRateLimiter(max_rpm=60)
        for item in items:
            limiter.wait_if_needed()
            result = call_api(item)
    """
    
    def __init__(self, max_rpm: int = 60, buffer: float = 0.9):
        """
        Initialize adaptive rate limiter.
        
        Args:
            max_rpm: Maximum requests per minute (default: 60 for Gemini API)
            buffer: Safety buffer (0.9 = use 90% of limit, default: 0.9)
        """
        self.max_rpm = max_rpm
        self.buffer = buffer
        self.effective_limit = int(max_rpm * buffer)
        self.call_times = []
    
    def wait_if_needed(self):
        """
        Wait only if approaching rate limit.
        
        Algorithm:
        1. Remove calls older than 60 seconds
        2. Count recent calls in the last minute
        3. If at/near limit, sleep until oldest call expires
        4. Record current call timestamp
        """
        now = time.time()
        
        # Remove calls older than 1 minute (60 seconds)
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        # Check if approaching rate limit
        if len(self.call_times) >= self.effective_limit:
            # At limit, wait until oldest call expires
            oldest_call = self.call_times[0]
            sleep_time = 60 - (now - oldest_call)
            
            if sleep_time > 0:
                time.sleep(sleep_time + 0.1)  # Add 0.1s buffer
                
                # Clean up expired calls after sleeping
                now = time.time()
                self.call_times = [t for t in self.call_times if now - t < 60]
        
        # Record this call
        self.call_times.append(now)
    
    def get_current_rate(self) -> int:
        """
        Get current requests per minute.
        
        Returns:
            Number of requests in the last 60 seconds
        """
        now = time.time()
        self.call_times = [t for t in self.call_times if now - t < 60]
        return len(self.call_times)
    
    def get_wait_time(self) -> float:
        """
        Calculate how long to wait before next call.
        
        Returns:
            Wait time in seconds (0 if can call immediately)
        """
        now = time.time()
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        if len(self.call_times) >= self.effective_limit:
            oldest_call = self.call_times[0]
            return max(0, 60 - (now - oldest_call))
        
        return 0.0
    
    def reset(self):
        """Reset the rate limiter (clear all call history)."""
        self.call_times = []
    
    def __repr__(self):
        return (f"AdaptiveRateLimiter(max_rpm={self.max_rpm}, "
                f"current_rate={self.get_current_rate()}, "
                f"wait_time={self.get_wait_time():.2f}s)")


class BatchRateLimiter:
    """
    Rate limiter optimized for batch processing.
    
    For batch calls where N items are processed in one API call,
    this limiter accounts for the batch size.
    
    Example:
        limiter = BatchRateLimiter(max_rpm=60, batch_size=5)
        for batch in batches:
            limiter.wait_if_needed()
            result = call_api_batch(batch)  # Processes 5 items
    """
    
    def __init__(self, max_rpm: int = 60, batch_size: int = 5, buffer: float = 0.9):
        """
        Initialize batch rate limiter.
        
        Args:
            max_rpm: Maximum requests per minute
            batch_size: Number of items processed per API call
            buffer: Safety buffer (0.9 = use 90% of limit)
        """
        self.max_rpm = max_rpm
        self.batch_size = batch_size
        self.buffer = buffer
        # Effective limit is based on batch size
        # If batch_size=5, each call processes 5 items
        # So we can make fewer calls
        self.effective_limit = int((max_rpm / batch_size) * buffer)
        self.call_times = []
    
    def wait_if_needed(self):
        """Wait only if approaching batch rate limit."""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        # Check if approaching limit
        if len(self.call_times) >= self.effective_limit:
            oldest_call = self.call_times[0]
            sleep_time = 60 - (now - oldest_call)
            
            if sleep_time > 0:
                time.sleep(sleep_time + 0.1)
                
                now = time.time()
                self.call_times = [t for t in self.call_times if now - t < 60]
        
        self.call_times.append(now)
    
    def get_current_rate(self) -> int:
        """Get current batch calls per minute."""
        now = time.time()
        self.call_times = [t for t in self.call_times if now - t < 60]
        return len(self.call_times)
    
    def get_effective_item_rate(self) -> int:
        """Get effective items processed per minute."""
        return self.get_current_rate() * self.batch_size
    
    def reset(self):
        """Reset the rate limiter."""
        self.call_times = []
    
    def __repr__(self):
        return (f"BatchRateLimiter(max_rpm={self.max_rpm}, "
                f"batch_size={self.batch_size}, "
                f"current_calls={self.get_current_rate()}, "
                f"items_per_min={self.get_effective_item_rate()})")


# Pre-configured rate limiters for common APIs

# Gemini API (Free tier: 15 RPM, Paid: 60 RPM)
gemini_free_limiter = AdaptiveRateLimiter(max_rpm=15, buffer=0.9)
gemini_paid_limiter = AdaptiveRateLimiter(max_rpm=60, buffer=0.9)

# OpenAI API (GPT-4: 500 RPM, GPT-3.5: 3500 RPM)
openai_gpt4_limiter = AdaptiveRateLimiter(max_rpm=500, buffer=0.95)
openai_gpt35_limiter = AdaptiveRateLimiter(max_rpm=3500, buffer=0.95)
