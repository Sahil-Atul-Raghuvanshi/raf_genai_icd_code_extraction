# Performance Optimization Guide - RAFgenAI System

## Executive Summary

The current implementation processes clinical notes sequentially with multiple LLM calls, resulting in **40-60 seconds** processing time per document. This guide outlines optimization strategies to reduce processing time by **70-80%** (target: 10-15 seconds per document).

---

## Current Performance Bottlenecks

### Identified Time-Consuming Operations

| Operation | Current Time | Frequency | Total Impact |
|-----------|--------------|-----------|--------------|
| **LLM Semantic Extraction** | 3s per call | 1 per chunk (10-50 chunks) | **30-150s** |
| **FAISS Code Correction** | 4s per call | 1 per invalid code (0-10 codes) | **0-40s** |
| **GEM Multi-Mapping Selection** | 3s per call | 1 per multi-map (0-5 codes) | **0-15s** |
| **Rate Limiting Delays** | 1s per pause | Every 5 operations | **2-8s** |
| **FAISS Index Loading** | 1-2s | Once per session | **1-2s** |
| **DataFrame Queries** | 0.1s per query | 100-500 queries | **10-50s** |

### Example Timeline (10-chunk document)

```
Current Performance (Sequential):
┌─────────────────────────────────────────────────────────┐
│ Semantic Extraction (10 chunks × 3s)         30s  ████████████████
│ Rate Limiting (2 pauses × 1s)                 2s  █
│ FAISS Corrections (2 codes × 4s)              8s  ████
│ GEM Selections (3 mappings × 3s)              9s  ████
│ DataFrame Queries & Processing                 5s  ██
│───────────────────────────────────────────────────┤
│ TOTAL TIME:                                   54s  ███████████████████████
└─────────────────────────────────────────────────────────┘

Target Performance (Optimized):
┌─────────────────────────────────────────────────────────┐
│ Semantic Extraction (batched)                  5s  ███
│ FAISS Corrections (parallel)                   2s  █
│ GEM Selections (optimized)                     2s  █
│ Processing                                     1s  █
│───────────────────────────────────────────────────┤
│ TOTAL TIME:                                   10s  ██████
└─────────────────────────────────────────────────────────┘

Improvement: 81% faster
```

---

## Optimization Strategies

### 🔥 Phase 1: High Impact Optimizations (60-80% Time Reduction)

#### 1. Batch LLM Processing

**Current Approach:**
```python
# Sequential calls - ONE chunk at a time
for chunk in chunks:
    result = extract_icd_from_chunk(chunk)  # 3s × 10 = 30s
```

**Optimized Approach:**
```python
# Batch processing - MULTIPLE chunks at once
def extract_icd_from_chunks_batch(chunks, batch_size=5):
    """Process multiple chunks in a single LLM call"""
    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_result = extract_icd_from_chunk_batch(batch)  # 5s for 5 chunks
        results.extend(batch_result)
    return results
```

**Benefits:**
- ⏱️ Time: 30s → 6s (80% reduction)
- 💰 Cost: Fewer API calls = lower cost
- 🎯 Accuracy: Maintained (same model, same data)

**Implementation Complexity:** Medium
- Requires prompt modification
- Need to parse batch responses
- Handle batch failures gracefully

---

#### 2. Parallel LLM Processing

**Current Approach:**
```python
# Sequential execution
for invalid_code in invalid_semantic_codes:
    correct_invalid_code(invalid_code)  # 4s × 3 = 12s
```

**Optimized Approach:**
```python
from concurrent.futures import ThreadPoolExecutor

# Parallel execution
def correct_codes_parallel(invalid_codes, conditions, max_workers=3):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(correct_invalid_code, code, cond)
            for code, cond in zip(invalid_codes, conditions)
        ]
        results = [f.result() for f in futures]
    return results
```

**Benefits:**
- ⏱️ Time: 12s → 4s (67% reduction for 3 parallel calls)
- ⚡ Throughput: 3x improvement
- 🔄 Scalability: Better resource utilization

**Implementation Complexity:** Medium-High
- Need thread-safe operations
- Handle concurrent API rate limits
- Implement proper error handling

**Caution:**
- Gemini API has rate limits (check: requests per minute)
- Use `max_workers=3` to avoid throttling
- Consider exponential backoff for failures

---

#### 3. Skip Low-Value Corrections

**Current Approach:**
```python
# Corrects ALL invalid codes
for invalid_code in invalid_semantic_codes:
    corrected = correct_invalid_code(invalid_code)
```

**Optimized Approach:**
```python
# Only correct if necessary
def should_correct(invalid_code, condition_text):
    """Determine if correction is worth the LLM call"""
    # Skip if:
    # 1. Code is close to valid (e.g., missing decimal: E119 → E11.9)
    # 2. Condition has low confidence
    # 3. Correction unlikely to change final outcome
    
    if is_simple_format_error(invalid_code):
        return fix_format(invalid_code)  # Instant fix
    
    if confidence_score(condition_text) < 0.5:
        return None  # Skip low-confidence
    
    return True  # Needs LLM correction

for invalid_code in invalid_semantic_codes:
    if should_correct(invalid_code, condition):
        corrected = correct_invalid_code(invalid_code)
```

**Benefits:**
- ⏱️ Time: 8s → 3s (60% reduction by skipping 2/3 corrections)
- 💰 Cost: Fewer LLM calls
- 🎯 Focus: Correct only high-value codes

**Implementation Complexity:** Low-Medium
- Implement confidence scoring
- Add simple format fixes
- Define skip criteria

---

#### 4. Optimize GEM Selection Logic

**Current Approach:**
```python
if len(selected_matches) > 1:
    # Always call LLM for multiple mappings
    best_code = select_best_icd10_from_gem(...)  # 3s
```

**Optimized Approach:**
```python
if len(selected_matches) > 1:
    # Check if LLM is needed
    if codes_are_functionally_equivalent(selected_matches):
        # Select based on simple rule (e.g., most specific)
        return select_most_specific(selected_matches)  # Instant
    
    if has_clear_evidence_match(selected_matches, evidence):
        # Direct match in evidence snippet
        return match_evidence_to_code(selected_matches, evidence)  # Instant
    
    # Only call LLM if genuinely ambiguous
    best_code = select_best_icd10_from_gem(...)  # 3s
```

**Benefits:**
- ⏱️ Time: 9s → 3s (67% reduction by skipping 2/3 LLM calls)
- 🎯 Accuracy: Similar (rule-based when obvious)
- 💰 Cost: Lower

**Implementation Complexity:** Medium
- Implement equivalence checking
- Add evidence matching logic
- Define clear selection rules

---

### ⚡ Phase 2: Medium Impact Optimizations (20-40% Reduction)

#### 5. Cache Master Data Lookups

**Current Approach:**
```python
# Repeated DataFrame queries
for code in codes:
    match = icd10_master_df[icd10_master_df["icd_code"] == code]
    description = match.iloc[0]["long_title"]  # Slow!
```

**Optimized Approach:**
```python
# Pre-build lookup dictionaries (once)
@st.cache_data
def build_icd_lookups():
    icd10_lookup = dict(zip(
        icd10_master_df['icd_code'], 
        icd10_master_df['long_title']
    ))
    icd10_billable = dict(zip(
        icd10_master_df['icd_code'],
        icd10_master_df['is_billable']
    ))
    return icd10_lookup, icd10_billable

# Fast lookups
icd10_lookup, icd10_billable = build_icd_lookups()
for code in codes:
    description = icd10_lookup.get(code, "Unknown")  # Instant!
```

**Benefits:**
- ⏱️ Time: 5s → 0.5s (90% reduction for DataFrame operations)
- 🚀 Speed: O(1) vs O(n) lookups
- 💾 Memory: Minimal (few MB for dictionaries)

**Implementation Complexity:** Low
- Simple dictionary conversion
- Add Streamlit caching
- Minimal code changes

---

#### 6. Pre-load FAISS Index

**Current Approach:**
```python
# Loads lazily on first correction
def correct_invalid_code(...):
    candidates = find_similar_by_invalid_code(...)  # Loads index here (2s)
```

**Optimized Approach:**
```python
# Pre-load on app startup
@st.cache_resource
def preload_faiss_index():
    """Load FAISS index once at startup"""
    return load_faiss_index()

# At app startup (before processing)
faiss_index = preload_faiss_index()  # 2s once, then cached

# During processing
def correct_invalid_code(...):
    candidates = find_similar_by_invalid_code(...)  # Instant (already loaded)
```

**Benefits:**
- ⏱️ Time: Moves 2s delay to startup (one-time cost)
- 📊 UX: Smoother processing experience
- 🔄 Reusability: Index cached across sessions

**Implementation Complexity:** Very Low
- Add `@st.cache_resource` decorator
- Call once at app start
- Already mostly implemented

---

#### 7. Reduce FAISS Search Space

**Current Approach:**
```python
# Search 3x candidates to account for billable filtering
search_k = top_k * 3  # Search 15 codes to get 5 billable
```

**Optimized Approach:**
```python
# Optimize search multiplier based on billable ratio
@st.cache_data
def calculate_billable_ratio(icd10_master_df):
    """Calculate % of billable codes"""
    billable_count = (icd10_master_df['is_billable'] == '1').sum()
    total_count = len(icd10_master_df)
    return billable_count / total_count  # ~0.85 (85% billable)

billable_ratio = calculate_billable_ratio(icd10_master_df)
search_multiplier = 1 / billable_ratio  # ~1.18 instead of 3

# Search fewer codes
search_k = int(top_k * search_multiplier)  # Search 6 codes instead of 15
```

**Benefits:**
- ⏱️ Time: 0.5s → 0.2s per FAISS search (60% reduction)
- 🎯 Accuracy: Same (still gets 5 billable codes)
- ⚡ Efficiency: Less computation

**Implementation Complexity:** Low
- Calculate ratio once
- Adjust multiplier
- Test coverage

---

#### 8. Smart Rate Limiting

**Current Approach:**
```python
# Fixed rate limiting
if i > 0 and i % 5 == 0:
    time.sleep(1)  # Always sleep
```

**Optimized Approach:**
```python
import time

class AdaptiveRateLimiter:
    def __init__(self, max_rpm=60):
        self.max_rpm = max_rpm
        self.call_times = []
    
    def wait_if_needed(self):
        """Only sleep if approaching rate limit"""
        now = time.time()
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        if len(self.call_times) >= self.max_rpm:
            # At limit, wait until oldest call expires
            sleep_time = 60 - (now - self.call_times[0])
            time.sleep(max(0, sleep_time))
        
        self.call_times.append(now)

limiter = AdaptiveRateLimiter(max_rpm=60)
for chunk in chunks:
    limiter.wait_if_needed()  # Only sleeps when necessary
    result = extract_icd_from_chunk(chunk)
```

**Benefits:**
- ⏱️ Time: 2s → 0s (no unnecessary waits)
- 🎯 Precision: Only waits when needed
- 🛡️ Safety: Still prevents rate limit errors

**Implementation Complexity:** Medium
- Implement rate limiter class
- Track API calls
- Handle edge cases

---

### 💡 Phase 3: Advanced Optimizations (Additional 10-20% Improvement)

#### 9. Progressive Rendering

**Current Approach:**
```python
# Process everything, then display
results = process_all_chunks(chunks)
display_results(results)
```

**Optimized Approach:**
```python
# Display results as they become available
result_placeholder = st.empty()
partial_results = []

for chunk in chunks:
    result = process_chunk(chunk)
    partial_results.append(result)
    
    # Update display immediately
    with result_placeholder:
        display_partial_results(partial_results)
```

**Benefits:**
- 📊 UX: User sees progress immediately
- ⏱️ Perceived Time: Feels faster
- 🔄 Interactivity: Can stop if results look good

**Implementation Complexity:** Medium
- Use Streamlit placeholders
- Handle partial data display
- Update UI incrementally

---

#### 10. Result Caching

**Current Approach:**
```python
# Always process, even for identical inputs
result = extract_icd_from_chunk(chunk)
```

**Optimized Approach:**
```python
from functools import lru_cache
import hashlib

@st.cache_data(ttl=3600)  # Cache for 1 hour
def extract_icd_from_chunk_cached(chunk_hash):
    """Cache results by chunk content hash"""
    return extract_icd_from_chunk(chunk)

def process_with_cache(chunk):
    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
    return extract_icd_from_chunk_cached(chunk_hash)
```

**Benefits:**
- ⏱️ Time: Instant for repeated chunks
- 💰 Cost: No redundant LLM calls
- 🔄 Reusability: Cache across sessions

**Implementation Complexity:** Low-Medium
- Add caching decorator
- Hash chunk content
- Handle cache invalidation

---

#### 11. Use Faster LLM Model

**Current Approach:**
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Current model
    temperature=0,
    request_timeout=30
)
```

**Alternative Models:**
```python
# Option 1: Gemini Flash (current) - Balanced
# Speed: 2-3s per call
# Accuracy: High
# Cost: Low

# Option 2: Gemini Pro (if available) - Better accuracy
# Speed: 4-5s per call
# Accuracy: Higher
# Cost: Medium

# Option 3: Smaller custom model - Faster
# Speed: 0.5-1s per call
# Accuracy: Lower
# Cost: Very low

# Recommendation: Stick with Flash for balance
# Or use tiered approach:
# - Flash for semantic extraction (needs accuracy)
# - Smaller model for simple corrections (needs speed)
```

**Benefits:**
- ⏱️ Time: Varies by model
- 💰 Cost: Trade-off between speed/accuracy/cost
- 🎯 Flexibility: Different models for different tasks

**Implementation Complexity:** Low
- Change model parameter
- Test accuracy impact
- Monitor performance

---

#### 12. Async/Await Pattern

**Current Approach:**
```python
# Synchronous execution
for chunk in chunks:
    result = extract_icd_from_chunk(chunk)
```

**Optimized Approach:**
```python
import asyncio
from langchain.callbacks import AsyncCallbackHandler

async def extract_icd_async(chunk):
    """Async version of extraction"""
    return await extract_icd_from_chunk_async(chunk)

async def process_chunks_async(chunks):
    """Process all chunks concurrently"""
    tasks = [extract_icd_async(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    return results

# Usage
results = asyncio.run(process_chunks_async(chunks))
```

**Benefits:**
- ⏱️ Time: Near-parallel execution
- 🔄 Concurrency: Better I/O utilization
- 📊 Scalability: Handles more requests

**Implementation Complexity:** High
- Convert to async functions
- Handle async Streamlit integration
- Manage async context

**Caution:**
- Streamlit has limited async support
- May require workarounds
- Test thoroughly

---

## Implementation Roadmap

### Quick Wins (Week 1)
**Priority:** Implement immediately for fast results

1. ✅ **Cache Master Data Lookups** (2 hours)
   - Build dictionaries for ICD lookups
   - Add Streamlit caching
   - Test performance

2. ✅ **Pre-load FAISS Index** (1 hour)
   - Add cache decorator
   - Call at startup
   - Verify caching

3. ✅ **Reduce FAISS Search Space** (2 hours)
   - Calculate billable ratio
   - Adjust multiplier
   - Test accuracy

**Expected Improvement:** 25-30% faster

---

### Medium-Term (Week 2-3)
**Priority:** Significant architectural improvements

4. ✅ **Skip Low-Value Corrections** (8 hours)
   - Implement confidence scoring
   - Add format fix logic
   - Define skip criteria

5. ✅ **Optimize GEM Selection** (8 hours)
   - Implement equivalence checking
   - Add evidence matching
   - Test accuracy

6. ✅ **Smart Rate Limiting** (4 hours)
   - Build adaptive limiter
   - Test with API
   - Monitor errors

**Expected Improvement:** 50-60% faster (cumulative)

---

### Long-Term (Week 4-6)
**Priority:** Major performance overhaul

7. ✅ **Batch LLM Processing** (16 hours)
   - Modify prompts for batching
   - Implement batch parsing
   - Handle edge cases
   - Extensive testing

8. ✅ **Parallel LLM Processing** (12 hours)
   - Implement thread pool
   - Handle concurrency
   - Test rate limits
   - Error handling

9. ⚠️ **Progressive Rendering** (8 hours)
   - Add Streamlit placeholders
   - Implement incremental updates
   - UX testing

**Expected Improvement:** 70-80% faster (cumulative)

---

## Performance Testing Methodology

### Benchmark Suite

Create standard test cases to measure improvements:

```python
# test_performance.py
import time
import pandas as pd

class PerformanceTest:
    def __init__(self):
        self.test_cases = {
            "small": "5-chunk document",
            "medium": "15-chunk document",
            "large": "30-chunk document"
        }
    
    def benchmark(self, optimize_level="baseline"):
        results = []
        for name, test_case in self.test_cases.items():
            start = time.time()
            process_document(test_case, optimize_level)
            elapsed = time.time() - start
            results.append({
                "test": name,
                "optimize": optimize_level,
                "time": elapsed
            })
        return pd.DataFrame(results)
    
    def compare_optimizations(self):
        baseline = self.benchmark("baseline")
        phase1 = self.benchmark("phase1")
        phase2 = self.benchmark("phase2")
        phase3 = self.benchmark("phase3")
        
        comparison = pd.concat([baseline, phase1, phase2, phase3])
        print(comparison)
        return comparison
```

### Metrics to Track

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| **Total Processing Time** | 54s | <15s | End-to-end timer |
| **LLM Calls** | 15 | <8 | Counter |
| **API Cost** | $0.50 | <$0.20 | Gemini pricing |
| **Memory Usage** | 200MB | <300MB | Process monitor |
| **Cache Hit Rate** | 0% | >60% | Cache stats |
| **Accuracy (ICD codes)** | 95% | >93% | Validation set |

---

## Risk Assessment & Mitigation

### Potential Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Accuracy Degradation** | Medium | High | Comprehensive testing, A/B comparison |
| **API Rate Limiting** | Medium | Medium | Adaptive rate limiter, monitoring |
| **Increased Memory Usage** | Low | Low | Monitor, optimize caching |
| **Complexity Increase** | High | Medium | Good documentation, tests |
| **Cache Invalidation Issues** | Low | Medium | Clear cache strategy, versioning |

### Rollback Plan

1. **Feature Flags:** Enable/disable optimizations
2. **Version Control:** Tag before major changes
3. **Performance Monitoring:** Track metrics continuously
4. **Fallback Logic:** Default to safe baseline if errors

---

## Cost-Benefit Analysis

### Development Cost

| Phase | Hours | Developer Cost | Risk |
|-------|-------|----------------|------|
| **Phase 1** | 20h | Low | Low |
| **Phase 2** | 40h | Medium | Medium |
| **Phase 3** | 60h | High | High |
| **Total** | 120h | - | - |

### Expected Benefits

| Benefit | Baseline | Optimized | Improvement |
|---------|----------|-----------|-------------|
| **Processing Time** | 54s | 10s | 81% faster |
| **User Satisfaction** | Medium | High | +40% |
| **Throughput** | 65 docs/hr | 360 docs/hr | 5.5x |
| **API Cost** | $0.50/doc | $0.15/doc | 70% cheaper |
| **Annual Savings** (10k docs) | - | $3,500 | - |

**ROI:** Positive after ~200 documents processed

---

## Monitoring & Observability

### Key Performance Indicators (KPIs)

```python
# Add timing decorators
import time
from functools import wraps

def monitor_performance(operation_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            
            # Log metrics
            log_metric(operation_name, elapsed)
            
            # Alert if slow
            if elapsed > threshold(operation_name):
                alert_slow_operation(operation_name, elapsed)
            
            return result
        return wrapper
    return decorator

@monitor_performance("semantic_extraction")
def extract_icd_from_chunk(chunk):
    # ... existing code ...
    pass
```

### Dashboard Metrics

Track in Streamlit or external dashboard:
- Average processing time per document
- LLM call count and latency
- Cache hit rate
- Error rate
- API cost per document

---

## Conclusion

### Summary

Implementing these optimizations can reduce processing time from **54 seconds to 10 seconds** (81% improvement) while maintaining accuracy and reducing costs.

### Recommended Action Plan

1. **Week 1:** Implement Quick Wins (Phase 1)
   - Immediate 30% improvement
   - Low risk, high reward

2. **Week 2-3:** Selective Phase 2 implementations
   - Focus on skip logic and GEM optimization
   - 50-60% cumulative improvement

3. **Week 4+:** Evaluate need for Phase 3
   - Batch processing if throughput critical
   - Async if handling multiple documents

### Success Criteria

✅ Processing time < 15 seconds  
✅ Accuracy maintained (>93%)  
✅ API cost reduced by 50%  
✅ User satisfaction improved  
✅ System stability maintained  

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-11  
**Status:** Ready for Implementation
