# ForexEdgeDiscovery MMLC - Edge Finder Expansion Plan

**Created:** 2025-12-06
**Last Updated:** 2025-12-06
**Status:** In Progress
**Goal:** Transform waveform visualization tool into a Data Generation Engine + Edge Finder

---

## Progress Overview

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Test Dataset & Pipeline Validation | :white_check_mark: Complete |
| 1 | State Capture Engine | :white_check_mark: Complete |
| 2 | Matrix Serialization | :white_check_mark: Complete |
| 3 | Future Truth / Metadata | :white_check_mark: Complete |
| 4 | Storage Strategy | :white_check_mark: Complete |
| 5 | VAE Training Pipeline | :white_check_mark: Complete |
| 6 | Vector Search / Inference | :white_check_mark: Complete |
| 7 | API Endpoints | :white_check_mark: Complete |
| 8 | Frontend Integration | :white_check_mark: Complete |

---

## Quick Reference

| Decision | Choice |
|----------|--------|
| ML Framework | PyTorch |
| Vector Search | In-memory scipy (<100K vectors) |
| Architecture | Integrated into existing FastAPI |
| Session Dimensions | Dynamic per timeframe/session |
| VAE Architecture | LSTM (handles variable length natively) |
| Latent Dimension | 32 (balanced expressiveness + speed) |
| Index Sampling | Every bar (query at any point in session) |
| Validation | Phase 0 with ~100 sessions first |

---

## Architecture Overview

```
TRAINING PIPELINE:
[Parquet OHLC] -> [StateCapture] -> [Matrix Serializer] -> [Session Dataset]
                                            |
                       [FutureTruth Computer] (MAE, Drift, Edge)
                                            |
                                    [VAE Trainer] -> [Model Weights] + [Latent Vectors]

INFERENCE PIPELINE:
[Live Session] -> [StateCapture] -> [Matrix] -> [VAE Encode] -> [Latent z]
                                                                    |
                                              [KNN Search (K=500)] -> [Unique Session Filter]
                                                                    |
                                                          [Edge Probability Stats]
```

---

## Phase 0: Test Dataset & Pipeline Validation :white_check_mark:

**Goal:** Build end-to-end pipeline with small subset (~100 sessions) to validate before full implementation

### Checklist
- [x] Implement Phases 1-4 with minimal code
- [x] Generate ~100 session datasets (EURUSD, NY, M10)
- [x] Verify matrix shapes and metadata correctness
- [x] Simple sanity checks:
  - [x] Matrix values in expected ranges
  - [x] ATR normalization working
  - [x] Future truth labels correct
- [ ] Optionally train tiny VAE (10 epochs) to validate training loop

### Validation Results (2025-12-06)
- Generated **100 sessions** (EURUSD, NY, M10)
- Matrix shape: **(54, 20)** per session
- ATR: ~0.000922 (correctly computed)
- Wave level occupancy:
  - L1: 54/54 bars (100% - always active)
  - L2: 40/54 bars (74%)
  - L3: 23/54 bars (43%)
  - L4: 14/54 bars (26%)
  - L5: 7/54 bars (13%)
- Metadata validates correctly (next_bar_move, session_drift, MAE)

---

## Phase 1: State Capture Engine :white_check_mark:

**Goal:** Capture wave stack state at every bar incrementally

### Checklist
- [x] Create `backend/app/waveform/state_snapshot.py`
- [x] Create `backend/app/waveform/streaming_engine.py`
- [x] Implement `WaveSnapshot` dataclass
- [x] Implement `StackSnapshot` dataclass
- [x] Implement `StreamingWaveformEngine` class
- [x] Add `process_session_with_snapshots()` method

### Key Classes

```python
@dataclass
class WaveSnapshot:
    level: int          # 1-5
    direction: int      # +1 (UP), -1 (DOWN)
    amplitude: float    # |end_price - start_price|
    duration_bars: int  # Bars since wave started

@dataclass
class StackSnapshot:
    bar_index: int
    timestamp: datetime
    close_price: float
    waves: list[WaveSnapshot]  # Up to 5 levels

class StreamingWaveformEngine:
    def feed_candle(self, candle: Candle) -> StackSnapshot
    def process_session_with_snapshots(self, df) -> tuple[list[Wave], list[StackSnapshot]]
```

### Files Created
- `backend/app/waveform/state_snapshot.py`
- `backend/app/waveform/streaming_engine.py`

---

## Phase 2: Matrix Serialization :white_check_mark:

**Goal:** Convert wave stack to fixed `[num_bars, 20]` matrix

### Checklist
- [x] Create `backend/app/edge_finder/matrix_serializer.py`
- [x] Implement `snapshot_to_row()` function
- [x] Implement `snapshots_to_matrix()` function
- [x] Implement `compute_session_start_atr()` function
- [x] Implement `get_session_bar_count()` function

### Matrix Schema (20 channels = 5 levels x 4 features)

| Channels | Level | Features |
|----------|-------|----------|
| 0-3 | L1 | Direction, Amplitude, Duration, Leg_Count |
| 4-7 | L2 | Direction, Amplitude, Duration, Leg_Count |
| 8-11 | L3 | Direction, Amplitude, Duration, Leg_Count |
| 12-15 | L4 | Direction, Amplitude, Duration, Leg_Count |
| 16-19 | L5 | Direction, Amplitude, Duration, Leg_Count |

### Normalization
- Direction: +1/-1, 0 if inactive
- Amplitude: `(end - start) / session_start_ATR` (signed)
- Duration: `bars_since_start / total_session_bars` (0-1)
- Leg_Count: `count / expected_max`

### Dynamic Dimensions by Session/Timeframe

| Session | M5 | M10 | M15 | M30 |
|---------|-----|-----|-----|-----|
| asia (9h) | 108 | 54 | 36 | 18 |
| london (9h) | 108 | 54 | 36 | 18 |
| ny (9h) | 108 | 54 | 36 | 18 |
| full_day (22h) | 264 | 132 | 88 | 44 |

### Files Created
- `backend/app/edge_finder/__init__.py`
- `backend/app/edge_finder/matrix_serializer.py`

---

## Phase 3: Future Truth / Metadata :white_check_mark:

**Goal:** Compute ground truth labels for each bar

### Checklist
- [x] Create `backend/app/edge_finder/future_truth.py`
- [x] Implement `BarMetadata` dataclass
- [x] Implement `SessionDataset` dataclass
- [x] Implement `generate_session_id()` function
- [x] Implement `compute_bar_metadata()` function
- [x] Implement `compute_session_dataset()` function

### Key Dataclasses

```python
@dataclass
class BarMetadata:
    session_id: str           # "EURUSD_2024-01-15_london_M5"
    bar_index: int
    next_bar_move_atr: float  # Next bar move in ATR units
    session_drift_atr: float  # End price - current, ATR normalized
    mae_to_session_end: float # Max Adverse Excursion
    mfe_to_session_end: float # Max Favorable Excursion
    bars_remaining: int

@dataclass
class SessionDataset:
    session_id: str
    pair: str
    date: date
    session: str
    timeframe: str
    matrix: np.ndarray        # [num_bars, 20]
    metadata: list[BarMetadata]
    session_start_atr: float
```

### Session ID Format
`{PAIR}_{DATE}_{SESSION}_{TIMEFRAME}`
Example: `EURUSD_2024-01-15_london_M5`

### Files Created
- `backend/app/edge_finder/future_truth.py`

---

## Phase 4: Storage Strategy :white_check_mark:

**Goal:** Save and load session datasets efficiently

### Checklist
- [x] Create `backend/app/edge_finder/storage.py`
- [x] Implement `get_edge_finder_path()` function
- [x] Implement `get_sessions_path()` function
- [x] Implement `save_session_dataset()` function
- [x] Implement `load_session_dataset()` function
- [x] Implement `list_session_files()` function
- [x] Implement `get_session_stats()` function
- [x] Create `backend/app/edge_finder/generator.py`
- [x] Implement `generate_session_dataset()` function
- [x] Implement `generate_test_dataset()` function

### Directory Structure

```
cache/
├── ohlc/                    # Existing parquet files
├── edge_finder/
│   ├── sessions/            # One .npz per session
│   │   └── EURUSD_2024-01-15_london_M5.npz
│   ├── models/
│   │   ├── vae_v1.pt        # PyTorch weights
│   │   └── vae_v1_config.json
│   └── vectors/
│       ├── latent_index.npz
│       └── metadata_index.parquet
```

### Session File (.npz) Contents
- `matrix`: [num_bars, 20] float32
- `session_id`: str
- `session_start_atr`: float
- `next_bar_moves`: [num_bars] float32
- `session_drifts`: [num_bars] float32
- `maes`: [num_bars] float32
- `mfes`: [num_bars] float32

### Files Created
- `backend/app/edge_finder/storage.py`
- `backend/app/edge_finder/generator.py`

---

## Phase 5: VAE Training Pipeline :white_check_mark:

**Goal:** Train VAE to encode waveform matrices into latent vectors

### Checklist
- [x] Create `backend/app/edge_finder/model/__init__.py`
- [x] Create `backend/app/edge_finder/model/vae.py`
- [x] Create `backend/app/edge_finder/model/trainer.py`
- [x] Create `backend/app/edge_finder/model/dataset.py`
- [x] Create `backend/app/edge_finder/model/config.py`
- [x] Implement `WaveformVAE` class (LSTM-based)
- [x] Implement `WaveformDataset` PyTorch Dataset
- [x] Implement `VAETrainer` class
- [x] Implement VAE loss (reconstruction + KL divergence)
- [x] Add training loop with progress tracking
- [x] Save model weights and config

### Validation Results (2025-12-06)
- Training on 100 sessions (80 train, 20 val)
- Model: 906,964 parameters
- 10 epochs completed successfully
- Best validation loss: 27.33
- Latent vectors: [batch, 32] with reasonable distribution
- Saved files: `vae_test.pt`, `vae_test_best.pt`, `vae_test_config.json`

### VAE Architecture (LSTM-based)

```python
class WaveformVAE(nn.Module):
    def __init__(self, input_channels=20, latent_dim=32):
        # Bidirectional LSTM encoder
        self.encoder = nn.LSTM(input_size=20, hidden_size=256,
                               num_layers=2, bidirectional=True)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        # Decoder reconstructs sequence
```

### Training Config
- `latent_dim`: 32
- `batch_size`: 32
- `learning_rate`: 1e-3
- `num_epochs`: 100
- `kl_weight`: 0.1 (beta-VAE)

### Loss Function
Masked MSE reconstruction + KL divergence

### Files to Create
- `backend/app/edge_finder/model/__init__.py`
- `backend/app/edge_finder/model/vae.py`
- `backend/app/edge_finder/model/dataset.py`
- `backend/app/edge_finder/model/trainer.py`
- `backend/app/edge_finder/model/config.py`

---

## Phase 6: Vector Search / Inference :white_check_mark:

**Goal:** Find similar historical patterns and compute edge probabilities

### Checklist
- [x] Create `backend/app/edge_finder/vector_index.py`
- [x] Create `backend/app/edge_finder/inference.py`
- [x] Implement `LatentVectorIndex` class
- [x] Implement KNN search via scipy `cdist`
- [x] Implement `search_unique_sessions()` for anti-bias filtering
- [x] Implement `EdgeInferenceEngine` class
- [x] Implement `EdgeProbabilities` dataclass
- [x] Build latent index from trained VAE

### Validation Results (2025-12-06)
- Index built from 10 test sessions: **110 vectors** (every 5th bar)
- Latent dimension: **32**
- KNN search working with euclidean distance
- Unique session filtering working (anti-bias)
- Edge probability computation:
  - Next bar up %: 70%
  - Session up %: 40%
  - Avg MAE: -2.14 ATR
  - Avg MFE: 3.23 ATR
  - Risk/Reward: 1.51
- Test script: `backend/test_inference.py`

### LatentVectorIndex
- Load all latent vectors + session IDs
- KNN via scipy `cdist` (sufficient for <100K)
- `search_unique_sessions()`: Filter to best match per session

### EdgeInferenceEngine
- Encode current matrix -> latent z
- Find K=500 similar historical patterns
- Filter to unique sessions (anti-bias)
- Aggregate outcomes into EdgeProbabilities

### EdgeProbabilities Output
- `next_bar_up_pct`, `next_bar_avg_move`
- `session_up_pct`, `session_avg_drift`
- `avg_mae`, `mae_percentiles` (25th, 50th, 75th, 95th)

### Files to Create
- `backend/app/edge_finder/vector_index.py`
- `backend/app/edge_finder/inference.py`

---

## Phase 7: API Endpoints :white_check_mark:

**Goal:** Expose edge finder functionality via REST API

### Checklist
- [x] Create `backend/app/api/edge_finder.py`
- [x] Create `backend/app/schemas/edge_finder.py`
- [x] Implement `/api/edge-finder/sessions/generate` endpoint
- [x] Implement `/api/edge-finder/sessions/stats` endpoint
- [x] Implement `/api/edge-finder/sessions/list` endpoint
- [x] Implement `/api/edge-finder/training/start` endpoint
- [x] Implement `/api/edge-finder/training/stop` endpoint
- [x] Implement `/api/edge-finder/training/status` endpoint
- [x] Implement `/api/edge-finder/models` endpoint
- [x] Implement `/api/edge-finder/index/build` endpoint
- [x] Implement `/api/edge-finder/index/load` endpoint
- [x] Implement `/api/edge-finder/index/status` endpoint
- [x] Implement `/api/edge-finder/inference` endpoint
- [x] Implement `/api/edge-finder/health` endpoint
- [x] Update `backend/app/api/router.py` to include edge_finder router

### Validation Results (2025-12-06)
- All 13 endpoints working correctly
- Index build: 5292 vectors from 100 sessions
- Inference test: 100 matches found, edge probabilities computed
- Response times: <100ms for inference

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/edge-finder/sessions/generate` | Build session datasets from OHLC |
| GET | `/api/edge-finder/sessions/stats` | Get session statistics |
| GET | `/api/edge-finder/sessions/list` | List available sessions |
| POST | `/api/edge-finder/training/start` | Start VAE training |
| POST | `/api/edge-finder/training/stop` | Stop training |
| GET | `/api/edge-finder/training/status` | Get training job status |
| GET | `/api/edge-finder/models` | List available models |
| GET | `/api/edge-finder/models/{name}` | Get model info |
| POST | `/api/edge-finder/index/build` | Build vector index |
| POST | `/api/edge-finder/index/load` | Load saved index |
| GET | `/api/edge-finder/index/status` | Get index status |
| POST | `/api/edge-finder/inference` | Run edge inference |
| GET | `/api/edge-finder/health` | Health check |

### Files Created
- `backend/app/api/edge_finder.py` - 13 API endpoints
- `backend/app/schemas/edge_finder.py` - Pydantic request/response models

### Files Modified
- `backend/app/api/router.py` - Added edge_finder router

---

## Phase 8: Frontend Integration :white_check_mark:

**Goal:** Add Edge Finder UI to the frontend

### Checklist
- [x] Create `frontend/src/components/sidebar/EdgeFinderTab.tsx`
- [x] Create `frontend/src/components/chart/EdgeStatsOverlay.tsx`
- [x] Create `frontend/src/hooks/useEdgeFinder.ts`
- [x] Update `frontend/src/types/index.ts` with edge finder types
- [x] Update `frontend/src/lib/api.ts` with edge finder endpoints
- [x] Update `frontend/src/components/layout/Sidebar.tsx` to add EdgeFinder tab
- [x] Update `frontend/src/App.tsx` with edge probabilities state
- [x] Update `frontend/src/components/layout/AppLayout.tsx` to pass edge state
- [x] Implement session generation UI
- [x] Implement training controls UI
- [x] Implement model selection and index management
- [x] Implement edge probability display
- [ ] Optional: Implement historical path ghosts overlay
- [ ] Optional: Implement risk zones (MAE percentiles) overlay

### Validation Results (2025-12-06)
- Frontend build: Successful (vite build)
- EdgeFinderTab: Full-featured sidebar with status, sessions, models, training, index management
- EdgeStatsOverlay: Chart overlay for edge probability display
- React Query hooks for all edge finder operations

### UI Components

1. **EdgeFinderTab** (new sidebar tab):
   - Session generation status & trigger
   - Model training controls
   - Live inference toggle
   - Edge probability display

2. **EdgeStatsOverlay** (chart overlay):
   - Edge probability indicator
   - Optional: historical path ghosts
   - Risk zones (MAE percentiles)

### Files to Create
- `frontend/src/components/sidebar/EdgeFinderTab.tsx`
- `frontend/src/components/chart/EdgeStatsOverlay.tsx`
- `frontend/src/hooks/useEdgeFinder.ts`
- `frontend/src/types/edge-finder.ts`

### Files to Modify
- `frontend/src/lib/api.ts`
- `frontend/src/components/layout/Sidebar.tsx`

---

## Phase Dependencies

```
Phase 0 (Test Dataset Validation) -----> Validates Phases 1-4 with small subset
    |
    v
Phase 1 (State Capture)
    |
    v
Phase 2 (Matrix Serialization)
    |
    v
Phase 3 (Future Truth) <---> Phase 4 (Storage) [parallel]
    |
    v
Phase 5 (VAE Training)
    |
    v
Phase 6 (Vector Search / Inference)
    |
    v
Phase 7 (API Endpoints)
    |
    v
Phase 8 (Frontend)
```

**Recommended Order:** 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8

---

## Critical Files Summary

### Created Files :white_check_mark:

| File | Purpose |
|------|---------|
| `backend/app/waveform/state_snapshot.py` | WaveSnapshot, StackSnapshot dataclasses |
| `backend/app/waveform/streaming_engine.py` | StreamingWaveformEngine for bar-by-bar capture |
| `backend/app/edge_finder/__init__.py` | Package exports |
| `backend/app/edge_finder/matrix_serializer.py` | Convert snapshots to [N, 20] matrices |
| `backend/app/edge_finder/future_truth.py` | Compute MAE, drift, next_bar metadata |
| `backend/app/edge_finder/storage.py` | Save/load .npz session files |
| `backend/app/edge_finder/generator.py` | Process OHLC into session datasets |
| `backend/app/edge_finder/model/__init__.py` | Model package exports |
| `backend/app/edge_finder/model/config.py` | TrainingConfig, TrainingState |
| `backend/app/edge_finder/model/dataset.py` | WaveformDataset, collate function |
| `backend/app/edge_finder/model/vae.py` | WaveformVAE, encoder, decoder, loss |
| `backend/app/edge_finder/model/trainer.py` | VAETrainer class |
| `backend/test_edge_finder.py` | Phase 0 validation script |
| `backend/test_vae_training.py` | Phase 5 validation script |
| `backend/app/edge_finder/vector_index.py` | LatentVectorIndex for KNN search |
| `backend/app/edge_finder/inference.py` | EdgeInferenceEngine for edge probabilities |
| `backend/app/api/edge_finder.py` | REST API endpoints |
| `backend/app/schemas/edge_finder.py` | Pydantic request/response models |
| `frontend/src/components/sidebar/EdgeFinderTab.tsx` | Sidebar tab for edge finder controls |
| `frontend/src/components/chart/EdgeStatsOverlay.tsx` | Chart overlay for edge probabilities |
| `frontend/src/hooks/useEdgeFinder.ts` | React Query hooks for edge finder |

### Files Modified

| File | Changes |
|------|---------|
| `backend/app/api/router.py` | Include edge_finder router |
| `frontend/src/lib/api.ts` | Add edge finder API calls |
| `frontend/src/types/index.ts` | Add edge finder TypeScript types |
| `frontend/src/components/layout/Sidebar.tsx` | Add EdgeFinder tab |
| `frontend/src/components/layout/AppLayout.tsx` | Pass edge probabilities state |
| `frontend/src/App.tsx` | Add edge probabilities state |

### Files Created (Phase 6-8) :white_check_mark:

```
backend/app/edge_finder/
├── vector_index.py          # Phase 6 - COMPLETE
└── inference.py             # Phase 6 - COMPLETE

backend/app/api/
└── edge_finder.py           # Phase 7 - COMPLETE

backend/app/schemas/
└── edge_finder.py           # Phase 7 - COMPLETE

frontend/src/components/sidebar/
└── EdgeFinderTab.tsx        # Phase 8 - COMPLETE

frontend/src/components/chart/
└── EdgeStatsOverlay.tsx     # Phase 8 - COMPLETE

frontend/src/hooks/
└── useEdgeFinder.ts         # Phase 8 - COMPLETE

frontend/src/types/
└── index.ts (updated)       # Phase 8 - COMPLETE
```

---

## Requirements

### Already Added :white_check_mark:
```
numpy>=1.24.0
torch>=2.0.0
scipy>=1.10.0
```

---

## Implementation Notes

1. **LSTM over CNN**: Handles variable-length sessions natively without padding complexity

2. **Unique Session Filter**: Prevents KNN from returning 50 matches all from one volatile day

3. **ATR Normalization**: Enables pattern recognition across pairs with different volatilities

4. **Incremental State Capture**: Wraps existing engine rather than modifying it heavily

5. **In-Memory Search**: scipy cdist sufficient for <100K vectors; can upgrade to FAISS later

---

## Session Checklist

When resuming work:
- [ ] Read this plan file
- [ ] Check "Progress Overview" table at top
- [ ] Find current phase section
- [ ] Check off completed items as you go
- [ ] Run tests after each phase
- [ ] Update this plan with any deviations

---

## Next Session Quick Start

1. Open this file: `EDGE_FINDER_PLAN.md`
2. Check "Progress Overview" table at top
3. All phases complete! Edge Finder is ready to use.
4. Follow the checklist for that phase
5. Mark items complete as you finish them
