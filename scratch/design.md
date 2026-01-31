# Silencio: RL-Trainable Music Environment

## Vision

**Silencio** = RL environment for training collaborative AI musicians.
**Reward** = metrics about how Cadenza users actually play.

Train policies that make users *better musicians* — measured by improvement, engagement, and return rate.

### The Loop

```
┌─────────────────────────────────────────────────────────────┐
│                         Cadenza                              │
│   (app: collects user behavior, deploys policies)           │
└──────────────────────────┬──────────────────────────────────┘
                           │ user metrics
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                         Silencio                             │
│   (library: RL training, policy optimization)               │
│                                                              │
│   Reward = f(timing_improvement, session_length, return)    │
│                                                              │
│   Policies:                                                  │
│     • Metronome → Drummer                                   │
│     • Drummer → Bassist                                     │
│     • Bassist → Pianist                                     │
│     • ...full band                                          │
└─────────────────────────────────────────────────────────────┘
```

### Progression

| Stage | Policy | Action Space | Reward Signal |
|-------|--------|--------------|---------------|
| v0 | Smart metronome | when to click | hand-coded (baseline) |
| v1 | Drummer | which drum, when | user timing improvement |
| v2 | Bassist | root/fifth, when | user stays in pocket |
| v3 | Pianist/Guitar | chord voicing, rhythm | user engagement, return |

---

## v0: Smart Metronome

A metronome that produces structured output: **symbolic timing + timbre embedding per beat**.

### Behavioral Model: Responsiveness Zones

The metronome doesn't just click — it *responds* to the human player with a zone-based model:

```
                    ◄─────── follow_limit ───────►
                    ◄───── follow_zone ─────►
                    ◄─── snap_zone ───►
     ────────────────────|●|────────────────────
                       beat
```

**Three zones:**

1. **Snap Zone** (±20ms): If you're close to on-beat, the AI holds you exactly on beat. Corrective, stabilizing. "You're doing great, stay right here."

2. **Follow Zone** (±100ms): If you drift further, the AI adjusts its tempo to follow your lead. Adaptive, responsive. "I hear you pushing, I'll come with you."

3. **Limits** (±200ms): Hard bounds. The AI won't let you go completely off the rails. "That's as far as I'll stretch."

```python
@dataclass
class ResponsivenessConfig:
    snap_threshold_ms: float = 20.0      # snap to grid within this
    follow_threshold_ms: float = 100.0   # follow human within this
    limit_threshold_ms: float = 200.0    # hard limit
    follow_strength: float = 0.5         # 0=ignore human, 1=match exactly

def compute_next_beat(
    expected_beat_time: float,
    human_onset_time: float,
    config: ResponsivenessConfig,
) -> float:
    """Compute when the AI should play its next beat."""
    deviation_ms = (human_onset_time - expected_beat_time) * 1000

    if abs(deviation_ms) < config.snap_threshold_ms:
        # Snap zone: hold them on beat
        return expected_beat_time

    elif abs(deviation_ms) < config.follow_threshold_ms:
        # Follow zone: blend between grid and human
        t = (abs(deviation_ms) - config.snap_threshold_ms) / (
            config.follow_threshold_ms - config.snap_threshold_ms
        )
        blend = t * config.follow_strength
        return expected_beat_time + blend * (human_onset_time - expected_beat_time)

    else:
        # Beyond follow zone: follow with limits
        max_deviation = config.limit_threshold_ms / 1000
        clamped_deviation = max(-max_deviation, min(max_deviation,
            human_onset_time - expected_beat_time))
        return expected_beat_time + clamped_deviation
```

**Why this matters for RL:**
- The responsiveness config becomes part of the **action space** or **learned parameters**
- Different musical contexts want different zones (tight jazz vs loose jam)
- The model learns when to snap vs when to follow

### Core Abstraction: Score

```python
class ResponseMode(Enum):
    SNAP = "snap"        # holding human on beat
    FOLLOW = "follow"    # adjusting to human tempo
    LIMIT = "limit"      # at maximum stretch

@dataclass
class Beat:
    """A single metronome event."""
    timestamp: float          # seconds from start (may be adjusted from grid)
    grid_timestamp: float     # ideal timestamp on perfect grid
    beat_number: int          # 1-indexed within measure
    measure: int              # 1-indexed measure number
    emphasis: float           # 0.0-1.0 (downbeat=1.0, others lower)
    timbre: TimbreEmbedding   # sound character for this beat
    response_mode: ResponseMode = ResponseMode.SNAP  # how AI responded

@dataclass
class Score:
    """Symbolic representation of metronome output."""
    bpm: float
    time_signature: tuple[int, int]  # (beats_per_measure, beat_unit)
    beats: list[Beat]

    def at(self, timestamp: float) -> Beat | None:
        """Get beat at or just before timestamp."""
        pass

    def window(self, start: float, end: float) -> list[Beat]:
        """Get all beats in time window."""
        pass

    def to_tensor(self) -> torch.Tensor:
        """Export for model consumption: [N, timing_features + embedding_dim]"""
        pass
```

### Timbre Embedding

Each beat can have a different timbre. This is the "unmarked exploration territory" for LLMs.

```python
@dataclass
class TimbreEmbedding:
    vector: np.ndarray  # shape: (embedding_dim,), e.g. 256 or 512

    @classmethod
    def from_text(cls, description: str, encoder: TextEncoder) -> "TimbreEmbedding":
        """
        Map natural language to timbre space.

        Examples:
            "warm woody click" → embedding
            "sharp metallic tick" → embedding
            "soft brushed snare" → embedding
        """
        return cls(vector=encoder.encode(description))

    @classmethod
    def from_audio(cls, audio: np.ndarray, encoder: AudioEncoder) -> "TimbreEmbedding":
        """Extract timbre from audio sample."""
        return cls(vector=encoder.encode(audio))

    def interpolate(self, other: "TimbreEmbedding", t: float) -> "TimbreEmbedding":
        """Blend between timbres. t=0 → self, t=1 → other."""
        return TimbreEmbedding(vector=(1-t) * self.vector + t * other.vector)

    def distance(self, other: "TimbreEmbedding") -> float:
        """Cosine distance in embedding space."""
        return 1 - np.dot(self.vector, other.vector) / (
            np.linalg.norm(self.vector) * np.linalg.norm(other.vector)
        )
```

### Metronome with Score Export

```python
class SmartMetronome:
    """Metronome that produces structured Score output."""

    def __init__(
        self,
        bpm: float = 120.0,
        time_signature: tuple[int, int] = (4, 4),
        timbre: TimbreEmbedding | None = None,
    ):
        self.bpm = bpm
        self.time_signature = time_signature
        self.default_timbre = timbre or TimbreEmbedding.neutral()
        self._beat_timbres: dict[int, TimbreEmbedding] = {}  # beat_number → timbre

    def set_timbre(self, beat: int, timbre: TimbreEmbedding):
        """Set custom timbre for a specific beat in the measure."""
        self._beat_timbres[beat] = timbre

    def set_timbre_pattern(self, pattern: list[TimbreEmbedding]):
        """Set timbre for entire measure pattern."""
        for i, t in enumerate(pattern, start=1):
            self._beat_timbres[i] = t

    def generate_score(self, duration_seconds: float) -> Score:
        """Generate Score for given duration."""
        beats_per_measure, _ = self.time_signature
        beat_duration = 60.0 / self.bpm
        beats = []

        timestamp = 0.0
        measure = 1
        beat_in_measure = 1

        while timestamp < duration_seconds:
            emphasis = 1.0 if beat_in_measure == 1 else 0.5
            timbre = self._beat_timbres.get(beat_in_measure, self.default_timbre)

            beats.append(Beat(
                timestamp=timestamp,
                beat_number=beat_in_measure,
                measure=measure,
                emphasis=emphasis,
                timbre=timbre,
            ))

            timestamp += beat_duration
            beat_in_measure += 1
            if beat_in_measure > beats_per_measure:
                beat_in_measure = 1
                measure += 1

        return Score(
            bpm=self.bpm,
            time_signature=self.time_signature,
            beats=beats,
        )
```

### Usage Example

```python
from silencio import SmartMetronome, TimbreEmbedding

# Create metronome
metro = SmartMetronome(bpm=90, time_signature=(4, 4))

# Set different timbres per beat
metro.set_timbre(1, TimbreEmbedding.from_text("deep kick drum"))
metro.set_timbre(2, TimbreEmbedding.from_text("closed hi-hat"))
metro.set_timbre(3, TimbreEmbedding.from_text("snare with rim"))
metro.set_timbre(4, TimbreEmbedding.from_text("closed hi-hat"))

# Generate 8 bars of score
score = metro.generate_score(duration_seconds=8 * 4 * (60/90))

# Export for model training
tensor = score.to_tensor()  # [N_beats, timing_features + embedding_dim]

# Get beats in a time window (for real-time)
upcoming = score.window(current_time, current_time + 0.5)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Silencio                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │ SmartMetro   │ ───→ │    Score     │ ───→ Model Input    │
│  │              │      │ (beats +     │                     │
│  │ bpm, time    │      │  timbres)    │                     │
│  │ sig, timbres │      │              │                     │
│  └──────────────┘      └──────────────┘                     │
│                              │                               │
│                              ▼                               │
│                    ┌──────────────────┐                     │
│                    │  Synthesis       │                     │
│                    │  (timbre →       │                     │
│                    │   audio)         │                     │
│                    └──────────────────┘                     │
│                              │                               │
│                              ▼                               │
│                         Audio Out                            │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  Python API (training)    │  Swift (inference + playback)   │
└─────────────────────────────────────────────────────────────┘
```

---

## Two Spaces

### Symbolic Space (Music Theory)

Discrete, interpretable structure. The "sheet music."

```python
@dataclass
class Pitch:
    note: int      # 0-11 (C=0, C#=1, ... B=11)
    octave: int    # 0-8

@dataclass
class Scale:
    root: Pitch
    intervals: tuple[int, ...]

@dataclass
class Rhythm:
    bpm: float
    time_signature: tuple[int, int]
    current_beat: float

@dataclass
class Harmony:
    chord: tuple[int, ...]  # intervals from root
```

### Embedding Space (Timbre)

Continuous, learnable. The "LLM hookpoint."

- Text → TimbreEmbedding (via transformer encoder)
- Audio → TimbreEmbedding (via audio encoder)
- TimbreEmbedding → Audio (via decoder/vocoder)

---

## RL Environment (future)

Built on top of Score abstraction:

```python
class SilencioEnv(gymnasium.Env):
    """RL environment for music collaboration."""

    def __init__(self, version: str = "v0"):
        self.metronome = SmartMetronome()
        self.score = None  # current score being performed

    def reset(self) -> Observation:
        self.score = self.metronome.generate_score(duration=16.0)
        return self._get_obs()

    def step(self, action) -> tuple[Observation, float, bool, bool, dict]:
        # action: what note/timbre to play
        # reward: how well it fits with the score
        pass
```

### Action Space Progression

| Version | Action | Description |
|---------|--------|-------------|
| v0 | root/silence | Binary: play root or stay silent |
| v1 | scale degree | Which note in the scale |
| v2 | degree + pitch | Continuous pitch control |
| v3 | degree + pitch + timbre | Full expressive control |

---

## Project Structure

```
silencio/
├── LICENSE
├── pyproject.toml
├── README.md
├── src/
│   └── silencio/
│       ├── __init__.py
│       ├── core/
│       │   ├── score.py        # Beat, Score
│       │   ├── metronome.py    # SmartMetronome
│       │   └── timbre.py       # TimbreEmbedding
│       ├── theory/
│       │   ├── pitch.py
│       │   ├── scale.py
│       │   ├── rhythm.py
│       │   └── harmony.py
│       ├── embedding/
│       │   ├── text_encoder.py
│       │   └── audio_encoder.py
│       ├── env/                 # RL environments (future)
│       │   └── ...
│       └── synthesis/           # Audio generation (future)
│           └── ...
├── swift/
│   └── SilencioInference/
│       ├── Package.swift
│       └── Sources/
│           └── ...
└── tests/
    └── ...
```

---

## Dependencies

**Python (core):**
- numpy
- dataclasses (stdlib)

**Python (embedding):**
- torch
- transformers (for text encoder)

**Python (RL, future):**
- gymnasium

**Swift:**
- CoreML
- AVFoundation

---

## Next Steps

1. **Implement core types**: `Beat`, `Score`, `TimbreEmbedding`, `SmartMetronome`
2. **Basic text encoder**: Map words → embedding (can start with sentence-transformers)
3. **Score serialization**: JSON/binary format for persistence
4. **Swift bridge**: Read Score in Swift for playback
5. **Simple synthesis**: Generate click sounds modulated by timbre

---

---

## Reward: Cadenza User Outcomes

Silencio doesn't simulate reward — it gets real signal from Cadenza users.

### Observable Metrics

```python
@dataclass
class UserSession:
    user_id: str
    timestamp: datetime

    # Timing data
    onset_times: list[float]      # when user played
    ai_beat_times: list[float]    # when AI played

    # Engagement
    duration_seconds: float
    completed: bool               # finished vs quit early

    # Feedback
    explicit_rating: int | None   # thumbs up/down, 1-5 stars

@dataclass
class UserHistory:
    sessions: list[UserSession]

    def timing_improvement(self, window: int = 5) -> float:
        """Variance reduction across recent sessions."""

    def return_rate(self, days: int = 7) -> float:
        """Probability of returning within N days."""

    def avg_session_duration(self) -> float:
        """Engagement proxy."""
```

### Reward Function

```python
def compute_reward(session: UserSession, history: UserHistory) -> float:
    """
    Reward for a single session, given user history.

    This is what we're maximizing — users becoming better musicians
    and wanting to keep practicing.
    """

    # Short-term: did they improve during this session?
    within_session = timing_improvement_within(session)

    # Medium-term: are they improving across sessions?
    across_sessions = history.timing_improvement(window=5)

    # Engagement: are they practicing more?
    engagement = session.duration_seconds / 600  # normalize to 10min

    # Retention: do they come back? (delayed, most important)
    # This is filled in later when we know if they returned
    retention = history.return_rate(days=3)

    return (
        0.2 * within_session +
        0.2 * across_sessions +
        0.2 * engagement +
        0.4 * retention
    )
```

### Training Loop

1. **Cadenza** logs all sessions with full timing data
2. **Silencio** trains policies offline on historical data
3. **Cadenza** deploys new policy, collects more data
4. Repeat

---

## Open Questions

1. **Embedding dimension**: 256? 512? Match existing audio models?
2. **Text encoder**: sentence-transformers, CLAP, or custom?
3. **Audio synthesis**: Wavetable with timbre-controlled filtering? Neural vocoder?
4. **Real-time latency target**: 10ms? 20ms?
5. **Exploration strategy**: How to try new policies without hurting user experience?
6. **Cold start**: What policy for new users with no history?
