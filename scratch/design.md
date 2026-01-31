# Silencio: RL-Trainable Music Environment

## Vision

Real-time responsive collaborative AI musicians. Start with a smart metronome that exports symbolic + timbre data. Build up to full band collaboration.

---

## v0: Smart Metronome

A metronome that produces structured output: **symbolic timing + timbre embedding per beat**.

### Core Abstraction: Score

```python
@dataclass
class Beat:
    """A single metronome event."""
    timestamp: float          # seconds from start
    beat_number: int          # 1-indexed within measure
    measure: int              # 1-indexed measure number
    emphasis: float           # 0.0-1.0 (downbeat=1.0, others lower)
    timbre: TimbreEmbedding   # sound character for this beat

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

## Open Questions

1. **Embedding dimension**: 256? 512? Match existing audio models?
2. **Text encoder**: sentence-transformers, CLAP, or custom?
3. **Audio synthesis**: Wavetable with timbre-controlled filtering? Neural vocoder?
4. **Real-time latency target**: 10ms? 20ms?
