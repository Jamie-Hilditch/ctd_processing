# Despiking

Despiking replaces individual out-of-place samples in a channel with
`NaN`, without touching anything else around them. It's opt-in per
channel (see [Enabling despiking for a channel](#enabling-despiking-for-a-channel)
below), and every despiked channel goes through the same shape: smooth
the channel to get a **reference** series, form the **residual** against
it, resolve a **rejection threshold**, and flag any point whose residual
magnitude exceeds it. This page covers how each of those two halves —
the reference and the threshold — actually work, and how to configure and
tune them. For the full field-by-field schema, see
[Configuration → Despiking](../configuration.md#despiking) and
[`DespikeSettings`](../api/config.md)/[`DespikeChannelOverride`](../api/config.md).

## The reference series

The reference is a rolling median of the channel over
`reference_window_length` samples, centered on each point (`reference_window_length`
must be odd). Near the array's edges the window shrinks rather than
mirror-padding, so an edge sample's reference is still a median, just
over fewer points.

The default, `31`, is deliberately wider than
`pyrsktools.RSK.despike`'s own default of `3`: a median needs on the
order of 30 samples in its window before it reliably *rejects* an
isolated spike rather than being dragged toward it. A 3-sample window is
barely wider than a single spike itself, so the spike can still pull the
"reference" partway toward its own value, understating the very residual
that's supposed to flag it.

The residual is simply `data - reference`. Everything below operates on
this residual — the reference computation itself doesn't change no
matter which `rejection_method` is chosen.

One consequence worth knowing: a spike sitting near the array's boundary
can drag the reference at a *neighboring* sample too, since the
edge-shrunk window there has fewer points to dilute it. It's normal to
see a spike's immediate neighbor flagged alongside it right at the start
or end of a cast.

## The rejection threshold

Once the residual exists, `rejection_method` decides what counts as "too
large." All three options funnel into a single `|residual| > threshold`
comparison — they only differ in how `threshold` itself is resolved, and
there is deliberately **no repeat/iterate option**: despiking runs one
detect-and-replace pass and stops. That's a departure from the package's
old MAD-based design, which could safely repeat passes because a median
absolute deviation isn't inflated by the spikes it's measuring. The
default method below is different: it depends on the tail's most extreme
points to estimate its shape. Masking those out and refitting on a second
pass would remove exactly the evidence the fit needs, biasing it the same
way fitting to a truncated tail does (see the warning under
`"gpd_percentile"` below).

### `"gpd_percentile"` (default)

Fits a Generalized Pareto Distribution (GPD) to the residual's tail via
peaks-over-threshold, then extrapolates to a rejection cutoff:

1. `fit_percentile` (default `90.0`) marks where the tail is assumed to
   begin — call this point `u`. Every `|residual|` above `u` is an
   "exceedance."
2. The exceedances' shape (`xi`) and scale (`sigma`) are fit via
   `scipy.stats.genpareto`.
3. That fit is extrapolated to solve for the threshold `x` such that
   `P(|residual| > x) = 1 - rejection_percentile / 100` (default
   `rejection_percentile = 99.9`).

Why a GPD instead of a fixed multiplier of a Gaussian-style spread (an
"N × MAD" or "N × std" rule, which is what the package used before)? The
Pickands–Balkema–de Haan theorem says the distribution of exceedances
above a high-enough threshold converges to a GPD *regardless of the
parent distribution's shape*. Real residuals (e.g. `practical_salinity`
against a rolling median) are often heavy-tailed rather than Gaussian, so
a fixed-multiplier rule calibrated for a Gaussian tail either misses real
spikes or floods the output with false positives on real data. The GPD
fit doesn't need to assume a shape for the bulk of the residual at all —
only that the extreme tail behaves like a GPD, which is generally true.

If fewer than `min_exceedances` points (default `200`) exceed `u`, the
fit is skipped in favor of a plain empirical percentile of `|residual|`
at `rejection_percentile` — a GPD fit on too few points isn't trustworthy,
so this falls back to something dependable instead of extrapolating from
noise. This matters most for short profiles/deployments with few samples.

### `"raw_percentile"`

Skips the fit entirely: the threshold is
`numpy.percentile(abs(residual), rejection_percentile)`, taken directly
from the data with no tail extrapolation. Useful when a data-driven
cutoff is still wanted but a GPD fit is unnecessary or would be unstable
(e.g. deliberately bypassing the fit while manually tuning a channel).

### `"absolute_threshold"`

Skips computing anything from the data: the threshold is
`rejection_threshold` verbatim, in the channel's own units. Useful once a
sensible cutoff is already known from prior analysis — no fit runs at
all, so behavior is completely deterministic run to run.

## Pooling across a deployment

Raw channels (`temperature`, `electrical_conductivity`) are despiked as a
single whole-deployment array, before any cast is split out — so they're
already effectively "pooled" with no extra work.

Derived channels (`practical_salinity` and everything computed from it)
are different: they're computed per profile, but their rejection
threshold is fit **pooled across every cast in the deployment** rather
than per cast. A GPD fit is more stable with more data, and one cast
alone is a much smaller sample than a whole deployment. This is why
`compute_derived_variables` processes a deployment's entire list of
profiles for one quantity at a time, rather than one profile fully
top-to-bottom before starting the next — see
[The processing pipeline](pipeline.md) for exactly where this runs. Every
profile is still flagged/replaced independently against that one shared
threshold, so a profile with no real spikes isn't affected by another
profile's.

## Enabling despiking for a channel

Despiking is off by default for every channel. Enable it per channel
under `[process.channels.<name>]`, then optionally override any
project-wide default in that channel's own `despiking` table:

```toml
[process.despiking]
rejection_percentile = 99.9  # project-wide default

[process.channels.practical_salinity]
despike = true

[process.channels.practical_salinity.despiking]
rejection_percentile = 99.5  # this channel only
```

Any field left out of a channel's `despiking` table inherits the
project-wide `[process.despiking]` value (or that field's own default, if
`[process.despiking]` doesn't set it either) — so a channel only needs to
override the fields it actually wants to change.

## Inspecting what got despiked

Every despike pass logs its outcome at `VERBOSE` and, if it replaced at
least one point, records the same description in the channel's (or, for
a pooled derived-channel fit, each profile's) permanent `history` — which
ends up as CF `standard_name`/global metadata on written output, so a
saved file is self-explanatory about how its own despiking decision was
made, without needing to re-run with verbose logging. The message shape
depends on `rejection_method`:

```text
despiked 3 point(s) (GPD fit: threshold=0.041, xi=0.52, sigma=0.0031, n=1200, n_exceedances=120)
despiked 3 point(s) (empirical percentile fallback: threshold=0.038, n=150, n_exceedances=12 < min_exceedances=200)
despiked 3 point(s) (raw percentile: threshold=0.038, percentile=99.9, n=1200)
despiked 3 point(s) (absolute threshold=0.05)
```

A `0` count is still logged (though not recorded to history) so a run
that despiked nothing stays visible. If a channel is flagging far more or
far fewer points than expected, this is the first place to look: an
unexpectedly small `n`/`n_exceedances` usually means too little data for
a stable fit (check `min_exceedances`), and an `"empirical percentile
fallback"` message where a real GPD fit was expected means the same
thing.

## See also

- [Configuration → Despiking](../configuration.md#despiking) — the full
  field reference.
- [`DespikeSettings`](../api/config.md) /
  [`DespikeChannelOverride`](../api/config.md) — the underlying schema.
- [The processing pipeline](pipeline.md) — where despiking runs relative
  to raw-channel corrections, geolocation, and derived variables.
