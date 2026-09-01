"""Configuration model and loading utilities for ctd_processing."""

import tomllib
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


def _check_output_dtype(value: str) -> None:
    """Validate that `value` names a floating-point numpy dtype.

    Parameters
    ----------
    value : str
        A dtype name, e.g. ``"float32"``.

    Raises
    ------
    ValueError
        If `value` is not a valid numpy dtype name, or names a
        non-floating dtype (e.g. an integer type) -- every channel this
        package writes holds a continuous physical quantity, so casting
        it to a non-floating dtype would silently truncate it.
    """
    try:
        dtype = np.dtype(value)
    except TypeError as exc:
        raise ValueError(
            f"output_dtype {value!r} is not a valid numpy dtype."
        ) from exc
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(
            f"output_dtype must be a floating-point dtype; got {value!r}."
        )


class ProjectSettings(BaseModel):
    """Project-level metadata attached to every output file.

    Attributes
    ----------
    name : str
        Human-readable name for this project. Intended to be attached
        to every output file's metadata (e.g. as a CF global attribute).
        Defaults to ``"my_ctd_processing_project"``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "my_ctd_processing_project"


class PathsSettings(BaseModel):
    """Directory locations for a ctd_processing project's pipeline stages.

    Attributes
    ----------
    rsk_directory : pathlib.Path
        Directory containing the raw ``.rsk`` deployment files to
        process. Required; there is no default. May be given as a
        relative or absolute path; :func:`load_settings` resolves a
        relative path against the directory containing the loaded
        config file (or the current working directory if none was
        given). ``process`` resolves ``--target`` filenames relative to
        this directory, or auto-discovers every top-level ``*.rsk``
        file inside it when no target is given. Existence of the
        directory is validated by the CLI at call time, not by this
        model.
    profiles_directory : pathlib.Path
        Directory for extracted profile files produced from
        `rsk_directory` deployments. Required; there is no default.
        Resolved the same way as `rsk_directory`.
    binned_directory : pathlib.Path
        Directory for pressure/depth-binned profile files produced from
        `profiles_directory`. Required; there is no default. Resolved
        the same way as `rsk_directory`. ``concatenate`` resolves
        ``--target`` filenames relative to this directory too, or
        auto-discovers every top-level binned file inside it (matching
        ``bin.output_format``'s extension) when no target is given.
    concatenated_file : pathlib.Path or None
        File ``concatenate`` writes its single, deduplicated,
        time-ordered CF-compliant netCDF output to. Optional; defaults
        to ``None``, meaning ``concatenate`` cannot run -- unlike every
        other field here, there is no valid "do nothing" behavior for a
        command whose entire job is to produce this one file, so it
        must be set explicitly before ``concatenate`` is used. Resolved
        the same way as `rsk_directory` when given.
    log_file : pathlib.Path or None
        File to write log records below ``ERROR`` level to. Optional;
        defaults to ``None``, meaning no such file is written. Resolved
        the same way as `rsk_directory` when given.
    error_log_file : pathlib.Path or None
        File to write log records at ``ERROR`` level and above to.
        Optional; defaults to ``None``, meaning no such file is
        written. Resolved the same way as `rsk_directory` when given.
    """

    model_config = ConfigDict(extra="forbid")

    rsk_directory: Path
    profiles_directory: Path
    binned_directory: Path
    concatenated_file: Path | None = None
    log_file: Path | None = None
    error_log_file: Path | None = None


class RawChannelSettings(BaseModel):
    """Per-raw-channel processing settings.

    Attributes
    ----------
    remove_holds : bool
        Whether to remove "held" (repeated-value) stretches from this
        channel's data before further processing. Defaults to ``True``.
    shift : int or None
        Number of samples to shift this channel's data by, e.g. to
        correct a known sensor response lag relative to the other
        channels. Follows pandas' ``Series.shift(periods=shift)``
        convention: a positive value delays the channel (each sample
        takes the value from `shift` samples earlier, leaving the first
        `shift` samples as NaN); a negative value advances it (each
        sample takes the value from ``abs(shift)`` samples later, leaving
        the last ``abs(shift)`` samples as NaN). Applied after
        `remove_holds` and before `offset`. Optional; defaults to
        ``None``, meaning no shift is applied.
    offset : float or None
        A fixed offset to add to this channel's data, e.g. to correct a
        known calibration bias. Applied after `remove_holds`/`shift`.
        Optional; defaults to ``None``, meaning no offset is applied.
    """

    model_config = ConfigDict(extra="forbid")

    remove_holds: bool = True
    shift: int | None = None
    offset: float | None = None


class ProfileSettings(BaseModel):
    """Settings for identifying individual profiles (casts) via `profinder`.

    Fields mirror `profinder.find_profiles`'s parameters, flattened into
    named, documented settings rather than passing through opaque `dict`
    kwargs. See `ctd_processing.process.profiles.find_profiles`, which
    applies these to the dataset's `sea_pressure` channel, and
    `ctd_processing.process.profiles.resolve_cast_slices`, which uses
    `direction` (not `speed_threshold_direction`) to decide which cast(s)
    of each identified turnaround are actually written out as profiles.

    Attributes
    ----------
    apply_smoothing : bool
        Whether to apply Savitzky-Golay smoothing to the pressure record
        before detecting profiles. Defaults to ``False``.
    window_length : int
        Savitzky-Golay window length, in samples. Only used if
        `apply_smoothing` is ``True``. Defaults to ``9``.
    polyorder : int
        Savitzky-Golay polynomial order. Only used if `apply_smoothing`
        is ``True``. Defaults to ``2``.
    min_pressure : float
        A profile must start and end at a sea pressure greater than this
        value, in dbar, to be considered a real, in-water profile.
        Defaults to ``-1.0``.
    peak_height : float
        Minimum sea pressure, in dbar, for a cast turnaround (maximum
        pressure) to be detected as a peak. Forwarded to
        `scipy.signal.find_peaks` as its ``height`` argument. Defaults to
        ``25.0``.
    peak_distance : int
        Minimum number of samples between detected cast turnarounds.
        Forwarded to `scipy.signal.find_peaks` as its ``distance``
        argument. Defaults to ``200``.
    peak_width : int
        Minimum width, in samples, of a detected cast turnaround.
        Forwarded to `scipy.signal.find_peaks` as its ``width`` argument.
        Defaults to ``200``.
    peak_prominence : float
        Minimum prominence, in dbar, of a detected cast turnaround.
        Forwarded to `scipy.signal.find_peaks` as its ``prominence``
        argument. Defaults to ``25.0``.
    trough_prominence : float
        Minimum prominence, in dbar, of a detected surface point (i.e. a
        trough in sea pressure). Forwarded to `scipy.signal.find_peaks`
        as its ``prominence`` argument. Defaults to ``2.0``.
    trough_distance : int
        Minimum number of samples between detected surface points.
        Forwarded to `scipy.signal.find_peaks` as its ``distance``
        argument. Defaults to ``5``.
    trough_width : int
        Minimum width, in samples, of a detected surface point.
        Forwarded to `scipy.signal.find_peaks` as its ``width`` argument.
        Defaults to ``5``.
    run_length : int
        Number of consecutive samples of consistent pressure change
        required to confirm a real descent/ascent, guarding against
        noise. Defaults to ``4``.
    min_pressure_change : float
        Minimum pressure change, in dbar, between consecutive samples
        for `run_length`'s consistency check. Defaults to ``0.01``.
    apply_speed_threshold : bool
        Whether to additionally require a minimum profiling speed
        (computed from sea pressure and time), rather than relying only
        on peak/trough detection. Defaults to ``False``.
    min_speed : float
        Minimum profiling speed, in dbar/s, required when
        `apply_speed_threshold` is ``True``. Defaults to ``0.2``.
    speed_threshold_direction : {"up", "down", "both"}
        Forwarded to `profinder.find_profiles`'s own ``direction``
        argument. Only relevant when `apply_speed_threshold` is
        ``True``: `profinder` always identifies both the downcast and
        upcast of every turnaround regardless of this setting, but only
        refines the boundary of the segment(s) named here using the
        `min_speed` check -- the other segment keeps its unrefined
        peak/trough boundary. Independent of `direction` below, which
        controls which cast(s) are written out as separate profiles, not
        which are identified or speed-refined. Defaults to ``"both"``.
    direction : {"up", "down", "both"}
        Which cast direction(s) to write out as separate profiles (see
        `ctd_processing.process.profiles.resolve_cast_slices`).
        ``"down"``/``"up"`` extracts only the downcast/upcast segment of
        each identified turnaround; ``"both"`` extracts the downcast and
        upcast as two separate profiles. In every case, the dwell between
        a turnaround's ``down_end`` and ``up_start`` (e.g. time spent at
        the bottom of a cast) is never included in an extracted profile.
        Defaults to ``"down"``.
    missing : {"raise", "drop"}
        How to handle non-finite values in the pressure record:
        ``"raise"`` raises an error, ``"drop"`` drops them before
        detection. Defaults to ``"drop"``, not `profinder`'s own
        ``"raise"`` default -- `sea_pressure` routinely has NaNs from
        upstream `remove_holds` processing (see
        `RawChannelSettings.remove_holds`), so treating that as an
        error would be routine pipeline behavior rejecting itself.
    """

    model_config = ConfigDict(extra="forbid")

    apply_smoothing: bool = False
    window_length: int = 9
    polyorder: int = 2
    min_pressure: float = -1.0
    peak_height: float = 25.0
    peak_distance: int = 200
    peak_width: int = 200
    peak_prominence: float = 25.0
    trough_prominence: float = 2.0
    trough_distance: int = 5
    trough_width: int = 5
    run_length: int = 4
    min_pressure_change: float = 0.01
    apply_speed_threshold: bool = False
    min_speed: float = 0.2
    speed_threshold_direction: Literal["up", "down", "both"] = "both"
    direction: Literal["up", "down", "both"] = "down"
    missing: Literal["raise", "drop"] = "drop"


class CTLagSettings(BaseModel):
    """Settings for the conductivity/temperature (CT) lag correction.

    See `ctd_processing.process.ct_lag`, which applies these to a
    dataset's `electrical_conductivity`, `temperature`,
    and `sea_pressure` channels once profiles have been identified (see
    `ProfileSettings`). Conductivity and temperature sensors respond at
    slightly different speeds, so the two channels sample the same water
    parcel at slightly different times; shifting conductivity by a small
    number of samples relative to temperature reduces the resulting
    salinity spiking. This computes a single shift for the whole
    deployment, not one per profile.

    Attributes
    ----------
    enabled : bool
        Whether to compute and apply this correction. Defaults to
        ``False``. To apply a known, fixed shift instead of computing one
        here, set `RawChannelSettings.shift` on the
        `electrical_conductivity` channel directly rather than
        enabling this -- there is no manual-value option here.
    sea_pressure_min : float or None
        If set, only samples with `sea_pressure` greater than or equal to
        this value (in dbar) feed the lag search. Optional; defaults to
        ``None``, meaning no lower bound. Useful for excluding unreliable
        near-surface measurements or sections with highly variable
        profiling speed.
    sea_pressure_max : float or None
        If set, only samples with `sea_pressure` less than or equal to
        this value (in dbar) feed the lag search. Optional; defaults to
        ``None``, meaning no upper bound.
    window_length : int
        Width, in samples, of the running-mean high-pass filter used to
        isolate salinity spiking when scoring a candidate lag. Must be
        odd. Defaults to ``21``.
    min_lag : int
        Smallest candidate lag, in samples, to search over. Defaults to
        ``-20``.
    max_lag : int
        Largest candidate lag, in samples, to search over. Defaults to
        ``20``. Must be greater than or equal to `min_lag`.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    sea_pressure_min: float | None = None
    sea_pressure_max: float | None = None
    window_length: int = 21
    min_lag: int = -20
    max_lag: int = 20

    @model_validator(mode="after")
    def _validate_search_range(self) -> "CTLagSettings":
        """Require an odd `window_length` and `min_lag` <= `max_lag`."""
        if self.window_length % 2 == 0:
            raise ValueError(
                f"window_length must be odd; got {self.window_length}."
            )
        if self.min_lag > self.max_lag:
            raise ValueError(
                f"min_lag ({self.min_lag}) must be <= max_lag ({self.max_lag})."
            )
        return self


class GeolocationSettings(BaseModel):
    """Settings for attaching a position to each extracted profile.

    See `ctd_processing.process.geolocation.attach_geolocation`, applied to
    each profile's own extracted `Dataset` after it is sliced out of the
    full deployment (see `ctd_processing.process.process_profile`).
    Every profile is given a `profile_start_time`/`profile_end_time` (its
    first/last `time` samples) and a `latitude`/`longitude` position,
    recorded in that profile's `Dataset.metadata`. This step is not
    optional -- exactly one of `external_dataset_path` or
    `reference_latitude`/`reference_longitude` must be configured; there is
    no numeric fallback, so an unconfigured project fails validation rather
    than silently attaching a placeholder position.

    Attributes
    ----------
    external_dataset_path : pathlib.Path or None
        Path to a netCDF file holding a `latitude_variable`/
        `longitude_variable` time series to interpolate each profile's
        position from, evaluated at the profile's start time (its
        "canonical" time). `latitude_variable`/`longitude_variable` must
        share their dimension with `time_variable`. Resolved the same way
        as `ctd_processing.config.PathsSettings.rsk_directory` when
        relative. Optional; defaults to ``None``. Mutually exclusive with
        `reference_latitude`/`reference_longitude`.
    latitude_variable : str
        Name of the latitude variable in `external_dataset_path`, in
        decimal degrees north. Defaults to ``"latitude"``.
    longitude_variable : str
        Name of the longitude variable in `external_dataset_path`, in
        decimal degrees east. Defaults to ``"longitude"``.
    time_variable : str
        Name of the time coordinate in `external_dataset_path` that
        `latitude_variable`/`longitude_variable` are indexed by. Defaults
        to ``"time"``.
    reference_latitude : float or None
        A fixed latitude, in decimal degrees north (``-90`` to ``90``),
        used for every profile instead of interpolating from an external
        dataset. Optional; defaults to ``None``. Mutually exclusive with
        `external_dataset_path`; must be set together with
        `reference_longitude`.
    reference_longitude : float or None
        A fixed longitude, in decimal degrees east (``-180`` to ``180``),
        used for every profile instead of interpolating from an external
        dataset. Optional; defaults to ``None``. Mutually exclusive with
        `external_dataset_path`; must be set together with
        `reference_latitude`.

    Raises
    ------
    ValueError
        If both `external_dataset_path` and a reference position are set,
        if only one of `reference_latitude`/`reference_longitude` is set,
        or if neither an external dataset nor a complete reference
        position is set.
    """

    model_config = ConfigDict(extra="forbid")

    external_dataset_path: Path | None = None
    latitude_variable: str = "latitude"
    longitude_variable: str = "longitude"
    time_variable: str = "time"
    reference_latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    reference_longitude: float | None = Field(default=None, ge=-180.0, le=180.0)

    @model_validator(mode="after")
    def _validate_exactly_one_source(self) -> "GeolocationSettings":
        """Require exactly one geolocation source to be configured."""
        has_external = self.external_dataset_path is not None
        has_lat = self.reference_latitude is not None
        has_lon = self.reference_longitude is not None
        if has_external and (has_lat or has_lon):
            raise ValueError(
                "Set either external_dataset_path or "
                "reference_latitude/reference_longitude, not both."
            )
        if has_lat != has_lon:
            raise ValueError(
                "reference_latitude and reference_longitude must be set "
                "together."
            )
        if not has_external and not has_lat:
            raise ValueError(
                "Set either external_dataset_path or "
                "reference_latitude/reference_longitude."
            )
        return self


class DerivedVariablesSettings(BaseModel):
    """Settings for TEOS-10 derived variables computed via `gsw`.

    See `ctd_processing.process.derived_variables.compute_derived_variables`,
    applied to each profile's `Dataset` after a position has been attached
    to it (see `GeolocationSettings`) but before it is written out (see
    `ctd_processing.process.process_profile`). Every quantity here is
    derived from the profile's `electrical_conductivity`,
    `temperature`, and `sea_pressure` channels, plus its
    `latitude`/`longitude` position. Disabling a field only omits that
    channel from the output -- the underlying practical/absolute
    salinity and conservative temperature chain is always computed
    internally regardless of which fields are enabled, since later
    quantities (e.g. `potential_density`) depend on it.

    Attributes
    ----------
    z : bool
        Whether to compute height, in meters (`gsw.z_from_p`; negative in
        the ocean), added under the channel key ``"z"``. Defaults to
        ``True``.
    practical_salinity : bool
        Whether to compute practical salinity (`gsw.SP_from_C`), added
        under ``"practical_salinity"`` -- overwriting any channel already
        present there (e.g. from the instrument's own onboard
        computation), since this one incorporates whatever upstream
        raw-channel/CT-lag corrections were applied, which an
        onboard-computed value would not. Defaults to ``True``.
    absolute_salinity : bool
        Whether to compute absolute salinity (`gsw.SA_from_SP`), added
        under ``"absolute_salinity"`` (overwriting, as above). Defaults
        to ``True``.
    conservative_temperature : bool
        Whether to compute conservative temperature (`gsw.CT_from_t`),
        added under ``"conservative_temperature"``. Defaults to ``True``.
    potential_density : bool
        Whether to compute potential density anomaly referenced to 0 dbar
        (`gsw.sigma0`), added under ``"density_anomaly"`` (overwriting,
        as above) -- the same channel key already used for an
        instrument's own onboard-computed potential density anomaly (see
        `ctd_processing.process.cf_channels`). Defaults to ``True``.
    potential_temperature : bool
        Whether to compute potential temperature referenced to 0 dbar
        (`gsw.pt0_from_t`), added under ``"potential_temperature"``
        (overwriting, as above). Defaults to ``False``.
    sound_speed : bool
        Whether to compute the speed of sound in sea water
        (`gsw.sound_speed`), added under
        ``"speed_of_sound"`` (overwriting, as above).
        Defaults to ``False``.
    density : bool
        Whether to compute in-situ density (`gsw.rho`), added under
        ``"sea_water_density"``. Defaults to ``False``.
    spiciness : bool
        Whether to compute spiciness referenced to 0 dbar
        (`gsw.spiciness0`), added under ``"spiciness"``. No CF standard
        name exists for this quantity. Defaults to ``False``.
    freezing_point : bool
        Whether to compute the Conservative Temperature at which seawater
        freezes (`gsw.CT_freezing`, with ``saturation_fraction=0`` --
        dissolved-air effects on the freezing point are not modeled),
        added under ``"freezing_point"``. No CF standard name exists for
        this quantity. Defaults to ``False``.
    thermal_expansion : bool
        Whether to compute the thermal expansion coefficient
        (`gsw.alpha`), added under ``"thermal_expansion_coefficient"``.
        No CF standard name exists for this quantity. Defaults to
        ``False``.
    haline_contraction : bool
        Whether to compute the haline contraction coefficient
        (`gsw.beta`), added under ``"haline_contraction_coefficient"``.
        No CF standard name exists for this quantity. Defaults to
        ``False``.
    oxygen_concentration : bool
        Whether to derive an oxygen concentration from a measured
        ``dissolved_oxygen_saturation`` channel and the oxygen
        solubility at 100% air-sea equilibrium (`gsw.O2sol`), added
        under ``"oxygen_concentration_from_saturation"``. Unlike every
        other field here, this one's input is not present on every
        dataset -- if enabled and `dataset.channels` has no
        ``dissolved_oxygen_saturation`` channel,
        `ctd_processing.process.derived_variables.compute_derived_variables`
        raises ``ValueError`` naming it, rather than silently skipping
        it. Defaults to ``False``.
    """

    model_config = ConfigDict(extra="forbid")

    z: bool = True
    practical_salinity: bool = True
    absolute_salinity: bool = True
    conservative_temperature: bool = True
    potential_density: bool = True
    potential_temperature: bool = False
    sound_speed: bool = False
    density: bool = False
    spiciness: bool = False
    freezing_point: bool = False
    thermal_expansion: bool = False
    haline_contraction: bool = False
    oxygen_concentration: bool = False


class DespikeChannelOverride(BaseModel):
    """One channel's override of the project-wide `DespikeSettings`.

    The value of `ChannelSettings.despiking` (a
    ``[process.channels.<name>.despiking]`` table), kept as a separate
    key from `ChannelSettings.despike` (the plain enable/disable
    ``bool``) so the two can't collide as a single TOML key that's
    sometimes a scalar and sometimes a table. See `ChannelSettings.despike`
    for how this combines with the project-wide defaults, and for
    despike timing.

    Attributes
    ----------
    threshold : float or None
        Override of the project-wide `DespikeSettings.threshold` for
        this channel. Optional; defaults to ``None``, meaning inherit.
    window_length : int or None
        Override of the project-wide `DespikeSettings.window_length` for
        this channel. Not validated here -- an even value only fails
        once merged and resolved (see `resolve_despike_settings`).
        Optional; defaults to ``None``, meaning inherit.
    max_iterations : int or None
        Override of the project-wide `DespikeSettings.max_iterations`
        for this channel. Optional; defaults to ``None``, meaning
        inherit.
    """

    model_config = ConfigDict(extra="forbid")

    threshold: float | None = None
    window_length: int | None = None
    max_iterations: int | None = None


class DespikeSettings(BaseModel):
    """Project-wide default settings for despiking with a rolling median.

    See `ctd_processing.process.despike` and
    `ctd_processing.config.resolve_despike_settings`. A channel is
    smoothed with a rolling median filter of `window_length` to get a
    "reference" series; points whose residual against that reference
    exceeds `threshold` standard deviations are replaced with NaN. This
    repeats, up to `max_iterations` times, stopping early the first pass
    that finds no new spikes -- removing large spikes can unmask smaller
    ones the previous pass's median/std missed.

    These are project-wide defaults only; whether a given channel is
    despiked at all, and any per-channel overrides of these values, are
    configured on that channel's `ChannelSettings.despike`/`despiking`
    (see `ProcessSettings.channels`).

    Attributes
    ----------
    threshold : float
        Number of standard deviations (of the residual against the
        rolling median) a point must exceed to be flagged as a spike.
        Defaults to ``2.0``, matching `pyrsktools.RSK.despike`'s own
        default.
    window_length : int
        Width, in samples, of the rolling median filter. Must be odd.
        Defaults to ``3``, matching `pyrsktools.RSK.despike`'s own
        default.
    max_iterations : int
        Maximum number of detect-and-replace passes to run. Defaults to
        ``5`` -- genuinely iterative by default, since a pass that finds
        nothing new stops early regardless.
    """

    model_config = ConfigDict(extra="forbid")

    threshold: float = 2.0
    window_length: int = 3
    max_iterations: int = 5

    @model_validator(mode="after")
    def _validate_window_length(self) -> "DespikeSettings":
        """Require an odd `window_length`."""
        if self.window_length % 2 == 0:
            raise ValueError(
                f"window_length must be odd; got {self.window_length}."
            )
        return self


class ChannelSettings(BaseModel):
    """Per-channel output settings: despiking and output precision.

    An entry in `ProcessSettings.channels`, keyed by channel name -- the
    same key namespace as `ProcessSettings.raw_channels`: raw channels
    (e.g. ``"temperature"``) and derived-variable channels (e.g.
    ``"practical_salinity"``, matching
    `ctd_processing.process.derived_variables`'s output keys). A channel
    with no entry here uses every default below.

    Attributes
    ----------
    despike : bool
        Whether to despike this channel. ``False`` (the default): not
        despiked. ``True``: despiked using the project-wide
        `ProcessSettings.despiking` defaults, as overridden by
        `despiking` below (see `resolve_despike_settings`). Despiking
        runs as soon as a channel exists -- for raw channels, before any
        derived variable is computed from them; for derived channels,
        immediately after that quantity is computed and before it feeds
        the next one (e.g. `practical_salinity` is despiked before it's
        used to compute `absolute_salinity`).
    despiking : DespikeChannelOverride
        This channel's overrides of the project-wide
        `ProcessSettings.despiking` defaults' `threshold`/
        `window_length`/`max_iterations` (a
        ``[process.channels.<name>.despiking]`` table). Only takes
        effect when `despike` is ``True`` -- a channel with `despike`
        left at ``False`` is not despiked regardless of what's set here.
        Optional; every field defaults to ``None``, meaning inherit the
        project-wide default.
    output_dtype : str or None
        The numpy floating-point dtype (e.g. ``"float32"``,
        ``"float64"``) this channel's data is cast to when written.
        Optional; defaults to ``None``, meaning use
        `ProcessSettings.output_dtype`, the project-wide default. See
        `resolve_output_dtype`.
    """

    model_config = ConfigDict(extra="forbid")

    despike: bool = False
    despiking: DespikeChannelOverride = Field(
        default_factory=DespikeChannelOverride
    )
    output_dtype: str | None = None

    @field_validator("output_dtype")
    @classmethod
    def _validate_output_dtype(cls, value: str | None) -> str | None:
        """Require `output_dtype`, if set, to be a floating numpy dtype."""
        if value is not None:
            _check_output_dtype(value)
        return value


class NetcdfCompressionSettings(BaseModel):
    """Zlib compression settings applied to floating-point netCDF variables.

    Shared, one source of truth, by `ProcessSettings.netcdf_compression`
    (profile-level netCDF; see
    `ctd_processing.process.save_netcdf.write_netcdf`) and
    `BinSettings.netcdf_compression` (deployment-level binned netCDF; see
    `ctd_processing.bin.save.save_binned_dataset`) -- both write via
    xarray + h5netcdf, so both compress the same way, via
    `ctd_processing.process.save_netcdf.netcdf_compression_encoding`. Only
    ever applies to floating-point data variables; every coordinate
    (``time``, and for binned files also the binning channel etc.) is
    always left uncompressed regardless of these settings -- a monotonic
    timestamp array compresses poorly and isn't worth the encoding
    complexity.

    Attributes
    ----------
    enabled : bool
        Whether to compress floating-point variables at all. Defaults to
        ``True``, matching this package's previous hardcoded behavior.
        ``False`` writes fully uncompressed netCDF files.
    complevel : int
        Zlib compression level passed to h5netcdf, ``0`` (fastest, no
        compression) to ``9`` (slowest, most compression). Only takes
        effect when `enabled` is ``True``. Defaults to ``4``, this
        package's previous hardcoded value.
    shuffle : bool
        Whether to apply HDF5's shuffle filter before compressing (groups
        like-significance bytes across values -- typically improves
        compression for floats at negligible cost). Only takes effect
        when `enabled` is ``True``. Defaults to ``True``, this package's
        previous hardcoded value.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    complevel: int = Field(default=4, ge=0, le=9)
    shuffle: bool = True


class ParquetCompressionSettings(BaseModel):
    """zstd compression settings for profile Parquet output.

    See `ctd_processing.process.save_parquet.write_parquet`. Byte-stream-
    split encoding for floating-point columns is unaffected by `enabled`
    -- it is a column-encoding transform, independent of the zstd codec
    choice, not a second compression algorithm.

    Attributes
    ----------
    enabled : bool
        Whether to zstd-compress at all. Defaults to ``True``, matching
        this package's previous hardcoded behavior. ``False`` writes
        with ``compression="none"``.
    level : int or None
        zstd compression level, ``1`` (fastest, least compression) to
        ``22`` (slowest, most compression). Only takes effect when
        `enabled` is ``True``. Defaults to ``None``, meaning pyarrow's/
        zstd's own default level -- matching this package's previous
        behavior, which never set a level explicitly.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    level: int | None = Field(default=None, ge=1, le=22)


class ZarrCompressionSettings(BaseModel):
    """Blosc compression settings for the binned Zarr store.

    See `ctd_processing.bin.save.save_binned_dataset`, which applies
    these to every floating-point data variable via
    `zarr.codecs.BloscCodec`; every coordinate is always left
    uncompressed, matching `NetcdfCompressionSettings`'s treatment of
    ``time``. There is no previous hardcoded behavior to preserve here --
    `save_binned_dataset` previously wrote Zarr with no explicit
    compressor at all, silently inheriting whatever zarr's own library
    default happened to be (as of zarr 3.3.0, an unconfigured, un-shuffled
    ``ZstdCodec(level=0)``). These settings replace that accidental
    default with an explicit, deliberately chosen one.

    Attributes
    ----------
    enabled : bool
        Whether to compress floating-point variables at all. Defaults to
        ``True``. ``False`` writes with an explicit empty ``compressors``
        list (zarr's raw ``BytesCodec`` only).
    cname : {"blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"}
        The Blosc-internal codec. Only takes effect when `enabled` is
        ``True``. Defaults to ``"zstd"``.
    clevel : int
        Blosc compression level, ``0`` (fastest, no compression) to ``9``
        (slowest, most compression). Only takes effect when `enabled` is
        ``True``. Defaults to ``5``, Blosc's own conventional default.
    shuffle : {"noshuffle", "shuffle", "bitshuffle"} or None
        Byte-shuffle filter applied before compressing. Only takes effect
        when `enabled` is ``True``. Defaults to ``None``, meaning
        `zarr.codecs.BloscCodec` picks automatically based on each
        variable's dtype.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    cname: Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"] = "zstd"
    clevel: int = Field(default=5, ge=0, le=9)
    shuffle: Literal["noshuffle", "shuffle", "bitshuffle"] | None = None


class ProcessSettings(BaseModel):
    """Settings specific to the ``process`` command.

    Attributes
    ----------
    read_channels : list[str]
        Restrict which channels are extracted from the deployment's raw
        ``.rsk`` data to exactly these RBR channel names -- the raw
        `pyrsktools.datatypes.Channel.longName` values the data is
        actually saved under on the instrument (e.g. ``"temperature"``,
        ``"conductivity"``), *not*
        `ctd_processing.process.cf_channels.channel_key_for_longname`'s
        derived key (the same keys `raw_channels` is keyed by -- e.g.
        ``"electrical_conductivity"`` for RBR's ``"conductivity"``).
        Applied while building the `Dataset` from the ``.rsk`` file (see
        `ctd_processing.process.build.build_dataset`), before any other
        processing step, so a channel excluded here is never read into
        memory at all. If a requested channel is not present in the
        deployment -- unrecognized, or reported by the instrument but not
        logged in this schedule -- that is an error, not a silent skip.
        Defaults to an empty list, meaning every channel with data
        present is extracted (the unfiltered default).
    raw_channels : dict[str, RawChannelSettings]
        Per-raw-channel processing settings, keyed by
        `ctd_processing.process.cf_channels.channel_key_for_longname`'s
        result for that channel -- the same short identifier a channel
        ends up under in `ctd_processing.process.dataset.Dataset.channels`
        (e.g. ``"temperature"``), not the raw pyrsktools field
        name and not the full CF `standard_name`. A channel needs no
        entry here at all; an absent entry just means
        `RawChannelSettings`'s defaults apply to it. Defaults to an empty
        dict.
    atmospheric_pressure : float or None
        Constant atmospheric pressure, in dbar, subtracted from the
        `absolute_pressure` channel to (re)compute `sea_pressure`,
        overwriting any `sea_pressure` channel already present in the
        dataset (see `ctd_processing.process.sea_pressure.
        compute_sea_pressure`). Optional; defaults to ``None``, meaning
        the dataset's own `sea_pressure` channel is trusted and used
        as-is -- RBR's Ruskin software commonly derives this itself from
        `absolute_pressure` and a per-deployment atmospheric reference
        before the ``.rsk`` file is even read, so there is nothing to
        recompute unless a different atmospheric pressure is wanted.
        With ``None``, it is an error for the dataset to have no
        `sea_pressure` channel at all.
    profiles : ProfileSettings
        Settings for identifying individual profiles from `sea_pressure`
        (see `ProfileSettings`). Optional; every field has a default.
    ct_lag : CTLagSettings
        Settings for the conductivity/temperature lag correction (see
        `CTLagSettings`), applied after `profiles` are identified.
        Optional; every field has a default, and `CTLagSettings.enabled`
        defaults to ``False``.
    profile_format : {"netcdf", "parquet"}
        File format for extracted profile files written to
        ``paths.profiles_directory`` (see
        `ctd_processing.process.save.save_profile`). ``"parquet"``
        (the default) is written per `parquet_compression`, with
        byte-stream-split encoding for float columns -- the better fit
        for this fast, size-sensitive intermediate stage. ``"netcdf"``
        writes CF-compliant files (``units``/``long_name``/
        ``standard_name`` as variable attributes, project/deployment
        metadata and processing history as global attributes) via
        `xarray` and `h5netcdf`, compressed per `netcdf_compression` --
        better for long-term self-description and interop with CF-aware
        tools (e.g. ERDDAP, OceanSITES). Defaults to ``"parquet"``.
    netcdf_compression : NetcdfCompressionSettings
        Compression settings applied when `profile_format` is
        ``"netcdf"`` (see `NetcdfCompressionSettings`). Optional; every
        field has a default. Ignored when `profile_format` is
        ``"parquet"``.
    parquet_compression : ParquetCompressionSettings
        Compression settings applied when `profile_format` is
        ``"parquet"`` (see `ParquetCompressionSettings`). Optional;
        every field has a default. Ignored when `profile_format` is
        ``"netcdf"``.
    geolocation : GeolocationSettings
        Settings for attaching a position to each extracted profile (see
        `GeolocationSettings`). Required -- unlike every other field of
        this class, it has no default, since `GeolocationSettings` itself
        has no valid unconfigured state.
    derived_variables : DerivedVariablesSettings
        Settings for TEOS-10 derived variables computed via `gsw` (see
        `DerivedVariablesSettings`), applied to each profile right after
        `geolocation` and before it is written out. Optional; every field
        has a default.
    despiking : DespikeSettings
        Project-wide default despike settings (see `DespikeSettings`).
        Whether a given channel is actually despiked, and any per-channel
        overrides of these defaults, are configured on that channel's
        entry in `channels` instead. Optional; every field has a default.
    channels : dict[str, ChannelSettings]
        Per-channel output settings -- despiking and `output_dtype` --
        keyed the same way as `raw_channels` (see `ChannelSettings`). A
        channel needs no entry here at all; an absent entry just means
        `ChannelSettings`'s defaults apply to it (not despiked, written
        in `output_dtype`). Defaults to an empty dict.
    output_dtype : str
        Project-wide default numpy floating-point dtype (e.g.
        ``"float32"``, ``"float64"``) every channel is cast to when
        written, unless overridden by that channel's own
        `ChannelSettings.output_dtype` (see `resolve_output_dtype`).
        Defaults to ``"float32"``.
    """

    model_config = ConfigDict(extra="forbid")

    read_channels: list[str] = Field(default_factory=list)
    raw_channels: dict[str, RawChannelSettings] = Field(default_factory=dict)
    atmospheric_pressure: float | None = None
    profiles: ProfileSettings = Field(default_factory=ProfileSettings)
    ct_lag: CTLagSettings = Field(default_factory=CTLagSettings)
    profile_format: Literal["netcdf", "parquet"] = "parquet"
    netcdf_compression: NetcdfCompressionSettings = Field(
        default_factory=NetcdfCompressionSettings
    )
    parquet_compression: ParquetCompressionSettings = Field(
        default_factory=ParquetCompressionSettings
    )
    geolocation: GeolocationSettings
    derived_variables: DerivedVariablesSettings = Field(
        default_factory=DerivedVariablesSettings
    )
    despiking: DespikeSettings = Field(default_factory=DespikeSettings)
    channels: dict[str, ChannelSettings] = Field(default_factory=dict)
    output_dtype: str = "float32"

    @field_validator("output_dtype")
    @classmethod
    def _validate_output_dtype(cls, value: str) -> str:
        """Require `output_dtype` to be a floating numpy dtype."""
        _check_output_dtype(value)
        return value


class BinSettings(BaseModel):
    """Settings specific to the ``bin`` command.

    See `ctd_processing.process.binning`, which bins every profile passed
    to ``bin`` onto a common grid along `channel`, averages every other
    channel within each bin, and stacks the results along a new
    ``profile`` dimension.

    Attributes
    ----------
    channel : str
        The channel key to bin by -- any key valid in
        `ctd_processing.process.dataset.Dataset.channels` (e.g. ``"z"``,
        ``"sea_pressure"``), not necessarily one of `Dataset.channels`
        default set. Defaults to ``"z"`` (height, in meters, negative in
        the ocean -- see
        `ctd_processing.config.DerivedVariablesSettings.z`).
    step : float
        Bin edge spacing, in `channel`'s units. May be negative for a
        decreasing grid, e.g. binning a downcast by `channel` ``"z"``,
        which is negative and decreases with depth. Must not be ``0``.
        Defaults to ``1.0``.
    first : float or None
        The first bin edge. Optional; if unset (the default), computed
        from the data actually being binned: the minimum of `channel`
        across every profile if `step` is positive, or the maximum if
        `step` is negative.
    last : float or None
        The bin edge to reach or pass. Optional; if unset (the default),
        computed the opposite way from `first`: the maximum of `channel`
        across every profile if `step` is positive, or the minimum if
        `step` is negative. Edges are generated as ``first``, ``first +
        step``, ``first + 2 * step``, ... , stopping at the first edge
        that reaches or passes `last` -- so, unlike `numpy.arange`, the
        final bin may slightly overshoot `last`. Must be greater than or
        equal to `first` if `step` is positive, or less than or equal to
        `first` if `step` is negative.
    output_format : {"netcdf", "zarr"}
        File format for the combined, binned dataset written to
        ``paths.binned_directory`` (see
        `ctd_processing.bin.save.save_binned_dataset`). ``"netcdf"``
        (the default) writes a CF-compliant file via `xarray` and
        `h5netcdf`, matching `ProcessSettings.profile_format`'s
        ``"netcdf"`` option and compressed per `netcdf_compression`.
        ``"zarr"`` writes a Zarr store instead, compressed per
        `zarr_compression`.
    netcdf_compression : NetcdfCompressionSettings
        Compression settings applied when `output_format` is
        ``"netcdf"`` (see `NetcdfCompressionSettings`). Optional; every
        field has a default. Ignored when `output_format` is ``"zarr"``.
    zarr_compression : ZarrCompressionSettings
        Compression settings applied when `output_format` is ``"zarr"``
        (see `ZarrCompressionSettings`). Optional; every field has a
        default. Ignored when `output_format` is ``"netcdf"``.

    Raises
    ------
    ValueError
        If `step` is ``0``, or if `first`/`last` are both given and `last`
        is on the wrong side of `first` for `step`'s sign.
    """

    model_config = ConfigDict(extra="forbid")

    channel: str = "z"
    step: float = 1.0
    first: float | None = None
    last: float | None = None
    output_format: Literal["netcdf", "zarr"] = "netcdf"
    netcdf_compression: NetcdfCompressionSettings = Field(
        default_factory=NetcdfCompressionSettings
    )
    zarr_compression: ZarrCompressionSettings = Field(
        default_factory=ZarrCompressionSettings
    )

    @model_validator(mode="after")
    def _validate_step_and_range(self) -> "BinSettings":
        """Require a nonzero `step` and, if both are given, a valid range."""
        if self.step == 0:
            raise ValueError("step must not be 0.")
        if self.first is not None and self.last is not None:
            if self.step > 0 and self.last < self.first:
                raise ValueError(
                    f"last ({self.last}) must be >= first ({self.first}) "
                    "when step > 0."
                )
            if self.step < 0 and self.last > self.first:
                raise ValueError(
                    f"last ({self.last}) must be <= first ({self.first}) "
                    "when step < 0."
                )
        return self


class InstrumentSettings(BaseModel):
    """Per-instrument overrides of `ProcessSettings`, keyed by serial number.

    An instrument's serial number is only known once a deployment's
    ``.rsk`` file has actually been read (see
    `ctd_processing.process.build.build_dataset`); it is never inferred
    from a filename. `Settings.instruments` is keyed by that serial
    number, as a string, so an override here follows a physical
    instrument across every deployment it appears in.

    Attributes
    ----------
    process : dict[str, Any]
        A partial, TOML-table-shaped `ProcessSettings` override. Only the
        fields given here are overridden; every other field falls back to
        the project-level `Settings.process` (see
        :func:`resolve_process_settings`). Nested tables (e.g.
        ``raw_channels.<name>``, ``profiles``) merge field-by-field rather
        than replacing the whole table. Not validated as `ProcessSettings`
        until merged with the project-level settings it overrides.
        Defaults to an empty dict, i.e. no overrides.
    """

    model_config = ConfigDict(extra="forbid")

    process: dict[str, Any] = Field(default_factory=dict)


class DeploymentSettings(BaseModel):
    """Per-deployment overrides of `ProcessSettings`, keyed by ``.rsk`` stem.

    `Settings.deployments` is keyed by a deployment's ``.rsk`` filename
    stem (i.e. the filename without its extension), e.g.
    ``"243188_20260809_0304"`` for ``243188_20260809_0304.rsk``. These
    overrides win over any matching `InstrumentSettings` override, which
    in turn wins over the project-level `Settings.process` (see
    :func:`resolve_process_settings`).

    Attributes
    ----------
    process : dict[str, Any]
        A partial, TOML-table-shaped `ProcessSettings` override. Only the
        fields given here are overridden; every other field falls back to
        the project-level `Settings.process` (and any matching
        `InstrumentSettings`). Nested tables (e.g.
        ``raw_channels.<name>``, ``profiles``) merge field-by-field rather
        than replacing the whole table. Not validated as `ProcessSettings`
        until merged with the settings it overrides. Defaults to an empty
        dict, i.e. no overrides.
    """

    model_config = ConfigDict(extra="forbid")

    process: dict[str, Any] = Field(default_factory=dict)


class Settings(BaseSettings):
    """Runtime configuration for ctd_processing.

    This model is the single extension point for every configuration
    option that ``process``, ``bin``, and ``concatenate`` need. Per
    repository convention, every field added here must have a
    corresponding, documented entry in the bundled starter template at
    ``ctd_processing/cli/templates/config.toml``.

    Attributes
    ----------
    project : ProjectSettings
        Project-level metadata attached to every output file, e.g.
        `name`. Optional; every field of `ProjectSettings` has a
        default.
    paths : PathsSettings
        Directory locations for the project's pipeline stages, e.g.
        `rsk_directory`. Required, since none of its fields have a
        default.
    process : ProcessSettings
        Settings specific to the ``process`` command, e.g. `raw_channels`.
        Required, since `ProcessSettings.geolocation` has no default.
    bin : BinSettings
        Settings specific to the ``bin`` command, e.g. `channel`/`step`.
        Optional; every field of `BinSettings` has a default.
    instruments : dict[str, InstrumentSettings]
        Per-instrument overrides of `process`, keyed by instrument serial
        number (see `InstrumentSettings`). Optional; defaults to an empty
        dict, i.e. no overrides. Resolve the effective `ProcessSettings`
        for a given instrument/deployment with
        :func:`resolve_process_settings`.
    deployments : dict[str, DeploymentSettings]
        Per-deployment overrides of `process`, keyed by ``.rsk`` filename
        stem (see `DeploymentSettings`). Optional; defaults to an empty
        dict, i.e. no overrides. These win over any matching
        `instruments` override. Resolve the effective `ProcessSettings`
        for a given instrument/deployment with
        :func:`resolve_process_settings`.
    """

    model_config = SettingsConfigDict(extra="forbid")

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    paths: PathsSettings
    process: ProcessSettings
    bin: BinSettings = Field(default_factory=BinSettings)
    instruments: dict[str, InstrumentSettings] = Field(default_factory=dict)
    deployments: dict[str, DeploymentSettings] = Field(default_factory=dict)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Restrict settings sources to explicit constructor keyword arguments.

        ctd_processing sources all configuration from an explicitly loaded
        TOML file plus ``--set`` overrides (see :func:`load_settings`), so
        environment variables, ``.env`` files, and secrets directories are
        deliberately excluded to avoid surprising, implicit configuration.

        Parameters
        ----------
        settings_cls : type[BaseSettings]
            The class being instantiated.
        init_settings : PydanticBaseSettingsSource
            Source representing explicit constructor keyword arguments.
        env_settings : PydanticBaseSettingsSource
            Unused.
        dotenv_settings : PydanticBaseSettingsSource
            Unused.
        file_secret_settings : PydanticBaseSettingsSource
            Unused.

        Returns
        -------
        tuple[PydanticBaseSettingsSource, ...]
            Single-element tuple containing only `init_settings`.
        """
        return (init_settings,)


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    r"""Parse ``--set`` command line overrides into a nested dictionary.

    Each pair must have the form ``key=value`` or ``section.key=value``,
    where dotted keys build nested dictionaries. `value` is parsed using
    TOML syntax, so it follows the same conventions as the configuration
    file itself (e.g. strings must be quoted, `true`/`false` for booleans).

    Parameters
    ----------
    pairs : list of str
        Override strings as supplied on the command line, e.g.
        ``["section.key=1", "other=\"text\""]``.

    Returns
    -------
    dict[str, Any]
        Nested dictionary of overrides.

    Raises
    ------
    ValueError
        If a pair does not contain ``=``, or if its value is not valid
        TOML syntax.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        key, separator, raw_value = pair.partition("=")
        if not separator:
            raise ValueError(
                f"Invalid --set value {pair!r}; expected key=value."
            )

        try:
            value = tomllib.loads(f"_ = {raw_value}")["_"]
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"Invalid TOML value in --set {pair!r}: {exc}"
            ) from exc

        target = overrides
        keys = key.split(".")
        for nested_key in keys[:-1]:
            target = target.setdefault(nested_key, {})
        target[keys[-1]] = value

    return overrides


def _deep_merge(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge `overrides` into `base`, returning a new dictionary.

    Parameters
    ----------
    base : dict[str, Any]
        The base dictionary.
    overrides : dict[str, Any]
        Dictionary of overrides to merge on top of `base`. Nested dicts
        are merged recursively; other values overwrite the base value.

    Returns
    -------
    dict[str, Any]
        A new, merged dictionary. Neither input is mutated.
    """
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_overrides(data: dict[str, Any], pairs: list[str]) -> dict[str, Any]:
    """Parse ``--set`` overrides and deep-merge them into `data`.

    Parameters
    ----------
    data : dict[str, Any]
        The base configuration dictionary, e.g. parsed from a TOML file.
    pairs : list of str
        Override strings as supplied on the command line. See
        :func:`parse_overrides`.

    Returns
    -------
    dict[str, Any]
        A new dictionary with `pairs` merged on top of `data`. `data` is
        not mutated.

    Raises
    ------
    ValueError
        If any element of `pairs` is malformed. See :func:`parse_overrides`.
    """
    return _deep_merge(data, parse_overrides(pairs))


def resolve_process_settings(
    settings: Settings,
    *,
    serial_number: str | None = None,
    stem: str | None = None,
) -> ProcessSettings:
    """Resolve the effective `ProcessSettings` for an instrument/deployment.

    Starts from `settings.process` and deep-merges in, in order, the
    matching `settings.instruments` override (if `serial_number` is given
    and present) and then the matching `settings.deployments` override
    (if `stem` is given and present) -- so a deployment override wins
    over an instrument override, which wins over the project-level
    default, for any field they both set. Fields left unset at every
    level keep their `ProcessSettings` default.

    Parameters
    ----------
    settings : Settings
        The loaded project settings.
    serial_number : str or None, optional
        Instrument serial number to look up in `settings.instruments`. If
        ``None`` or not present in `settings.instruments`, no instrument
        override is applied.
    stem : str or None, optional
        Deployment ``.rsk`` filename stem to look up in
        `settings.deployments`. If ``None`` or not present in
        `settings.deployments`, no deployment override is applied.

    Returns
    -------
    ProcessSettings
        The resolved, validated settings.

    Raises
    ------
    pydantic.ValidationError
        If the merged overrides do not form valid `ProcessSettings`.
    """
    merged = settings.process.model_dump(mode="json")

    if serial_number is not None:
        instrument = settings.instruments.get(serial_number)
        if instrument is not None:
            merged = _deep_merge(merged, instrument.process)

    if stem is not None:
        deployment = settings.deployments.get(stem)
        if deployment is not None:
            merged = _deep_merge(merged, deployment.process)

    return ProcessSettings.model_validate(merged)


def resolve_despike_settings(
    process_settings: ProcessSettings,
) -> dict[str, DespikeSettings]:
    """Resolve the effective `DespikeSettings` for every configured channel.

    For each `process_settings.channels` entry whose `despike` is
    ``True``, deep-merges its `despiking`'s non-``None`` override fields
    onto `process_settings.despiking`'s own `threshold`/`window_length`/
    `max_iterations` (the project-wide defaults) and validates the
    result -- the same override mechanism `resolve_process_settings`
    uses for instrument/deployment overrides, applied one level deeper.
    A channel with `despike` left at its default of ``False`` is not
    despiked at all -- regardless of what its `despiking` holds -- and
    so has no entry in the returned dict.

    Parameters
    ----------
    process_settings : ProcessSettings
        Settings providing `despiking` (the project-wide defaults) and
        `channels` (each channel's despike enablement/overrides).

    Returns
    -------
    dict[str, DespikeSettings]
        The resolved, validated despike settings, keyed by channel.

    Raises
    ------
    pydantic.ValidationError
        If a channel's merged override does not form valid
        `DespikeSettings`.
    """
    base = process_settings.despiking.model_dump(mode="json")
    resolved: dict[str, DespikeSettings] = {}
    for name, channel_settings in process_settings.channels.items():
        if not channel_settings.despike:
            continue
        override = channel_settings.despiking.model_dump(
            mode="json", exclude_none=True
        )
        resolved[name] = DespikeSettings.model_validate(
            _deep_merge(base, override)
        )
    return resolved


def resolve_output_dtype(process_settings: ProcessSettings, name: str) -> str:
    """Resolve the effective output dtype for one channel.

    Parameters
    ----------
    process_settings : ProcessSettings
        Settings providing `output_dtype` (the project-wide default) and
        `channels` (each channel's own override, if any).
    name : str
        The channel's key in `process_settings.channels`/
        `ctd_processing.process.dataset.Dataset.channels`.

    Returns
    -------
    str
        `name`'s `ChannelSettings.output_dtype` if set, else
        `process_settings.output_dtype`, the project-wide default.
    """
    channel_settings = process_settings.channels.get(name)
    if channel_settings is not None and channel_settings.output_dtype:
        return channel_settings.output_dtype
    return process_settings.output_dtype


def _validate_declared_overrides(settings: Settings) -> None:
    """Eagerly validate every declared instrument/deployment override.

    Each `settings.instruments`/`settings.deployments` entry is merged
    onto `settings.process` and validated on its own -- independently of
    every other entry -- so a typo'd or invalid override fails fast at
    config-load time rather than only when that specific instrument or
    deployment is later processed. `channels.*.despiking` overrides are
    additionally expanded via `resolve_despike_settings`, so a bad
    per-channel despike override (e.g. an even `window_length`) also
    fails here rather than at first use. This does not validate every
    instrument x deployment combination (which one recorded which
    deployment is only known once a ``.rsk`` file is actually read), so a
    genuine conflict between an instrument and a deployment override can
    still only surface at processing time.

    Parameters
    ----------
    settings : Settings
        The loaded project settings.

    Raises
    ------
    pydantic.ValidationError
        If any declared override does not form valid `ProcessSettings`
        (or, for `channels.*.despiking`, `DespikeSettings`), annotated
        with a note identifying the offending
        ``[instruments.*]``/``[deployments.*]`` table.
    """
    for serial_number in settings.instruments:
        try:
            resolved = resolve_process_settings(
                settings, serial_number=serial_number
            )
            resolve_despike_settings(resolved)
        except ValidationError as exc:
            exc.add_note(f"in [instruments.{serial_number}.process]")
            raise

    for stem in settings.deployments:
        try:
            resolved = resolve_process_settings(settings, stem=stem)
            resolve_despike_settings(resolved)
        except ValidationError as exc:
            exc.add_note(f"in [deployments.{stem}.process]")
            raise


def load_settings(
    config_path: Path | None = None, set_: list[str] | None = None
) -> Settings:
    """Load :class:`Settings` from a TOML configuration file and CLI overrides.

    Parameters
    ----------
    config_path : pathlib.Path or None, optional
        Path to a TOML configuration file to load. If ``None`` (default),
        no file is read.
    set_ : list of str or None, optional
        ``--set key=value`` override strings to apply on top of
        `config_path` (or on top of field defaults if `config_path` is
        ``None``). See :func:`parse_overrides` for syntax.

    Returns
    -------
    Settings
        The loaded and validated settings. Relative
        ``paths.rsk_directory``, ``paths.profiles_directory``,
        ``paths.binned_directory``, (when given) ``paths.concatenated_file``,
        ``paths.log_file``, and ``paths.error_log_file``, and (when given)
        ``process.geolocation.external_dataset_path`` values are resolved
        against the directory containing `config_path` (or the current
        working directory if `config_path` is ``None``), so a project's
        config resolves correctly regardless of where it is loaded from.

    Raises
    ------
    FileNotFoundError
        If `config_path` is given but does not point to an existing file.
    ValueError
        If `set_` contains a malformed override. See :func:`parse_overrides`.
    pydantic.ValidationError
        If the merged configuration contains unknown or invalid keys, or
        if any declared ``[instruments.*]``/``[deployments.*]`` override
        does not form valid `ProcessSettings` once merged onto `process`
        (see :func:`resolve_process_settings`).
    """
    data: dict[str, Any] = {}
    if config_path is not None:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    if set_:
        data = merge_overrides(data, set_)

    settings = Settings.model_validate(data)
    _validate_declared_overrides(settings)

    base_dir = config_path.parent if config_path is not None else Path.cwd()
    settings.paths.rsk_directory = base_dir / settings.paths.rsk_directory
    settings.paths.profiles_directory = (
        base_dir / settings.paths.profiles_directory
    )
    settings.paths.binned_directory = base_dir / settings.paths.binned_directory
    if settings.paths.concatenated_file is not None:
        settings.paths.concatenated_file = (
            base_dir / settings.paths.concatenated_file
        )
    if settings.paths.log_file is not None:
        settings.paths.log_file = base_dir / settings.paths.log_file
    if settings.paths.error_log_file is not None:
        settings.paths.error_log_file = base_dir / settings.paths.error_log_file
    if settings.process.geolocation.external_dataset_path is not None:
        settings.process.geolocation.external_dataset_path = (
            base_dir / settings.process.geolocation.external_dataset_path
        )

    return settings
