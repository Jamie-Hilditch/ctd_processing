"""Configuration model and loading utilities for ctd_processing."""

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
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
        the same way as `rsk_directory`.
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
    applies these to the dataset's `sea_pressure` channel.

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
    direction : {"up", "down", "both"}
        Which cast direction(s) to identify profiles for. Defaults to
        ``"down"``.
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
    direction: Literal["up", "down", "both"] = "down"
    missing: Literal["raise", "drop"] = "drop"


class CTLagSettings(BaseModel):
    """Settings for the conductivity/temperature (CT) lag correction.

    See `ctd_processing.process.ct_lag`, which applies these to a
    dataset's `sea_water_electrical_conductivity`, `sea_water_temperature`,
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
        `sea_water_electrical_conductivity` channel directly rather than
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
    derived from the profile's `sea_water_electrical_conductivity`,
    `sea_water_temperature`, and `sea_pressure` channels, plus its
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
        ``"speed_of_sound_in_sea_water"`` (overwriting, as above).
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


class DespikeSettings(BaseModel):
    """Settings for despiking one channel with an iterative rolling median.

    See `ctd_processing.process.despike`. The channel is smoothed with a
    rolling median filter of `window_length` to get a "reference" series;
    points whose residual against that reference exceeds `threshold`
    standard deviations are replaced with NaN. This repeats, up to
    `max_iterations` times, stopping early the first pass that finds no
    new spikes -- removing large spikes can unmask smaller ones the
    previous pass's median/std missed.

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


class ProcessSettings(BaseModel):
    """Settings specific to the ``process`` command.

    Attributes
    ----------
    raw_channels : dict[str, RawChannelSettings]
        Per-raw-channel processing settings, keyed by
        `ctd_processing.process.cf_channels.channel_key_for_longname`'s
        result for that channel -- the same short identifier a channel
        ends up under in `ctd_processing.process.dataset.Dataset.channels`
        (e.g. ``"sea_water_temperature"``), not the raw pyrsktools field
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
        (the default) is written with zstd compression and byte-stream-split
        encoding for float columns -- the better fit for this fast,
        size-sensitive intermediate stage. ``"netcdf"`` writes
        CF-compliant files (``units``/``long_name``/``standard_name`` as
        variable attributes, project/deployment metadata and processing
        history as global attributes) via `xarray` and `h5netcdf` --
        better for long-term self-description and interop with CF-aware
        tools (e.g. ERDDAP, OceanSITES), at the cost of a larger,
        less-compressed file. Defaults to ``"parquet"``.
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
    despike : DespikeSettings
        Project-wide default despike settings (see `DespikeSettings`),
        used as the base that `despike_channels` entries override.
        Optional; every field has a default.
    despike_channels : dict[str, dict[str, Any]]
        Which channels to despike, and their per-channel overrides of
        `despike`. Keyed by channel name -- the same namespace for raw
        channels (e.g. ``"sea_water_temperature"``, matching
        `raw_channels`' own keys) and derived-variable channels (e.g.
        ``"practical_salinity"``, matching
        `ctd_processing.process.derived_variables`'s output keys). A
        channel is despiked if and only if it has an entry here, even an
        empty one (``{}``, meaning "use `despike`'s defaults for this
        channel"); each entry's fields override `despike` field-by-field
        (see `resolve_despike_settings`) -- the same partial-override
        pattern `InstrumentSettings.process`/`DeploymentSettings.process`
        use, merged one level deeper. Defaults to an empty dict, i.e. no
        channel is despiked.
    """

    model_config = ConfigDict(extra="forbid")

    raw_channels: dict[str, RawChannelSettings] = Field(default_factory=dict)
    atmospheric_pressure: float | None = None
    profiles: ProfileSettings = Field(default_factory=ProfileSettings)
    ct_lag: CTLagSettings = Field(default_factory=CTLagSettings)
    profile_format: Literal["netcdf", "parquet"] = "parquet"
    geolocation: GeolocationSettings
    derived_variables: DerivedVariablesSettings = Field(
        default_factory=DerivedVariablesSettings
    )
    despike: DespikeSettings = Field(default_factory=DespikeSettings)
    despike_channels: dict[str, dict[str, Any]] = Field(default_factory=dict)


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

    For each channel named in `process_settings.despike_channels`,
    deep-merges that channel's partial override onto
    `process_settings.despike` (the project-wide defaults) and validates
    the result -- the same override mechanism `resolve_process_settings`
    uses for instrument/deployment overrides, applied one level deeper.
    A channel with no entry in `despike_channels` is not despiked at all,
    and so has no entry in the returned dict.

    Parameters
    ----------
    process_settings : ProcessSettings
        Settings providing `despike`/`despike_channels`.

    Returns
    -------
    dict[str, DespikeSettings]
        The resolved, validated despike settings, keyed by channel name.

    Raises
    ------
    pydantic.ValidationError
        If a channel's merged override does not form valid
        `DespikeSettings`.
    """
    base = process_settings.despike.model_dump(mode="json")
    return {
        name: DespikeSettings.model_validate(_deep_merge(base, override))
        for name, override in process_settings.despike_channels.items()
    }


def _validate_declared_overrides(settings: Settings) -> None:
    """Eagerly validate every declared instrument/deployment override.

    Each `settings.instruments`/`settings.deployments` entry is merged
    onto `settings.process` and validated on its own -- independently of
    every other entry -- so a typo'd or invalid override fails fast at
    config-load time rather than only when that specific instrument or
    deployment is later processed. `despike_channels` overrides are
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
        (or, for `despike_channels`, `DespikeSettings`), annotated with a
        note identifying the offending
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
        ``paths.binned_directory``, (when given) ``paths.log_file`` and
        ``paths.error_log_file``, and (when given)
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
    if settings.paths.log_file is not None:
        settings.paths.log_file = base_dir / settings.paths.log_file
    if settings.paths.error_log_file is not None:
        settings.paths.error_log_file = base_dir / settings.paths.error_log_file
    if settings.process.geolocation.external_dataset_path is not None:
        settings.process.geolocation.external_dataset_path = (
            base_dir / settings.process.geolocation.external_dataset_path
        )

    return settings
