# `init`

```
ctd-processing init [OPTIONS]
```

Writes a starter `config.toml` for a new project, from the package's
bundled default template (or a caller-supplied `--template`), and creates
the directories it references.

`[process.geolocation]` has no default and `init` has no way to know a
project's real position, so it is left unset (commented out) in the
written file — the config will not validate for `process`/`bin`/
`concatenate` until it's filled in by hand. See
[Configuration → Geolocation](../configuration.md#geolocation).

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--name` | `my_ctd_processing_project` | Human-readable project name, written as `project.name`. |
| `--rsk-directory` | `rsk_files` | Directory for raw `.rsk` deployment files, written as `paths.rsk_directory`. Resolved and created relative to `--working-dir` if relative. |
| `--profiles-directory` | `profiles` | Directory for extracted profile files, written as `paths.profiles_directory`. Resolved/created the same way. |
| `--binned-directory` | `binned` | Directory for binned profile files, written as `paths.binned_directory`. Resolved/created the same way. |
| `--log-file` | unset | File to write log records below `ERROR` level to, written as `paths.log_file`. |
| `--error-log-file` | unset | File to write log records at `ERROR` level and above to, written as `paths.error_log_file`. |
| `--working-dir` | current directory | Directory to write `config.toml` into, and to resolve relative directory options against. Created if it doesn't exist. |
| `--template` | bundled default | Use this TOML file as the starting point instead of the bundled default template. |
| `--set` | — | Override a config option on top of the above, e.g. `--set project.name=...`. Repeatable. |
| `--force`, `-f` | off | Overwrite an existing `config.toml` in the working directory. |

Plus the [shared logging options](index.md#options-shared-by-every-command).

## Example

```bash
ctd-processing init \
  --name "my_survey" \
  --rsk-directory rsk_files \
  --profiles-directory profiles \
  --binned-directory binned \
  --working-dir my_survey_project
```

Since these directory options are always applied on top of the template,
the written `config.toml` is always re-serialized and does not preserve
the bundled template's comments.
