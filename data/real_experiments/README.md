# Physical Robot Experiment Data

This directory contains the physical robot experiment data associated with the
paper.

## Data File

- `raw_exp_data.mat`: raw synchronized robot experiment data, saved as a
  MATLAB v5 `.mat` file.

The file contains the following arrays:

| Variable | Shape | Description |
| --- | ---: | --- |
| `all_send_u` | `(12, 118150)` | Commanded input/probing signal, denoted `X_U`. |
| `all_wrench_ATI` | `(6, 118150)` | Force and torque measurements from the ATI sensor. |
| `time_step` | `(1, 118150)` | Timestamp sequence. |
| `robot_pose` | `(6, 118150)` | Robot end-effector position/pose variable, denoted `X_Z`. |
| `robot_joint` | `(7, 118150)` | Robot joint positions. |

The second dimension indexes synchronized time samples.

External videos:

- Probe experiment: <https://www.dropbox.com/scl/fi/gtrxmd8juf8imjyvv3u8k/probe.mp4?rlkey=90z5i6tph3a2qb54fm9tn9u8a&dl=0>
- Observation experiment: <https://www.dropbox.com/scl/fi/xom30hqhicx03cz5dz5yv/22obs.mp4?rlkey=cgk04zerw2qg3xo8e0ptwkcys&dl=0>
