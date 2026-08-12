# Data Prep

Scripts that create local caches or convert raw datasets into the array formats used by training.

- `cache_data.py`: builds the MC Maze cache used by `data.py`.
- `prepare_physionetmi_context.py`: exports padded PhysioNetMI `.npy` arrays with labels, subjects, and keys.
- `convert_physionetmi_lmdb.py`: older LMDB-to-`.npy` PhysioNetMI converter.
