#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import hdf5plugin
os.environ.setdefault("HDF5_PLUGIN_PATH", hdf5plugin.PLUGIN_PATH)

import h5py
import numpy as np


def copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def create_row_filtered_dataset(
    src_ds: h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    row_indices: np.ndarray,
    total_rows: int,
    chunk_rows: int,
) -> None:
    if src_ds.ndim == 0 or src_ds.shape[0] != total_rows:
        dst_ds = dst_group.create_dataset(name, data=src_ds[()])
        copy_attrs(src_ds, dst_ds)
        return

    out_shape = (len(row_indices),) + src_ds.shape[1:]
    create_kwargs = {"shape": out_shape, "dtype": src_ds.dtype}
    compression = src_ds.compression
    if compression == "unknown":
        compression = None

    if compression is not None:
        create_kwargs["compression"] = compression
    if compression is not None and src_ds.compression_opts is not None:
        create_kwargs["compression_opts"] = src_ds.compression_opts
    if compression is not None and src_ds.shuffle:
        create_kwargs["shuffle"] = True
    if compression is not None and src_ds.fletcher32:
        create_kwargs["fletcher32"] = True

    if src_ds.chunks is not None:
        chunk0 = min(chunk_rows, out_shape[0]) if out_shape[0] > 0 else 1
        create_kwargs["chunks"] = (chunk0,) + src_ds.shape[1:]

    if compression == "gzip":
        create_kwargs["compression"] = src_ds.compression
        create_kwargs["compression_opts"] = src_ds.compression_opts or 4

    dst_ds = dst_group.create_dataset(name, **create_kwargs)
    copy_attrs(src_ds, dst_ds)

    for out_start in range(0, len(row_indices), chunk_rows):
        out_end = min(out_start + chunk_rows, len(row_indices))
        src_idx = row_indices[out_start:out_end]
        dst_ds[out_start:out_end] = src_ds[src_idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an HDF5 subset by sampling complete episodes."
    )
    parser.add_argument("--input", type=Path, required=True, help="Source .h5 file.")
    parser.add_argument("--output", type=Path, required=True, help="Output .h5 file.")
    parser.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="Fraction of episodes to keep, e.g. 0.05.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=1024,
        help="Row copy chunk size to limit peak host memory.",
    )
    args = parser.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        raise ValueError("--fraction must be in (0, 1].")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as src:
        ep_col_name = "episode_idx" if "episode_idx" in src else "ep_idx"
        episodes = src[ep_col_name][:]
        unique_episodes = np.unique(episodes)
        keep_count = max(1, int(round(len(unique_episodes) * args.fraction)))

        rng = np.random.default_rng(args.seed)
        kept_episodes = np.sort(
            rng.choice(unique_episodes, size=keep_count, replace=False)
        )
        row_indices = np.nonzero(np.isin(episodes, kept_episodes))[0]
        total_rows = len(episodes)

        with h5py.File(args.output, "w") as dst:
            copy_attrs(src, dst)
            for name, obj in src.items():
                if isinstance(obj, h5py.Group):
                    dst_group = dst.create_group(name)
                    copy_attrs(obj, dst_group)
                    for ds_name, ds_obj in obj.items():
                        create_row_filtered_dataset(
                            ds_obj,
                            dst_group,
                            ds_name,
                            row_indices,
                            total_rows,
                            args.chunk_rows,
                        )
                else:
                    create_row_filtered_dataset(
                        obj, dst, name, row_indices, total_rows, args.chunk_rows
                    )

    print(f"kept_episodes={len(kept_episodes)}")
    print(f"kept_rows={len(row_indices)}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
