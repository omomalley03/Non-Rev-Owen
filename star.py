"""Mark a run's about.txt as "starred".

Usage
-----
    python star.py                 # star the most recent run
    python star.py --run 2         # star the 2nd most recent run
    python star.py --run runs/foo  # star a specific run dir (or about.txt path)
    python star.py --synth         # star the most recent synthetic run
    python star.py --synth --run 2 # star the 2nd most recent synthetic run
    python star.py --comment "good regularization"   # star + attach a comment
"""

import argparse
import os

from paths import RUNS_DIR, SYNTH_RUNS_DIR

STAR_HEADER = "★ STARRED ★"
COMMENT_PREFIX = "# comment: "


def resolve_run_dir(arg_run, root):
    """Resolve --run (None | index | path) to a run directory under `root`."""
    # explicit path: accept a run dir or an about.txt path directly
    if arg_run is not None and not str(arg_run).isdigit():
        path = arg_run
        if os.path.isfile(path) and os.path.basename(path) == "about.txt":
            return os.path.dirname(path)
        if os.path.isdir(path):
            return path
        raise FileNotFoundError(f"No such run dir or about.txt: {arg_run!r}")

    if not os.path.isdir(root):
        raise FileNotFoundError(f"No {root!r} directory found.")
    runs = sorted(
        [os.path.join(root, d) for d in os.listdir(root)
         if os.path.isfile(os.path.join(root, d, "about.txt"))],
        key=os.path.getmtime, reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No runs with about.txt found in {root!r}.")

    if arg_run is None:
        return runs[0]
    idx = int(arg_run) - 1
    if idx < 0 or idx >= len(runs):
        raise ValueError(f"--run {arg_run} out of range (1–{len(runs)})")
    return runs[idx]


def star_run(run_dir, comment=None):
    about = os.path.join(run_dir, "about.txt")
    if not os.path.isfile(about):
        raise FileNotFoundError(f"No about.txt in {run_dir!r}")
    with open(about) as f:
        lines = f.read().split("\n")

    # strip any existing star header + comment line to get the original body
    already = lines and lines[0] == STAR_HEADER
    if already:
        lines = lines[1:]
    existing_comment = None
    if lines and lines[0].startswith(COMMENT_PREFIX):
        existing_comment = lines[0][len(COMMENT_PREFIX):]
        lines = lines[1:]

    # a new --comment overrides; otherwise keep an existing one
    final_comment = comment if comment is not None else existing_comment

    header = [STAR_HEADER]
    if final_comment:
        header.append(COMMENT_PREFIX + final_comment)
    with open(about, "w") as f:
        f.write("\n".join(header + lines))

    if already and comment is None:
        print(f"Already starred → {about}")
    elif comment is not None:
        print(f"Starred with comment → {about}")
    else:
        print(f"Starred → {about}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None,
                        help="Integer (1=most recent) or explicit run dir / about.txt path.")
    parser.add_argument("--synth", action="store_true",
                        help="Operate on synth_runs/ instead of runs/.")
    parser.add_argument("--comment", default=None,
                        help="Attach a comment to the starred about.txt.")
    args = parser.parse_args()

    root = SYNTH_RUNS_DIR if args.synth else RUNS_DIR
    run_dir = resolve_run_dir(args.run, root)
    star_run(run_dir, comment=args.comment)


if __name__ == "__main__":
    main()
