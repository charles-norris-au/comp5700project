"""
kde_compare.py
--------------
Comparison utilities for KDE YAML files produced by kde_extractor.py.

Functions
---------
load_yaml_outputs(output_dir)
    Discover and load the two YAML files written by process_two_files().

compare_element_names(yaml_path_1, yaml_path_2, out_path)
    Report KDE names that appear in one file but not the other.

compare_elements_and_requirements(yaml_path_1, yaml_path_2, out_path)
    Report per-name and per-requirement differences as structured tuples.
"""

import os
import glob
import yaml


# ---------------------------------------------------------
# 1. AUTO-LOAD THE TWO OUTPUT YAML FILES FROM TASK 1
# ---------------------------------------------------------

def load_yaml_outputs(output_dir="kde_outputs"):
    """
    Automatically discover the two YAML files written by process_two_files().
    Files are expected to follow the naming pattern: input<N>-*_kdes.yaml

    Args:
        output_dir: Directory passed to process_two_files() as output_dir.

    Returns:
        (path_1, path_2): Absolute paths to input1-*.yaml and input2-*.yaml.

    Raises:
        FileNotFoundError: If the directory is missing or fewer than two
                           matching YAML files are found.
    """
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: '{output_dir}'")

    pattern = os.path.join(output_dir, "input*-*_kdes.yaml")
    matches = sorted(glob.glob(pattern))          # sort → input1 before input2

    if len(matches) < 2:
        raise FileNotFoundError(
            f"Expected at least 2 YAML files matching '{pattern}', "
            f"found {len(matches)}: {matches}"
        )

    if len(matches) > 2:
        print(f"[WARN] Found {len(matches)} YAML files; using the first two: "
              f"{matches[0]}, {matches[1]}")

    return matches[0], matches[1]


def _load_kde_yaml(path):
    """Load a KDE YAML file and return a dict mapping name → set(requirements)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    kde_map = {}
    for element in raw.values():
        if not isinstance(element, dict):
            continue
        name = element.get("name", "").strip()
        reqs = element.get("requirements", []) or []
        if name:
            # If the same name appears more than once (merged duplicates) union reqs
            existing = kde_map.get(name, set())
            kde_map[name] = existing | {str(r).strip() for r in reqs if r}

    return kde_map


# ---------------------------------------------------------
# 2. COMPARE ELEMENT NAMES ONLY
# ---------------------------------------------------------

def compare_element_names(yaml_path_1, yaml_path_2, out_path="name_diff.txt"):
    """
    Identify KDE names present in one file but absent in the other.

    Output format (one entry per differing name):
        PRESENT-IN-<file>  ABSENT-IN-<file>  <name>

    If there are no differences the file contains a single line:
        NO DIFFERENCES IN REGARDS TO ELEMENT NAMES

    Args:
        yaml_path_1: Path to the first KDE YAML file.
        yaml_path_2: Path to the second KDE YAML file.
        out_path:    Path for the output TEXT file.

    Returns:
        out_path
    """
    label_1 = os.path.basename(yaml_path_1)
    label_2 = os.path.basename(yaml_path_2)

    kde_1 = _load_kde_yaml(yaml_path_1)
    kde_2 = _load_kde_yaml(yaml_path_2)

    names_1 = set(kde_1.keys())
    names_2 = set(kde_2.keys())

    only_in_1 = sorted(names_1 - names_2)
    only_in_2 = sorted(names_2 - names_1)

    lines = []

    for name in only_in_1:
        lines.append(f"PRESENT-IN-{label_1}  ABSENT-IN-{label_2}  {name}")

    for name in only_in_2:
        lines.append(f"ABSENT-IN-{label_1}  PRESENT-IN-{label_2}  {name}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            f.write("NO DIFFERENCES IN REGARDS TO ELEMENT NAMES\n")

    print(f"[compare_element_names] {len(lines)} difference(s) → {out_path}")
    return out_path


# ---------------------------------------------------------
# 3. COMPARE ELEMENT NAMES AND REQUIREMENTS
# ---------------------------------------------------------

def compare_elements_and_requirements(yaml_path_1, yaml_path_2, out_path="full_diff.txt"):
    """
    Identify differences in both KDE names and their requirements.

    Tuple format, one per line:
        NAME,ABSENT-IN-<file>,PRESENT-IN-<file>,NA
            → KDE exists in one file but not the other (requirements not applicable)

        NAME,ABSENT-IN-<file>,PRESENT-IN-<file>,<REQ>
            → KDE exists in both files but <REQ> is only in one of them

    Args:
        yaml_path_1: Path to the first KDE YAML file.
        yaml_path_2: Path to the second KDE YAML file.
        out_path:    Path for the output TEXT file.

    Returns:
        out_path
    """
    label_1 = os.path.basename(yaml_path_1)
    label_2 = os.path.basename(yaml_path_2)

    kde_1 = _load_kde_yaml(yaml_path_1)
    kde_2 = _load_kde_yaml(yaml_path_2)

    names_1 = set(kde_1.keys())
    names_2 = set(kde_2.keys())

    tuples = []

    # -- KDEs present in file 1 only ----------------------------------------
    for name in sorted(names_1 - names_2):
        tuples.append((name, f"ABSENT-IN-{label_2}", f"PRESENT-IN-{label_1}", "NA"))

    # -- KDEs present in file 2 only ----------------------------------------
    for name in sorted(names_2 - names_1):
        tuples.append((name, f"ABSENT-IN-{label_1}", f"PRESENT-IN-{label_2}", "NA"))

    # -- KDEs present in both: compare requirements -------------------------
    for name in sorted(names_1 & names_2):
        reqs_1 = kde_1[name]
        reqs_2 = kde_2[name]

        # Requirements only in file 1
        for req in sorted(reqs_1 - reqs_2):
            tuples.append((name, f"ABSENT-IN-{label_2}", f"PRESENT-IN-{label_1}", req))

        # Requirements only in file 2
        for req in sorted(reqs_2 - reqs_1):
            tuples.append((name, f"ABSENT-IN-{label_1}", f"PRESENT-IN-{label_2}", req))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        if tuples:
            for t in tuples:
                f.write(",".join(t) + "\n")
        else:
            f.write("NO DIFFERENCES IN REGARDS TO ELEMENT NAMES OR REQUIREMENTS\n")

    print(f"[compare_elements_and_requirements] {len(tuples)} difference(s) → {out_path}")
    return out_path


# ---------------------------------------------------------
# 4. ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    yaml_1, yaml_2 = load_yaml_outputs(output_dir="kde_outputs")
    print(f"Loaded:\n  {yaml_1}\n  {yaml_2}\n")

    compare_element_names(
        yaml_1, yaml_2,
        out_path="kde_outputs/name_diff.txt",
    )

    compare_elements_and_requirements(
        yaml_1, yaml_2,
        out_path="kde_outputs/full_diff.txt",
    )
