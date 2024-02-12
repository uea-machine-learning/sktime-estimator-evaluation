"""Labels PRs based on title and change list.

Must be run in a github action with the pull_request_target event.

Based on the scikit-learn v1.3.1 label_title_regex.py script.
"""

import json
import os
import re
import sys

from github import Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo = context_dict["repository"]
g = Github(sys.argv[1])
repo = g.get_repo(repo)
pr_number = context_dict["event"]["number"]
pr = repo.get_pull(number=pr_number)
labels = [label.name for label in pr.get_labels()]

if "[bot]" in pr.user.login:
    sys.exit(0)

# title labels
title = pr.title

title_regex_to_labels = [
    (r"\bENH\b", "enhancement"),
    (r"\bMNT\b", "maintenance"),
    (r"\bBUG\b", "bug"),
    (r"\bDOC\b", "documentation"),
    (r"\bREF\b", "refactor"),
    (r"\bDEP\b", "dependencies"),
]

title_labels = [
    label for regex, label in title_regex_to_labels if re.search(regex, title)
]
title_labels_to_add = list(set(title_labels) - set(labels))

# content labels
paths = [file.filename for file in pr.get_files()]

content_paths_to_labels = [
    ("_tsml_research_resources/", "tsml research resources"),
    ("tsml_eval/datasets/", "datasets"),
    ("tsml_eval/estimators/", "estimators"),
    ("tsml_eval/evaluation/", "evaluation"),
    ("examples/", "examples"),
    ("tsml_eval/experiments/", "experiments"),
    ("tsml_eval/publications/", "publications"),
    ("results/", "results"),
    ("tsml_eval/testing/", "testing"),
]

present_content_labels = [
    label for _, label in content_paths_to_labels if label in labels
]

content_labels = [
    label
    for package, label in content_paths_to_labels
    if any([package in path for path in paths])
]
content_labels = list(set(content_labels))

content_labels_to_add = content_labels
content_labels_status = "used"
if len(present_content_labels) > 0:
    content_labels_to_add = []
    content_labels_status = "ignored"
if len(content_labels) > 3:
    content_labels_to_add = []
    content_labels_status = (
        "large" if content_labels_status != "ignored" else "ignored+large"
    )

# add to PR
if title_labels_to_add or content_labels_to_add:
    pr.add_to_labels(*title_labels_to_add + content_labels_to_add)

with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
    print(f"title-labels={title_labels}".replace(" ", ""), file=fh)  # noqa: T201
    print(  # noqa: T201
        f"title-labels-new={title_labels_to_add}".replace(" ", ""), file=fh
    )
    print(f"content-labels={content_labels}".replace(" ", ""), file=fh)  # noqa: T201
    print(f"content-labels-status={content_labels_status}", file=fh)  # noqa: T201
