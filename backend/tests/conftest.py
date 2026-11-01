"""Every test/data/<dataset>/annotations.json is discovered automatically and each image becomes
one test case. Tests record their metrics in a shared Report, printed as a summary at the end."""
from collections import defaultdict
from functools import reduce

import pytest

from tests.evaluation import IOU_THRESHOLD, Report, load_all_samples

REPORT_KEY = pytest.StashKey[Report]()


def pytest_configure(config):
    config.stash[REPORT_KEY] = Report()


def pytest_generate_tests(metafunc):
    if "sample" in metafunc.fixturenames:
        samples = load_all_samples()
        metafunc.parametrize("sample", samples, ids=[s.id for s in samples])


@pytest.fixture
def report(request) -> Report:
    return request.config.stash[REPORT_KEY]


def fmt(value):
    return "-" if value is None else f"{value:.2f}"


def by_dataset(results):
    groups = defaultdict(dict)
    for sample_id, result in results.items():
        groups[sample_id.split("/", 1)[0]][sample_id] = result
    return groups


def print_table(tr, title, header, results, row):
    if not results:
        return
    tr.write_sep("=", title)
    tr.write_line(header)
    for dataset, rows in sorted(by_dataset(results).items()):
        for sample_id, result in rows.items():
            tr.write_line(row(sample_id, result))
        tr.write_line(row(f"{dataset} (total)", reduce(lambda a, b: a + b, rows.values())), bold=True)
        tr.write_line("")


def pytest_terminal_summary(terminalreporter, config):
    report = config.stash[REPORT_KEY]
    for name, results in [("opencv", report.detection), ("yolo", report.yolo_detection)]:
        print_table(
            terminalreporter,
            f"{name} checkbox detection (IoU >= {IOU_THRESHOLD})",
            f"{'sample':32} {'GT':>5} {'pred':>5} {'TP':>5} {'prec':>6} {'recall':>6} {'F1':>6}",
            results,
            lambda name, r: f"{name:32} {r.n_gt:5} {r.n_pred:5} {r.tp:5} "
                            f"{fmt(r.precision):>6} {fmt(r.recall):>6} {fmt(r.f1):>6}",
        )
    for name, results in [("pretrained", report.classification), ("own", report.own_classification)]:
        print_table(
            terminalreporter,
            f"{name} checkbox classification (ground-truth boxes, ambiguous excluded, positive = checked)",
            f"{'sample':32} {'n':>5} {'acc':>6} {'prec':>6} {'recall':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}",
            results,
            lambda name, r: f"{name:32} {r.n:5} {fmt(r.accuracy):>6} {fmt(r.precision):>6} {fmt(r.recall):>6} "
                            f"{r.tp:5} {r.fp:5} {r.fn:5} {r.tn:5}",
        )
