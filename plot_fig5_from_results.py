"""Generate left- and right-ear fig. 5 PDFs from the latest four saved optimisation runs."""

import matplotlib

matplotlib.use("Agg")

from DAFx2026_fig_rendering import regenerate_fig5_from_latest_runs


if __name__ == "__main__":
    left_pdf = regenerate_fig5_from_latest_runs(ear_idx=0, output_pdf="DAFx2026_fig5_left_ear.pdf")
    right_pdf = regenerate_fig5_from_latest_runs(ear_idx=1, output_pdf="DAFx2026_fig5_right_ear.pdf")
    print(f"Saved {left_pdf}")
    print(f"Saved {right_pdf}")
