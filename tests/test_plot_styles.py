import unittest

from satmulator.plot_styles import (
    METHOD_ALPHA,
    METHOD_ORDER,
    bar_kwargs,
    canonical_method,
    css_rgba,
    line_kwargs,
    method_colors,
    method_hatches,
    method_markers,
    method_style,
    ordered_methods,
    run_display_label,
    violin_body_kwargs,
)


class PlotStylesTests(unittest.TestCase):
    def test_defines_fixed_method_identity(self) -> None:
        self.assertEqual(
            METHOD_ORDER,
            (
                "local-only",
                "nearest-sunlit",
                "greedy-energy",
                "phoenix2",
                "Method3",
            ),
        )
        self.assertEqual(
            method_colors(),
            ["#7F7F7F", "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728"],
        )
        self.assertEqual(method_hatches(), ["--", r"\\", "xx", "//", "**"])
        self.assertEqual(method_markers(), ["o", "s", "^", "D", "*"])
        self.assertEqual(METHOD_ALPHA, 0.75)

    def test_accepts_existing_lowercase_directory_names(self) -> None:
        self.assertEqual(canonical_method("phoenix2"), "phoenix2")
        self.assertEqual(canonical_method("method3"), "Method3")
        self.assertEqual(method_style("phoenix2").label, "phoenix")
        self.assertEqual(method_style("phoenix2").color, "#2CA02C")
        self.assertEqual(method_style("method3").color, "#D62728")

    def test_normalizes_run_labels(self) -> None:
        self.assertEqual(run_display_label("phoenix2"), "phoenix")
        self.assertEqual(run_display_label("method3"), "method3")

    def test_orders_input_by_global_order(self) -> None:
        self.assertEqual(
            ordered_methods(["method3", "local-only", "phoenix2"]),
            ["local-only", "phoenix2", "Method3"],
        )

    def test_raises_on_unknown_method(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown method"):
            method_style("random")

    def test_exposes_plot_type_kwargs(self) -> None:
        self.assertEqual(
            bar_kwargs("nearest-sunlit"),
            {
                "facecolor": "#1F77B4",
                "edgecolor": "#222222",
                "alpha": 0.75,
                "hatch": r"\\",
            },
        )
        self.assertEqual(
            violin_body_kwargs("phoenix2"),
            {"facecolor": "#2CA02C", "edgecolor": "#222222", "alpha": 0.75},
        )
        self.assertEqual(
            line_kwargs("Method3"),
            {"color": "#D62728", "alpha": 0.75, "marker": "*"},
        )
        self.assertEqual(css_rgba("local-only"), "rgba(127, 127, 127, 0.75)")


if __name__ == "__main__":
    unittest.main()
