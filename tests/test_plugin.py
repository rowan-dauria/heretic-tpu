# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

from heretic_tpu.plugin import is_builtin_plugin, load_plugin, resolve_plugin_name
from heretic_tpu.scorer import Scorer
from heretic_tpu.scorers.keyword_rate import KeywordRate


class PluginNameTests(unittest.TestCase):
    def test_upstream_names_are_redirected(self) -> None:
        self.assertEqual(
            resolve_plugin_name("heretic.scorers.keyword_rate.KeywordRate"),
            "heretic_tpu.scorers.keyword_rate.KeywordRate",
        )

    def test_port_and_external_names_are_unchanged(self) -> None:
        for name in [
            "heretic_tpu.scorers.keyword_rate.KeywordRate",
            "my_plugins.scorers.Custom",
            "plugins/custom.py:Custom",
        ]:
            with self.subTest(name=name):
                self.assertEqual(resolve_plugin_name(name), name)

    def test_builtin_detection(self) -> None:
        self.assertTrue(is_builtin_plugin("heretic.scorers.keyword_rate.KeywordRate"))
        self.assertTrue(
            is_builtin_plugin("heretic_tpu.scorers.keyword_rate.KeywordRate")
        )
        self.assertFalse(is_builtin_plugin("my_plugins.scorers.Custom"))
        self.assertFalse(is_builtin_plugin("heretic_extra.scorers.Custom"))

    def test_upstream_name_loads_port_plugin(self) -> None:
        self.assertIs(
            load_plugin("heretic.scorers.keyword_rate.KeywordRate", Scorer),
            KeywordRate,
        )


if __name__ == "__main__":
    unittest.main()
