# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Top-level child package for giq-music's text2music child.

A top-level package (not ``giq_music``) so importing it imports nothing of
giq: the acestep interpreter carries torch and ACE-Step, never giq's venv.
The child speaks giq's protocol through ``giq_child`` (stdlib only). Both
travel on PYTHONPATH (see ``giq_music.adapter``).
"""
