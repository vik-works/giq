# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Top-level child package for giq-sfx's text2sfx child.

A top-level package (not ``giq_sfx``) so importing it imports nothing of
giq: the mosssfx interpreter carries torch and MOSS-SoundEffect, never giq's
venv. The child speaks giq's protocol through ``giq_child`` (stdlib only).
Both travel on PYTHONPATH (see ``giq_sfx.adapter``).
"""
