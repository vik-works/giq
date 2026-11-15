# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Top-level child package for giq-3d's image2mesh child.

A top-level package (not ``giq_3d``) so importing it imports nothing of
giq: the pixal3d interpreter carries torch and Pixal3D's extensions, never
giq's venv. The child speaks giq's protocol through ``giq_child`` (stdlib
only). Both travel on PYTHONPATH (see ``giq_3d.adapter``).
"""
