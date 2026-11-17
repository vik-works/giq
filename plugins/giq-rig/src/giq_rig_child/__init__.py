# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Top-level child package for giq-rig's mesh2rig child.

A top-level package (not ``giq_rig``) so importing it imports nothing of
giq: the skintokens interpreter carries torch, flash-attn and bpy, never
giq's venv. The child speaks giq's protocol through ``giq_child`` (stdlib
only). Both travel on PYTHONPATH (see ``giq_rig.adapter``).
"""
