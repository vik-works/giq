# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Top-level child package for giq-imajev's decide child.

A top-level package (not ``giq_imajev``) so importing it imports nothing of
giq: the imajev interpreter carries torch/PEFT, never giq's venv. The child
speaks giq's protocol through ``giq_child`` (stdlib only). Both travel on
PYTHONPATH (see ``giq_imajev.adapter``), alongside the imajev checkout.
"""
