// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { CopyButton } from "../../components/CopyButton";
import "./CopyCommand.css";

/* A command for the operator to run on the server, with a copy button. The
   dashboard does not run it: installing a plugin changes giq's own
   environment, and a hardened service may not write its model store. */
export function CopyCommand({ command }: { command: string }) {
  return (
    <div className="rc-add-command">
      <code className="mono">{command}</code>
      <CopyButton text={command} />
    </div>
  );
}
