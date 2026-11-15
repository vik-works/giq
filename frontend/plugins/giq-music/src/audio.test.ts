// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { audioBlob } from "./audio";

describe("audioBlob", () => {
  it("decodes the bytes and types them by format", async () => {
    const blob = audioBlob(btoa("ID3\u0004"), "mp3");
    expect(blob.type).toBe("audio/mpeg");
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(new Uint8Array([73, 68, 51, 4]));
  });

  it("falls back to octet-stream for an unknown format", () => {
    expect(audioBlob(btoa("x"), "xyz").type).toBe("application/octet-stream");
  });
});
