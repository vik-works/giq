// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

/* Pure: a track from the job API as something the browser can play. */

const MIME: Record<string, string> = {
  mp3: "audio/mpeg",
  flac: "audio/flac",
  wav: "audio/wav",
  opus: "audio/ogg",
};

/** A base64 track as a Blob the <audio> element can play. */
export function audioBlob(b64: string, format: string): Blob {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: MIME[format] ?? "application/octet-stream" });
}
