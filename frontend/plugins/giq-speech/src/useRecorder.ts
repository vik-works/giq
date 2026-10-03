// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@giq/plugin-ui";

/* Microphone capture needs a secure context (localhost or https): browsers
   hide getUserMedia elsewhere, so on a LAN address over plain http the
   button is not offered at all and file upload is the way in. */
export const canRecord = (): boolean =>
  typeof window !== "undefined" &&
  window.isSecureContext &&
  !!navigator.mediaDevices?.getUserMedia &&
  typeof MediaRecorder !== "undefined";

export interface Recorder {
  available: boolean;
  recording: boolean;
  /** The last finished recording (audio/webm). */
  blob: Blob | null;
  error: string | null;
  toggle: () => void;
  clear: () => void;
}

export function useRecorder(): Recorder {
  const [available] = useState(canRecord);
  const [recording, setRecording] = useState(false);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const rec = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);

  const release = () => {
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
  };
  // The microphone light goes off when the sandbox does.
  useEffect(() => () => {
    if (rec.current?.state === "recording") rec.current.stop();
    release();
  }, []);

  const toggle = useCallback(async () => {
    if (rec.current?.state === "recording") {
      rec.current.stop();
      return;
    }
    setError(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.current = s;
      const chunks: Blob[] = [];
      const r = new MediaRecorder(s);
      r.ondataavailable = (e) => chunks.push(e.data);
      r.onstop = () => {
        setBlob(new Blob(chunks, { type: "audio/webm" }));
        setRecording(false);
        release();
      };
      rec.current = r;
      r.start();
      setRecording(true);
    } catch (e) {
      // Permission denied or no input device: say so instead of failing silently.
      setError(api.errorText(e));
      release();
    }
  }, []);

  return { available, recording, blob, error, toggle: () => void toggle(), clear: () => setBlob(null) };
}
