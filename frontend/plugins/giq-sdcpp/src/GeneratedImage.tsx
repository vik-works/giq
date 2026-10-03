// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import "./GeneratedImage.css";

export interface GeneratedImageProps {
  b64: string;
  alt: string;
}

/** A PNG that came back from an image job. */
export function GeneratedImage({ b64, alt }: GeneratedImageProps) {
  return <img className="pl-giq-sdcpp-gen" src={`data:image/png;base64,${b64}`} alt={alt} />;
}
