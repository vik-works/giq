<!--
SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)

SPDX-License-Identifier: Apache-2.0
-->

# UI glossary (English → German)

One term per concept, used the same way in every view. Add a term here before
using it in a second place. German UI text is neutral and imperative, with no
"Sie" ("Modell entladen", not "Entladen Sie das Modell"). Technical terms that
German developers use as-is stay English: Token, Prompt, Sandbox, Thinking,
Engine, Job, Seed, Scheduler, LLM, VRAM, GPU.

## Hardware

| English | German | Notes |
| --- | --- | --- |
| GPU | GPU | "GPU 1", plural "GPUs" |
| card | Karte | the physical GPU; "default card" → "Standardkarte" |
| VRAM | VRAM | |
| load (utilisation) | Auslastung | the GPU's busy percentage |
| power / power draw | Leistung / Leistungsaufnahme | "Leistung 305 W / 450 W" |
| power limit | Leistungslimit | |
| temperature | Temperatur (short: Temp.) | |
| fan | Lüfter | |
| throttle / throttling | Drosselung | "thermal throttle" → "thermische Drosselung" |
| disk | Datenträger | |
| storage / disk space | Speicherplatz | |
| free | frei | "12,3 GB frei" |
| in use | belegt | "VRAM belegt" |

## Domain and residency

The six domain terms are ADR-003's; use them in no other sense.

| English | German | Notes |
| --- | --- | --- |
| engine | Engine | plural "Engines"; a runtime build: llama.cpp, vllm, sd.cpp |
| weights | Gewichte | one checkpoint on disk or in the HF cache; "Gewichte löschen" |
| recipe | Rezept | weights + engine + params; plural "Rezepte"; its name is what a client sends as `model` |
| recipe file | Rezeptdatei | plural "Rezeptdateien" |
| instance | Instanz | a recipe running on a card; plural "Instanzen" — never the file |
| inventory | Bestand | the weights and engines on this machine |
| model | Modell | only the name a client asks for (`model`, `/v1/models`) — a recipe, seen from outside |
| resident (adj.) | vorgehalten | an instance kept loaded by policy; "3 vorgehaltene Instanzen" |
| residents (n.) | vorgehaltene Instanzen | never "Residenten" |
| residency | Vorhaltung | the policy row/column heading |
| keep warm (policy `pinned`) | warm halten | the label shown for pinned |
| on demand (policy `auto`) | bei Bedarf | |
| off (policy `off`) | aus | |
| pin / pinned | warm halten / warm gehalten | the verb behind "keep warm": "Trotzdem warm halten?" |
| pinned set | warm gehaltene Rezepte | |
| revert (to default) | zurücksetzen | "Auf Standard zurücksetzen" |
| load | laden | |
| loaded | geladen | |
| unload | entladen | |
| evict / eviction | verdrängen / Verdrängung | "evicted for …" → "verdrängt für …" |
| bind (to a card) / binding | zuweisen / Zuweisung | "Bind anyway?" → "Trotzdem zuweisen?" |
| fits (badge) | passt | |
| fits after eviction | passt nach Verdrängung | |
| won't fit now | passt gerade nicht | |
| never fits | passt nie | |
| measured / estimate | gemessen / geschätzt | VRAM figures |
| modality | Modalität | the kind of job: chat, image, speech… |
| params | Parameter | an engine's settings in a recipe |
| built-in (recipe) | mitgeliefert | "mitgeliefertes Rezept" |
| left out (a file giq could not use) | ausgelassen | |

## Modalities

| English | German |
| --- | --- |
| LLM | LLM |
| Text to image | Text zu Bild |
| Image edit | Bildbearbeitung |
| Speech recognition (`audio`) | Spracherkennung |
| Speech to text (`stt`) | Sprache zu Text |
| Text to speech | Text zu Sprache |
| Voiceprint (`embed`) | Stimmabdruck |
| Document OCR | Dokument-OCR |
| Depth | Tiefe |
| vision (capability) | Bildverständnis |

## Serving and queue

| English | German | Notes |
| --- | --- | --- |
| control surface (overview) | Leitstand | the nav entry and page title |
| serving | Betrieb | "Pause serving" → "Betrieb pausieren" |
| pause / resume | pausieren / fortsetzen | |
| paused | pausiert | |
| pause when idle (graceful) | pausieren, sobald frei | |
| force pause | sofort pausieren | "force" as an override elsewhere → "erzwingen" |
| drain / draining | abarbeiten / arbeite ab… | |
| queue | Warteschlange | |
| queued / waiting | wartend | "3 wartend" |
| job | Job | plural "Jobs" |
| running | läuft / laufend | "2 laufende Jobs" |
| cancel (a job) | abbrechen | |
| in-flight work | laufende Arbeit | |
| idle | Leerlauf | state label; "im Leerlauf" in sentences |
| ready | bereit | |
| blocked | blockiert | |
| unreachable | nicht erreichbar | |
| live | live | the status dot label |
| poll | Abfrage | "Abfrage alle 3 s" |
| request | Anfrage | |
| access / exposed | Zugriff / offen | "open to your network" → "offen im Netzwerk" |

## Usage and metrics

| English | German | Notes |
| --- | --- | --- |
| usage | Nutzung | nav entry |
| calls | Aufrufe | |
| failed | fehlgeschlagen | |
| token(s) | Token | plural also "Token" (not "Tokens") |
| tokens in / out | Eingabe-Token / Ausgabe-Token | short: "Token ein / aus" |
| tok/s | Token/s | |
| throughput | Durchsatz | |
| latency | Latenz | |
| queue wait | Wartezeit | |
| run time | Laufzeit | |
| era (GPU era) | Ära | plural "Ären"; "the card era" → "Kartenära" |
| period / range | Zeitraum | "24 h", "7 T", "30 T", "90 T" |
| day / week / month / all | Tag / Woche / Monat / Gesamt | |
| recent jobs | letzte Jobs | |
| evictions | Verdrängungen | |

## Sandbox

| English | German | Notes |
| --- | --- | --- |
| sandbox | Sandbox | |
| chat | Chat | |
| tools / tool calling | Tools / Tool-Aufrufe | |
| prompt | Prompt | |
| negative prompt | Negativ-Prompt | |
| thinking | Thinking | "thinking tokens" → "Thinking-Token" |
| answer | Antwort | |
| stop | stoppen | |
| jump to latest | zum Neuesten springen | |
| transcript | Transkript | |
| speaker | Sprecher | |
| voice | Stimme | |
| record / recording | aufnehmen / Aufnahme | |
| upload | hochladen | |
| run | ausführen | |
| temperature (sampling) | Temperatur | |
| max tokens | max. Token | |

## Actions and chrome

| English | German |
| --- | --- |
| OK / Cancel / Confirm / Close | OK / Abbrechen / Bestätigen / Schließen |
| Delete | Löschen |
| Refresh | Aktualisieren |
| Retry | Erneut versuchen |
| More actions | Weitere Aktionen |
| Theme: System / Light / Dark | Design: System / Hell / Dunkel |
| Language | Sprache |
| No data yet | Noch keine Daten |
| Loading… | Lädt… |
