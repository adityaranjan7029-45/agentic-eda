/**
 * Read just the header row of a CSV in the browser, so the user can pick the
 * target column from a list. Handles a UTF-8 BOM, quoted names ("a,b"), escaped
 * quotes ("say ""hi""") and Windows line endings. Only the first 64 KB are read.
 */

export const HEADER_BYTES = 64 * 1024;

export function parseHeader(text: string): string[] {
  const input = text.replace(/^﻿/, "");
  const columns: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < input.length; i++) {
    const ch = input[i];
    if (quoted) {
      if (ch === '"' && input[i + 1] === '"') {
        field += '"';
        i++;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"' && field === "") {
      quoted = true;
    } else if (ch === ",") {
      columns.push(field.trim());
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      break;
    } else {
      field += ch;
    }
  }
  columns.push(field.trim());
  return columns.length === 1 && columns[0] === "" ? [] : columns;
}

const TARGET_HINTS = [/default/i, /^target$/i, /^label$/i, /churn/i, /fraud/i, /^y$/i, /_flag$/i, /outcome/i];

/** Best guess for the outcome column, or "" if nothing looks like one. */
export function guessTarget(columns: string[]): string {
  for (const hint of TARGET_HINTS) {
    const hit = columns.find((c) => hint.test(c));
    if (hit) return hit;
  }
  return "";
}

export async function readHeader(file: Blob): Promise<string[]> {
  return parseHeader(await file.slice(0, HEADER_BYTES).text());
}

/** The upload limit to show, from the server's MAX_UPLOAD_MB (same variable the API enforces). */
export function uploadLimitMb(raw: string | undefined, fallback = 50): number {
  const value = Number(raw);
  return Number.isInteger(value) && value >= 1 && value <= 500 ? value : fallback;
}
