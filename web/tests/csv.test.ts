import { describe, expect, it, vi } from "vitest";

import { HEADER_BYTES, guessTarget, parseHeader, readHeader, uploadLimitMb } from "../lib/csv";

describe("parseHeader", () => {
  it("reads a plain header row and stops at the line break", () => {
    expect(parseHeader("id,age,default_flag\n1,30,0\n")).toEqual(["id", "age", "default_flag"]);
  });
  it("handles a BOM, Windows line endings and spaces", () => {
    expect(parseHeader("﻿ id , income \r\n1,2")).toEqual(["id", "income"]);
  });
  it("handles quoted names with commas and escaped quotes", () => {
    expect(parseHeader('"a,b","say ""hi""",c\n')).toEqual(["a,b", 'say "hi"', "c"]);
  });
  it("returns nothing for an empty file", () => {
    expect(parseHeader("")).toEqual([]);
  });
  it("handles old Mac line endings (\\r only)", () => {
    expect(parseHeader("a,b\r1,2\r")).toEqual(["a", "b"]);
  });
  it("never loads the whole file: only the first 64 KB is read", async () => {
    const slice = vi.fn((start: number, end: number) => new Blob(["x,y,target\n"]).slice(start, end));
    const huge = { slice, text: () => { throw new Error("whole file read"); } } as unknown as Blob;
    expect(await readHeader(huge)).toEqual(["x", "y", "target"]);
    expect(slice).toHaveBeenCalledWith(0, HEADER_BYTES);
  });
  it("reads only the start of a large file", async () => {
    const header = "x,y,target\n";
    const blob = new Blob([header + "1,2,0\n".repeat(200_000)]);
    expect(await readHeader(blob)).toEqual(["x", "y", "target"]);
  });
});

describe("guessTarget", () => {
  it("prefers obvious outcome names", () => {
    expect(guessTarget(["application_id", "credit_score", "default_flag"])).toBe("default_flag");
    expect(guessTarget(["customer", "tenure", "Churn"])).toBe("Churn");
    expect(guessTarget(["a", "b", "target"])).toBe("target");
  });
  it("returns nothing when no column looks like an outcome", () => {
    expect(guessTarget(["a", "b", "c"])).toBe("");
  });
});

describe("uploadLimitMb", () => {
  it("uses the server's MAX_UPLOAD_MB", () => {
    expect(uploadLimitMb("10")).toBe(10);
    expect(uploadLimitMb("50")).toBe(50);
  });
  it("falls back to 50 when it's missing or nonsense", () => {
    for (const raw of [undefined, "", "abc", "0", "-5", "2.5", "100000"]) expect(uploadLimitMb(raw)).toBe(50);
  });
});
