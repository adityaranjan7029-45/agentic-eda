/** Plain-language messages for API errors (the API's own `detail` is used when it's specific). */

export function errorMessage(status: number, detail?: unknown): string {
  const text = typeof detail === "string" ? detail : undefined;
  switch (status) {
    case 401:
      return "Your session has expired. Please sign in again.";
    case 404:
      return "That run doesn't exist, or it isn't yours.";
    case 409:
      return text ?? "This run is still in progress.";
    case 413:
      return text ? `That file is too large: ${text}.` : "That file is too large.";
    case 415:
      return "Only .csv files are supported.";
    case 422:
      return text ? capitalise(text) + "." : "That file couldn't be read as a CSV.";
    case 429:
      return text ? capitalise(text) + "." : "You already have runs in progress. Please wait for one to finish.";
    case 502:
    case 503:
      return "The analysis service is unavailable right now. Please try again in a minute.";
    default:
      return text ? capitalise(text) : `Something went wrong (error ${status}).`;
  }
}

function capitalise(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export async function errorFromResponse(response: Response): Promise<string> {
  let detail: unknown;
  try {
    detail = (await response.json()).detail;
  } catch {
    detail = undefined;
  }
  return errorMessage(response.status, detail);
}
