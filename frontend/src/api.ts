// Allows React App to communicate with the FastAPI backend


const API = (import.meta.env.VITE_API_URL as string) ?? "";


// Function to send POST requests

async function postJSON<T>(path: string, body: unknown): Promise<T> {
    const result = await fetch(`${API}${path}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
    });
    // Error Handling
    if (!result.ok) {
        const text = await result.text();
        throw new Error(`${result.status} ${result.statusText}: ${text}`);
    }
    return result.json() as Promise<T>
}

export type AnalyzeResponse = {
  sentiment_overall: Record<string, number>;
  bias_overall: Record<string, number>;
  sentiment: string;
  bias: string;
  charged_total: number;
  charged_unique: string[];
  sentences: any[];
  top_positive?: { text: string; score: number }[];
  top_negative?: { text: string; score: number }[];
  top_biased?:   { text: string; score: number; side?: 'liberal'|'conservative' }[];
};

// Wrapper object for calling your backend
export const api = {
    analyzeUrl: (url:string) => postJSON<AnalyzeResponse>("/analyze-url", {url}),
    analyzeText: (text: string) => postJSON<AnalyzeResponse>("/analyze-text", {text}),
};