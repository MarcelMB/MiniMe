import "./e2e";
import { browserFetch } from "./fetch";
import { support } from "./support";
import { showUnsupported } from "./Unsupported";

export type { EventHandlers } from "../transport";
export { openEvents } from "./events";
export const transportFetch = browserFetch;

// no OPFS: the back end has nowhere to keep runs or its cache (`main.tsx` stays main's)
if (!support().opfs) showUnsupported();
