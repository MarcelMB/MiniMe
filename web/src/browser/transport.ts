import { browserFetch } from "./fetch";

export type { EventHandlers } from "../transport";
export { openEvents } from "./events";
export const transportFetch = browserFetch;
