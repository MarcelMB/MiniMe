import { useEffect, useState } from "react";

/** Call ``fn`` ``ms`` after the last call; ``cancel`` drops a pending call. */
export function debounce<A extends unknown[]>(fn: (...a: A) => void, ms: number) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const run = (...a: A) => {
    if (timer !== undefined) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = undefined;
      fn(...a);
    }, ms);
  };
  run.cancel = () => {
    if (timer !== undefined) clearTimeout(timer);
    timer = undefined;
  };
  return run;
}

/** ``value`` as it was ``ms`` ago: follows changes only once they stop. */
export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const d = debounce(setV, ms);
    d(value);
    return d.cancel;
  }, [value, ms]);
  return v;
}
