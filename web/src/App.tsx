import { Home } from "./home/Home";
import { JobPage } from "./launch/JobPage";
import { NewRun } from "./launch/NewRun";
import { href, useRoute } from "./router";
import { RunPage } from "./run/RunPage";

export function App() {
  const route = useRoute();
  return (
    <>
      <header className="topbar">
        <a className="brand" href={href({ name: "home" })}>
          MiniMe
        </a>
        <nav aria-label="Main">
          <a className="btn" href={href({ name: "home" })}>
            Runs
          </a>
          <a className="btn" href={href({ name: "new" })}>
            New run
          </a>
        </nav>
      </header>
      {route.name === "home" && <Home />}
      {route.name === "new" && <NewRun />}
      {route.name === "job" && <JobPage key={route.id} id={route.id} />}
      {route.name === "run" && (
        <RunPage key={route.id} id={route.id} page={route.page} />
      )}
    </>
  );
}
