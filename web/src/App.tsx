import { Home } from "./home/Home";
import { JobPage } from "./launch/JobPage";
import { NewRun } from "./launch/NewRun";
import { href, useRoute } from "./router";
import { RunPage } from "./run/RunPage";

const REPO = "https://github.com/MarcelMB/MiniMe";

export function App() {
  const route = useRoute();
  return (
    <>
      <header className="topbar">
        <a className="brand" href={href({ name: "home" })}>
          <picture>
            <source media="(prefers-color-scheme: dark)" srcSet="./minime-logo-dark.svg" />
            <img src="./minime-logo.svg" alt="MiniMe" />
          </picture>
        </a>
        <nav aria-label="Main">
          <a className="btn" href={href({ name: "home" })}>
            Runs
          </a>
          <a className="btn" href={href({ name: "new" })}>
            New run
          </a>
        </nav>
        <nav className="project-links" aria-label="Project">
          <a className="btn" href={REPO} target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
          <a className="btn" href={`${REPO}/issues/new`} target="_blank" rel="noopener noreferrer">
            Found an issue?
          </a>
        </nav>
      </header>
      {route.name === "home" && <Home />}
      {route.name === "new" && <NewRun />}
      {route.name === "job" && <JobPage key={route.id} id={route.id} />}
      {route.name === "run" && <RunPage key={route.id} id={route.id} page={route.page} />}
    </>
  );
}
