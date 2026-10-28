import { useMemo, useState } from "react";
import { getJSON, postJSON } from "../api";
import { ErrorNote, Loading, Toggle } from "../controls";
import { fmtBytes, fmtDuration, fmtInt } from "../format";
import { FolderPicker } from "../home/FolderPicker";
import { useResource } from "../poll";
import { go } from "../router";
import type { Job, ParamMeta, RawInfo } from "../types";
import {
  changedOnly,
  defaultRunName,
  failingCount,
  offersOverwrite,
  overwriteAfterError,
} from "./params";
import { ParamsForm } from "./ParamsForm";
import { Preview } from "./Preview";

export function NewRun() {
  const defaults = useResource("params", () =>
    getJSON<{ groups: string[]; params: ParamMeta[] }>("/params/defaults"),
  );
  const [raw, setRaw] = useState("");
  const [picking, setPicking] = useState(true);
  const info = useResource(raw ? `raw:${raw}` : null, () =>
    getJSON<RawInfo>(`/raw/info?path=${encodeURIComponent(raw)}`),
  );
  const [name, setName] = useState(() => defaultRunName(new Date()));
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [keep, setKeep] = useState(false);
  const [overwrite, setOverwrite] = useState(false);
  const [offered, setOffered] = useState(false);
  const [submitErr, setSubmitErr] = useState("");
  const [busy, setBusy] = useState(false);

  const merged = useMemo(() => {
    const d: Record<string, unknown> = {};
    for (const m of defaults.data?.params ?? []) d[m.name] = m.default;
    return { ...d, ...values };
  }, [defaults.data, values]);
  const failing = failingCount(errors);
  const invalid = failing > 0;
  const resetOverwrite = () => {
    setOverwrite(false);
    setOffered(false);
    setSubmitErr("");
  };
  const outDir = raw
    ? `${raw.replace(/\/+$/, "").split("/").slice(0, -1).join("/")}/processed/${name}`
    : "";

  const start = () => {
    setBusy(true);
    setSubmitErr("");
    postJSON<Job>("/jobs", {
      raw_dir: raw,
      name,
      params: changedOnly(merged, defaults.data!.params),
      keep_store: keep,
      overwrite,
    }).then(
      (job) => go({ name: "job", id: job.id }),
      (e: Error) => {
        setSubmitErr(e.message);
        setOverwrite(overwriteAfterError(e.message, overwrite));
        setOffered(offered || offersOverwrite(e.message));
        setBusy(false);
      },
    );
  };

  return (
    <div className="page">
      <h1>New run</h1>
      <div className="bento">
        <section className="card span4" aria-label="1. Raw videos">
          <span className="label">
            1 · Folder with the raw videos (0.avi, 1.avi, …)
          </span>
          <div className="row">
            <span className="mono">{raw || "none chosen"}</span>
            <span className="spacer" />
            <button
              className="btn"
              type="button"
              onClick={() => setPicking(true)}
            >
              Choose folder
            </button>
          </div>
          {raw && (
            <Loading res={info} what="the videos">
              {(i) => (
                <span className="muted">
                  {i.segments} files · {fmtInt(i.frames)} frames · {i.width} ×{" "}
                  {i.height}
                  {i.fps
                    ? ` · ${i.fps} fps · ${fmtDuration(i.frames / i.fps)}`
                    : " · frame rate unknown"}{" "}
                  · {fmtBytes(i.bytes)}
                </span>
              )}
            </Loading>
          )}
        </section>
        {picking && (
          <FolderPicker
            start={raw}
            pickLabel="Use these videos"
            accept={(e) => e.has_avi}
            onPick={(p) => {
              setRaw(p);
              resetOverwrite();
              setPicking(false);
            }}
            onClose={() => setPicking(false)}
          />
        )}
        {info.data && defaults.data && (
          <Preview
            raw={raw}
            info={info.data}
            values={merged}
            onChange={(n, v) => {
              setValues((x) => ({ ...x, [n]: v }));
              setErrors((x) => ({ ...x, [n]: "" }));
            }}
          />
        )}
        <section className="card span4" aria-label="3. Output">
          <span className="label">3 · Name of the run</span>
          <label className="field">
            Run name
            <input
              type="text"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                resetOverwrite();
              }}
            />
          </label>
          {outDir && <span className="muted mono">{outDir}</span>}
          <Toggle
            label="Keep the preprocessed tile store (as big as the raw videos; needed for the Preprocessed view)"
            checked={keep}
            onChange={setKeep}
          />
        </section>
        <section className="card span4" aria-label="4. Parameters">
          <span className="label">4 · Parameters</span>
          <Loading res={defaults} what="parameters">
            {(d) => (
              <ParamsForm
                metas={d.params}
                groups={d.groups}
                values={merged}
                errors={errors}
                onChange={(n, v, e) => {
                  // undefined: the field unmounted; only clear its error
                  if (v !== undefined) setValues((x) => ({ ...x, [n]: v }));
                  setErrors((x) => ({ ...x, [n]: e }));
                }}
              />
            )}
          </Loading>
        </section>
      </div>
      {submitErr && <ErrorNote message={submitErr} />}
      {offered && (
        <Toggle
          label="Overwrite the existing run"
          checked={overwrite}
          onChange={setOverwrite}
        />
      )}
      <div className="row">
        <span className="spacer" />
        {invalid && (
          <span className="error" role="status">
            {failing} {failing === 1 ? "parameter needs" : "parameters need"}{" "}
            fixing
          </span>
        )}
        <button
          className="btn primary"
          type="button"
          disabled={!raw || !info.data || invalid || busy || !defaults.data}
          onClick={start}
        >
          Start run
        </button>
      </div>
    </div>
  );
}
