import { useEffect, useMemo, useState } from "react";
import { getJSON, postJSON } from "../api";
import { ErrorNote, Loading, Toggle } from "../controls";
import { fmtBytes, fmtDuration, fmtInt } from "../format";
import { FolderPicker } from "../home/FolderPicker";
import { useFolders } from "../home/Home";
import { useResource } from "../poll";
import { go } from "../router";
import type { Job, ParamMeta, RawInfo } from "../types";
import { fpsOverride } from "./fps";
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
  const folders = useFolders();
  const [prefilled, setPrefilled] = useState(false);
  // start from the videos of the newest demo download or run, until the user picks a folder
  useEffect(() => {
    if (prefilled || folders.data === undefined) return;
    setPrefilled(true);
    const last = folders.data.folders.find((f) => f.raw);
    if (last?.raw && raw === "") {
      setRaw(last.raw);
      setPicking(false);
    }
  }, [folders.data, prefilled, raw]);
  const info = useResource(raw ? `raw:${raw}` : null, () =>
    getJSON<RawInfo>(`/raw/info?path=${encodeURIComponent(raw)}`),
  );
  const [fpsText, setFpsText] = useState("");
  // a new folder starts from the frame rate of its metadata file
  useEffect(() => setFpsText(info.data?.fps ? String(info.data.fps) : ""), [info.data]);
  const fps = fpsOverride(fpsText, info.data?.fps ?? null);
  const shownFps = fps.fps ?? info.data?.fps ?? null;
  const [name, setName] = useState(() => defaultRunName(new Date()));
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [keep, setKeep] = useState(false);
  const [tune, setTune] = useState(true); // the run is meant to end with estimates; off = tune by hand
  const [diameter, setDiameter] = useState<number | null>(null);
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
  const invalid = failing > 0 || fps.error !== "";
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
      tune,
      tune_args: tune && diameter ? { diameter } : {},
      fps: fps.fps,
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
          <span className="label">1 · Folder with the raw videos</span>
          <span className="muted">
            Each video's file name must start with a number (0.avi, 1.avi, … or
            0_session.avi). The videos are read in the order of that number, so
            2.avi comes before 10.avi; .avi files whose name does not start with
            a number are ignored.
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
                <>
                  <span className="muted">
                    {i.segments} files · {fmtInt(i.frames)} frames · {i.width} ×{" "}
                    {i.height}
                    {shownFps
                      ? ` · ${shownFps} fps · ${fmtDuration(i.frames / shownFps)}`
                      : " · frame rate unknown"}{" "}
                    · {fmtBytes(i.bytes)}
                  </span>
                  <span className="label">Read in this order</span>
                  <span className="mono video-order">{i.names.join(" → ")}</span>
                  {i.skipped.length > 0 && (
                    <span className="muted">
                      Ignored (name does not start with a number):{" "}
                      <span className="mono">{i.skipped.join(", ")}</span>
                    </span>
                  )}
                  <label className="field">
                    Frame rate of the recording (Hz)
                    <input
                      type="number"
                      min={1}
                      max={1000}
                      step="any"
                      placeholder="unknown"
                      value={fpsText}
                      onChange={(e) => setFpsText(e.target.value)}
                    />
                  </label>
                  <span className={fps.error ? "error" : "muted"}>
                    {fps.error ||
                      (i.fps
                        ? fps.fps
                          ? `replaces the ${i.fps} fps of the metadata file`
                          : "from the metadata file (meta.json or metaData.json)"
                        : "no metadata file found; type it in for time axes and durations")}
                  </span>
                </>
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
            info={{ ...info.data, fps: shownFps }}
            values={merged}
            onChange={(n, v) => {
              setValues((x) => ({ ...x, [n]: v }));
              setErrors((x) => ({ ...x, [n]: "" }));
            }}
            onDiameter={setDiameter}
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
          <Toggle
            label={`Auto-tune the seed, init and CNMF parameters after the run (a few more minutes; the proposal opens on the run's Tune page${
              diameter ? `, cell diameter ${diameter} px from step 2` : ""
            }). Switch off only to set every parameter by hand.`}
            checked={tune}
            onChange={setTune}
          />
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
