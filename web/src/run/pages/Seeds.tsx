import { useState } from "react";
import { getJSON } from "../../api";
import { Loading, Segmented } from "../../controls";
import { fmtInt } from "../../format";
import { ImageView } from "../../ImageView";
import { useResource } from "../../poll";
import type { Seeds as SeedsT } from "../../types";
import { rerunQuery, useRun } from "../RunContext";

type Step = keyof SeedsT;
const STEPS: { value: Step; label: string }[] = [
  { value: "init", label: "Initial" },
  { value: "pnr", label: "After PNR" },
  { value: "merged", label: "Merged" },
];

export function Seeds() {
  const { summary: s, maxProj, rerun } = useRun();
  const [step, setStep] = useState<Step>("merged");
  const res = useResource(`seeds:${s.id}:${rerun ?? ""}`, () => getJSON<SeedsT>(`/runs/${s.id}/seeds${rerunQuery(rerun)}`));
  return (
    <div className="bento">
      <section className="card span3" aria-label="Seeds on the max projection">
        <div className="row">
          <span className="label">Seeds on the max projection</span>
          <span className="spacer" />
          <Segmented label="Seed step" options={STEPS} value={step} onChange={setStep} />
        </div>
        <Loading res={res} what="the seeds (older runs: about 2.5 min on 3 h)">
          {(sd) => <ImageView image={maxProj} points={sd[step]} label={`${step} seeds`} />}
        </Loading>
      </section>
      <section className="card" aria-label="Seed counts">
        <span className="label">Seeds per step</span>
        {res.data &&
          STEPS.map((x) => (
            <div key={x.value}>
              <span className="muted">{x.label}</span>
              <div className="big" style={{ fontSize: 28 }}>
                {fmtInt(res.data![x.value].length)}
              </div>
            </div>
          ))}
      </section>
    </div>
  );
}
