"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8010";

type Dict = Record<string, unknown>;

type RunSummary = {
  run_id: string;
  run_name: string;
  status: string;
  metric_score: number;
  metric_key: string;
  start_time?: string | number | null;
  params: Record<string, string>;
  metrics: Record<string, number>;
};

type ChatMessage = {
  role: "assistant" | "user";
  content: string;
};

const TOP_K_OPTIONS: Array<number | "all"> = ["all", ...Array.from({ length: 50 }, (_, i) => i + 1)];

function parseDateTime(started: RunSummary["start_time"]): { date: string; time: string } {
  if (started === undefined || started === null) return { date: "-", time: "-" };

  if (typeof started === "number") {
    let ts = started;
    if (ts > 1e12) ts /= 1000;
    const dt = new Date(ts * 1000);
    if (Number.isNaN(dt.getTime())) return { date: "-", time: "-" };
    return { date: dt.toLocaleDateString("ko-KR"), time: dt.toLocaleTimeString("ko-KR") };
  }

  const s = started.trim();
  if (!s) return { date: "-", time: "-" };

  const asNum = Number(s);
  if (!Number.isNaN(asNum)) {
    let ts = asNum;
    if (ts > 1e12) ts /= 1000;
    const dt = new Date(ts * 1000);
    if (!Number.isNaN(dt.getTime())) {
      return { date: dt.toLocaleDateString("ko-KR"), time: dt.toLocaleTimeString("ko-KR") };
    }
  }

  const dt = new Date(s);
  if (Number.isNaN(dt.getTime())) return { date: s, time: "-" };
  return { date: dt.toLocaleDateString("ko-KR"), time: dt.toLocaleTimeString("ko-KR") };
}

function toRunSummaryFromRawRuns(rawRuns: Dict[], metricKey: string): RunSummary[] {
  const metricCol = metricKey.startsWith("metrics.") ? metricKey : `metrics.${metricKey}`;

  return rawRuns
    .map((run): RunSummary | null => {
      const score = run[metricCol];
      const metricScore = Number(score);
      if (Number.isNaN(metricScore)) return null;

      const params: Record<string, string> = {};
      const metrics: Record<string, number> = {};
      Object.entries(run).forEach(([key, value]) => {
        if (key.startsWith("params.") && value !== null && value !== undefined) {
          params[key.replace("params.", "")] = String(value);
        }
        if (key.startsWith("metrics.") && value !== null && value !== undefined) {
          const asNum = Number(value);
          if (!Number.isNaN(asNum)) metrics[key.replace("metrics.", "")] = asNum;
        }
      });

      return {
        run_id: String(run.run_id ?? ""),
        run_name: String(run["tags.mlflow.runName"] ?? ""),
        status: String(run.status ?? ""),
        metric_score: metricScore,
        metric_key: metricCol,
        start_time: (run.start_time as string | number | null | undefined) ?? null,
        params,
        metrics,
      };
    })
    .filter((v): v is RunSummary => v !== null);
}

async function getJson<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(`${API_BASE_URL}${path}`);
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  }
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`${path} 요청 실패 (${res.status})`);
  return (await res.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} 요청 실패 (${res.status})`);
  return (await res.json()) as T;
}

export default function Page() {
  const [experimentNames, setExperimentNames] = useState<string[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState("");
  const [metricKeys, setMetricKeys] = useState<string[]>([]);
  const [selectedMetric, setSelectedMetric] = useState("");
  const [selectedTopK, setSelectedTopK] = useState<number | "all">("all");

  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runsMeta, setRunsMeta] = useState<{ experiment_name?: string; metric_key?: string; top_k?: number | "all" }>({});
  const [experimentDetail, setExperimentDetail] = useState<Dict | null>(null);
  const [showExperimentModal, setShowExperimentModal] = useState(false);

  const [reportText, setReportText] = useState("");
  const [reportMeta, setReportMeta] = useState<{ experiment_name?: string; metric_key?: string }>({});
  const [reportModel, setReportModel] = useState("");
  const [reportInfo, setReportInfo] = useState("");

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    { role: "assistant", content: "안녕하세요. 실험 관련 질문을 입력해 주세요." },
  ]);
  const [chatInput, setChatInput] = useState("");
  const [showChatModal, setShowChatModal] = useState(false);

  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadNames = async () => {
      try {
        const res = await getJson<{ experiment_names: string[] }>("/mlflow/names");
        setExperimentNames(res.experiment_names ?? []);
      } catch (e) {
        setError(e instanceof Error ? e.message : "실험 목록 조회 실패");
      }
    };
    void loadNames();
  }, []);

  useEffect(() => {
    if (!selectedExperiment) {
      setMetricKeys([]);
      setSelectedMetric("");
      return;
    }

    const loadMetricKeys = async () => {
      try {
        const res = await getJson<{ metric_keys: string[] }>("/mlflow/metric-keys", {
          experiment_name: selectedExperiment,
        });
        setMetricKeys(res.metric_keys ?? []);
      } catch (e) {
        setMetricKeys([]);
        setSelectedMetric("");
        setError(e instanceof Error ? e.message : "Metric key 목록 조회 실패");
      }
    };
    void loadMetricKeys();
  }, [selectedExperiment]);

  const canSubmit = useMemo(() => Boolean(selectedExperiment && selectedMetric), [selectedExperiment, selectedMetric]);

  const onViewExperiment = async () => {
    if (!canSubmit) {
      setError("experiment name과 metric key를 먼저 선택해주세요.");
      return;
    }
    setLoading(true);
    setError("");
    setInfo("");
    try {
      const detailRes = await getJson<{ experiment: Dict }>("/mlflow/experiment", {
        experiment_name: selectedExperiment,
      });
      setExperimentDetail(detailRes.experiment ?? {});
      setShowExperimentModal(true);

      let runList: RunSummary[] = [];
      if (selectedTopK === "all") {
        const runsRes = await getJson<{ runs: Dict[] }>("/mlflow/runs", { experiment_name: selectedExperiment });
        runList = toRunSummaryFromRawRuns(runsRes.runs ?? [], selectedMetric).sort((a, b) => b.metric_score - a.metric_score);
      } else {
        const topRunsRes = await getJson<{ top_runs: RunSummary[] }>("/mlflow/top-runs", {
          experiment_name: selectedExperiment,
          metric_key: selectedMetric,
          top_k: String(selectedTopK),
          ascending: "false",
        });
        runList = topRunsRes.top_runs ?? [];
      }
      setRuns(runList);
      setRunsMeta({ experiment_name: selectedExperiment, metric_key: selectedMetric, top_k: selectedTopK });
    } catch (e) {
      setError(e instanceof Error ? e.message : "조회 실패");
      setRuns([]);
    } finally {
      setLoading(false);
    }
  };

  const onGenerateReport = async () => {
    if (!canSubmit) {
      setError("보고서 생성을 위해 experiment name과 metric key를 선택해주세요.");
      return;
    }

    const reportTopK = selectedTopK === "all" ? 3 : Math.min(selectedTopK, 3);
    setLoading(true);
    setError("");
    setInfo(selectedTopK === "all" || (typeof selectedTopK === "number" && selectedTopK > 3) ? "보고서 API는 상위 3개 run만 사용합니다." : "");

    try {
      const topRunsRes = await getJson<{ top_runs: RunSummary[] }>("/mlflow/top-runs", {
        experiment_name: selectedExperiment,
        metric_key: selectedMetric,
        top_k: String(reportTopK),
        ascending: "false",
      });

      const topRuns = topRunsRes.top_runs ?? [];
      if (!topRuns.length) {
        setError("보고서를 만들 run 데이터가 없습니다.");
        return;
      }

      const reportRes = await postJson<{ report: string; model: string; metadata: Dict }>("/report/", {
        experiment_name: selectedExperiment,
        metric_key: selectedMetric,
        top_runs: topRuns,
      });

      setReportText(reportRes.report ?? "");
      setReportModel(reportRes.model ?? "");
      setReportInfo(JSON.stringify(reportRes.metadata ?? {}, null, 2));
      setReportMeta({ experiment_name: selectedExperiment, metric_key: selectedMetric });
    } catch (e) {
      setError(e instanceof Error ? e.message : "보고서 생성 실패");
    } finally {
      setLoading(false);
    }
  };

  const onChatSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const message = chatInput.trim();
    if (!message) return;

    setChatInput("");
    setChatMessages((prev) => [...prev, { role: "user", content: message }]);

    try {
      const payload: Record<string, string> = { message };
      if (selectedExperiment.trim()) payload.experiment_name = selectedExperiment.trim();
      if (selectedMetric.trim()) payload.metric_key = selectedMetric.trim();
      if (reportText) {
        payload.report_text = reportText;
        if (reportMeta.experiment_name) payload.experiment_name = reportMeta.experiment_name;
        if (reportMeta.metric_key) payload.metric_key = reportMeta.metric_key;
      }
      const chatRes = await postJson<{ answer: string }>("/chat/ask", payload);
      setChatMessages((prev) => [...prev, { role: "assistant", content: chatRes.answer || "응답이 비어 있습니다." }]);
    } catch (e) {
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: e instanceof Error ? e.message : "챗봇 요청 실패" },
      ]);
    }
  };

  return (
    <main>
      <h1>MLflow + FastAPI + Next.js 템플릿</h1>

      <section className="card">
        <div className="grid grid-3">
          <div>
            <label htmlFor="experiment">Experiment name</label>
            <select id="experiment" value={selectedExperiment} onChange={(e) => setSelectedExperiment(e.target.value)}>
              <option value="">선택하세요</option>
              {experimentNames.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="metric">Metric key</label>
            <select id="metric" value={selectedMetric} onChange={(e) => setSelectedMetric(e.target.value)}>
              <option value="">선택하세요</option>
              {metricKeys.map((metric) => (
                <option key={metric} value={metric}>
                  {metric}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="topk">Top K</label>
            <select
              id="topk"
              value={String(selectedTopK)}
              onChange={(e) => setSelectedTopK(e.target.value === "all" ? "all" : Number(e.target.value))}
            >
              {TOP_K_OPTIONS.map((opt) => (
                <option key={String(opt)} value={String(opt)}>
                  {opt === "all" ? "Default(전체)" : opt}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="row" style={{ marginTop: 10 }}>
          <button type="button" onClick={() => void onViewExperiment()} disabled={loading}>
            실험 정보 조회
          </button>
          <button type="button" className="secondary" onClick={() => void onGenerateReport()} disabled={loading}>
            실험 보고서 생성
          </button>
        </div>
        {loading && <p className="muted">요청 처리 중...</p>}
        {info && <p className="info">{info}</p>}
        {error && <p className="error">{error}</p>}
      </section>

      <section className="card">
        <h3>Runs</h3>
        {!runs.length ? (
          <p className="muted">실험 정보 조회를 누르면 runs가 표시됩니다.</p>
        ) : (
          <>
            <p className="muted">
              experiment `{runsMeta.experiment_name}` · `{runsMeta.metric_key}` 기준 {runs.length}개 (K={String(runsMeta.top_k)})
            </p>
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Score</th>
                  <th>Status</th>
                  <th>Date</th>
                  <th>Time</th>
                  <th>Parameters</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const { date, time } = parseDateTime(run.start_time);
                  return (
                    <tr key={run.run_id}>
                      <td>{run.run_name || "-"}</td>
                      <td>{run.metric_score}</td>
                      <td>{run.status || "-"}</td>
                      <td>{date}</td>
                      <td>{time}</td>
                      <td>
                        <pre>{JSON.stringify({ run_id: run.run_id, params: run.params, metrics: run.metrics }, null, 2)}</pre>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        )}
      </section>

      <section className="card">
        <div className="row">
          <button type="button" onClick={() => setShowChatModal(true)}>
            Chat
          </button>
        </div>
        {reportText ? (
          <>
            <p className="muted" style={{ marginTop: 10 }}>
              모델: `{reportModel || "-"}` · 메타데이터:
            </p>
            <pre>{reportInfo}</pre>
            <h3>Report</h3>
            <pre>{reportText}</pre>
          </>
        ) : (
          <p className="muted" style={{ marginTop: 10 }}>
            아직 생성된 보고서가 없습니다.
          </p>
        )}
      </section>

      {showExperimentModal && (
        <div className="modal" onClick={() => setShowExperimentModal(false)}>
          <div className="panel" onClick={(e) => e.stopPropagation()}>
            <h3>실험 정보</h3>
            <pre>{JSON.stringify(experimentDetail ?? {}, null, 2)}</pre>
            <div className="row">
              <button type="button" onClick={() => setShowExperimentModal(false)}>
                닫기
              </button>
            </div>
          </div>
        </div>
      )}

      {showChatModal && (
        <div className="modal" onClick={() => setShowChatModal(false)}>
          <div className="panel chatPanel" onClick={(e) => e.stopPropagation()}>
            <h3>실험 Q&A 챗봇</h3>
            <div className="chatLog">
              {chatMessages.map((m, idx) => (
                <div className="chatBubble" key={`${m.role}-${idx}`}>
                  <div className="chatRole">{m.role}</div>
                  <div>{m.content}</div>
                </div>
              ))}
            </div>
            <form onSubmit={onChatSubmit} className="row">
              <input value={chatInput} onChange={(e) => setChatInput(e.target.value)} placeholder="질문을 입력하세요" />
              <button type="submit">전송</button>
            </form>
            <div className="row" style={{ marginTop: 8 }}>
              <button type="button" className="secondary" onClick={() => setShowChatModal(false)}>
                닫기
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
