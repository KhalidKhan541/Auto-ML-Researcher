/**
 * Cloudflare Worker — ML Experiment Agent API
 *
 * Routes:
 *   POST /api/experiment        — start a new experiment
 *   GET  /api/experiment/:id    — poll for state/progress
 *   POST /api/experiment/:id/cancel — cancel a running experiment
 *   GET  /api/experiments       — list all experiments (from KV)
 *
 * The Worker calls your Python backend (deployed on e.g. Modal / Railway)
 * which runs the LangGraph agent. For full serverless, swap the Python
 * backend call with Workers AI + Durable Objects (see notes at bottom).
 */

export interface Env {
  EXPERIMENTS_KV: KVNamespace;   // bind in wrangler.toml
  AGENT_BACKEND_URL: string;     // secret: your Python service URL
  AGENT_API_KEY: string;         // secret: auth key for backend
}

// ── CORS helper ──────────────────────────────────────────────────────────────

function cors(res: Response): Response {
  const h = new Headers(res.headers);
  h.set("Access-Control-Allow-Origin", "*");
  h.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  h.set("Access-Control-Allow-Headers", "Content-Type, Authorization");
  return new Response(res.body, { status: res.status, headers: h });
}

function json(data: unknown, status = 200): Response {
  return cors(
    new Response(JSON.stringify(data), {
      status,
      headers: { "Content-Type": "application/json" },
    })
  );
}

// ── Experiment store (KV) ─────────────────────────────────────────────────────

interface Experiment {
  id: string;
  goal: string;
  status: "queued" | "running" | "converged" | "max_iter" | "failed";
  created_at: number;
  updated_at: number;
  iteration: number;
  max_iterations: number;
  iteration_log: IterationEntry[];
  current_node?: string;
  error?: string;
}

interface IterationEntry {
  iteration: number;
  hypothesis: string;
  architecture: string;
  metrics: Record<string, number | null>;
  verdict: string;
}

async function saveExperiment(env: Env, exp: Experiment): Promise<void> {
  await env.EXPERIMENTS_KV.put(`exp:${exp.id}`, JSON.stringify(exp), {
    expirationTtl: 60 * 60 * 24 * 7, // 7 days
  });
}

async function getExperiment(env: Env, id: string): Promise<Experiment | null> {
  const raw = await env.EXPERIMENTS_KV.get(`exp:${id}`);
  return raw ? JSON.parse(raw) : null;
}

async function listExperiments(env: Env): Promise<Experiment[]> {
  const keys = await env.EXPERIMENTS_KV.list({ prefix: "exp:" });
  const results = await Promise.all(
    keys.keys.map(async (k) => {
      const raw = await env.EXPERIMENTS_KV.get(k.name);
      return raw ? JSON.parse(raw) : null;
    })
  );
  return results.filter(Boolean).sort((a, b) => b.created_at - a.created_at);
}

// ── Backend proxy ─────────────────────────────────────────────────────────────

async function callBackend(env: Env, path: string, body?: unknown) {
  const res = await fetch(`${env.AGENT_BACKEND_URL}${path}`, {
    method: body ? "POST" : "GET",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": env.AGENT_API_KEY,
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Backend error ${res.status}: ${err}`);
  }
  return res.json();
}

// ── Route handlers ────────────────────────────────────────────────────────────

async function startExperiment(req: Request, env: Env): Promise<Response> {
  const body = await req.json<{ goal: string; max_iterations?: number }>();
  if (!body?.goal?.trim()) {
    return json({ error: "goal is required" }, 400);
  }

  const id = crypto.randomUUID().slice(0, 8);
  const exp: Experiment = {
    id,
    goal: body.goal,
    status: "queued",
    created_at: Date.now(),
    updated_at: Date.now(),
    iteration: 0,
    max_iterations: body.max_iterations ?? 3,
    iteration_log: [],
    current_node: "hypothesize",
  };

  await saveExperiment(env, exp);

  // Fire-and-forget: ask the Python backend to run the experiment
  // The backend will PATCH /api/experiment/:id/update as each node completes
  callBackend(env, "/run", { experiment_id: id, goal: body.goal, max_iterations: exp.max_iterations })
    .then(async (result) => {
      const updated: Experiment = {
        ...exp,
        status: result.status ?? "converged",
        iteration: result.iteration ?? 0,
        iteration_log: result.iteration_log ?? [],
        updated_at: Date.now(),
      };
      await saveExperiment(env, updated);
    })
    .catch(async (err) => {
      exp.status = "failed";
      exp.error = String(err);
      exp.updated_at = Date.now();
      await saveExperiment(env, exp);
    });

  return json({ experiment_id: id, status: "queued" }, 202);
}

async function getExperimentHandler(id: string, env: Env): Promise<Response> {
  const exp = await getExperiment(env, id);
  if (!exp) return json({ error: "not found" }, 404);
  return json(exp);
}

async function listExperimentsHandler(env: Env): Promise<Response> {
  const exps = await listExperiments(env);
  return json(exps);
}

async function updateExperiment(req: Request, id: string, env: Env): Promise<Response> {
  // Called by the Python backend to push progress updates
  const patch = await req.json<Partial<Experiment>>();
  const exp = await getExperiment(env, id);
  if (!exp) return json({ error: "not found" }, 404);

  const updated = { ...exp, ...patch, updated_at: Date.now() };
  await saveExperiment(env, updated);
  return json({ ok: true });
}

// ── Main handler ──────────────────────────────────────────────────────────────

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const { pathname, method } = { pathname: url.pathname, method: request.method };

    if (method === "OPTIONS") return cors(new Response(null, { status: 204 }));

    // Serve frontend (static assets handled by Cloudflare Pages — see wrangler.toml)
    if (pathname === "/" || !pathname.startsWith("/api/")) {
      return cors(new Response("Use Cloudflare Pages for the frontend.", { status: 200 }));
    }

    try {
      // POST /api/experiment
      if (pathname === "/api/experiment" && method === "POST") {
        return await startExperiment(request, env);
      }

      // GET /api/experiments
      if (pathname === "/api/experiments" && method === "GET") {
        return await listExperimentsHandler(env);
      }

      // GET /api/experiment/:id
      const getMatch = pathname.match(/^\/api\/experiment\/([^/]+)$/);
      if (getMatch && method === "GET") {
        return await getExperimentHandler(getMatch[1], env);
      }

      // PATCH /api/experiment/:id/update  (internal backend callback)
      const updateMatch = pathname.match(/^\/api\/experiment\/([^/]+)\/update$/);
      if (updateMatch && method === "PATCH") {
        return await updateExperiment(request, updateMatch[1], env);
      }

      return json({ error: "not found" }, 404);
    } catch (err) {
      return json({ error: String(err) }, 500);
    }
  },
};

/*
 * ── wrangler.toml (place in project root) ──────────────────────────────────
 *
 * name = "ml-agent-worker"
 * main = "worker/index.ts"
 * compatibility_date = "2024-09-01"
 *
 * [[kv_namespaces]]
 * binding = "EXPERIMENTS_KV"
 * id = "YOUR_KV_NAMESPACE_ID"
 *
 * [vars]
 * AGENT_BACKEND_URL = "https://your-python-service.railway.app"
 *
 * # Secrets (set via: wrangler secret put AGENT_API_KEY)
 * # AGENT_API_KEY
 *
 * # For Pages + Worker together:
 * # Deploy frontend to Cloudflare Pages, set custom domain
 * # Worker handles /api/* routes via Pages Functions or standalone Worker route
 */
