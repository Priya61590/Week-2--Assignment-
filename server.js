/**
 * Node.js Delivery Time Service
 * Wraps the Python Flask ML API and exposes a clean REST endpoint
 *
 * Run:  node server.js
 * Port: 3000
 */

const http  = require("http");
const https = require("https");
const url   = require("url");

// ── Config ───────────────────────────────────
const NODE_PORT  = process.env.PORT || 3000;
const ML_API_URL = process.env.ML_API_URL || "http://localhost:5000";

// ── Utility: call the Flask ML API ───────────
function callMLApi(path, method, body) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(ML_API_URL + path);
    const isHttps   = parsedUrl.protocol === "https:";
    const lib       = isHttps ? https : http;

    const payload = body ? JSON.stringify(body) : null;
    const options = {
      hostname : parsedUrl.hostname,
      port     : parsedUrl.port || (isHttps ? 443 : 80),
      path     : parsedUrl.pathname,
      method   : method,
      headers  : {
        "Content-Type"   : "application/json",
        "Content-Length" : payload ? Buffer.byteLength(payload) : 0,
      },
    };

    const req = lib.request(options, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        try {
          resolve({ status: res.statusCode, body: JSON.parse(data) });
        } catch {
          resolve({ status: res.statusCode, body: { raw: data } });
        }
      });
    });

    req.on("error", reject);
    if (payload) req.write(payload);
    req.end();
  });
}

// ── Route handlers ────────────────────────────
async function handlePredict(res, bodyStr) {
  let input;
  try {
    input = JSON.parse(bodyStr);
  } catch {
    return sendJson(res, 400, { error: "Invalid JSON body" });
  }

  // Validate required fields
  const required = [
    "source_city", "destination_city",
    "parcel_weight_kg", "distance_km",
    "weather_condition", "vehicle_type",
  ];
  const missing = required.filter((k) => !(k in input));
  if (missing.length) {
    return sendJson(res, 400, {
      error: "Missing required fields",
      missing,
      example: {
        source_city: "Mumbai", destination_city: "Pune",
        parcel_weight_kg: 2.5, distance_km: 148,
        weather_condition: "Clear", vehicle_type: "Bike",
      },
    });
  }

  try {
    const mlRes = await callMLApi("/predict", "POST", input);

    if (mlRes.status === 200) {
      const result = mlRes.body;
      // Add a friendly human-readable message
      result.message = `Your parcel from ${result.source_city} to ${result.destination_city} is expected to arrive in approximately ${result.predicted_delivery_hours} hours.`;
      sendJson(res, 200, result);
    } else {
      sendJson(res, mlRes.status, {
        error             : "Prediction service error",
        details           : mlRes.body,
        prediction_failed : true,
      });
    }
  } catch (err) {
    sendJson(res, 503, {
      error   : "ML service unavailable. Is the Flask API running?",
      details : err.message,
    });
  }
}

async function handleInfo(res) {
  try {
    const mlRes = await callMLApi("/info", "GET", null);
    sendJson(res, mlRes.status, mlRes.body);
  } catch (err) {
    sendJson(res, 503, { error: "ML service unavailable", details: err.message });
  }
}

async function handleHealth(res) {
  try {
    const mlRes = await callMLApi("/health", "GET", null);
    sendJson(res, 200, {
      node_service : "ok",
      ml_api       : mlRes.status === 200 ? "ok" : "error",
      ml_api_url   : ML_API_URL,
    });
  } catch {
    sendJson(res, 200, {
      node_service : "ok",
      ml_api       : "unreachable",
      ml_api_url   : ML_API_URL,
    });
  }
}

// ── HTTP Server ───────────────────────────────
const server = http.createServer((req, res) => {
  const parsed   = url.parse(req.url, true);
  const pathname = parsed.pathname;
  const method   = req.method;

  // CORS headers
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (method === "OPTIONS") { res.writeHead(204); return res.end(); }

  if (pathname === "/" && method === "GET") {
    return sendJson(res, 200, {
      service   : "Delivery Time Node.js Service",
      version   : "1.0",
      endpoints : {
        "POST /api/predict" : "Predict delivery time",
        "GET  /api/info"    : "View model info & valid input values",
        "GET  /api/health"  : "Health check",
      },
    });
  }

  if (pathname === "/api/health" && method === "GET") {
    return handleHealth(res);
  }

  if (pathname === "/api/info" && method === "GET") {
    return handleInfo(res);
  }

  if (pathname === "/api/predict" && method === "POST") {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => handlePredict(res, body));
    return;
  }

  sendJson(res, 404, { error: "Route not found" });
});

function sendJson(res, status, data) {
  const payload = JSON.stringify(data, null, 2);
  res.writeHead(status, {
    "Content-Type"   : "application/json",
    "Content-Length" : Buffer.byteLength(payload),
  });
  res.end(payload);
}

server.listen(NODE_PORT, () => {
  console.log(`[Node.js] Delivery service running on http://localhost:${NODE_PORT}`);
  console.log(`[Node.js] ML API target: ${ML_API_URL}`);
  console.log(`[Node.js] POST http://localhost:${NODE_PORT}/api/predict`);
});
