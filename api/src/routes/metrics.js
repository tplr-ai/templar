const express = require('express');
const router = express.Router();
const { InfluxDB } = require('@influxdata/influxdb-client');
const influxConfig = require('../config/influxdb');

// Create InfluxDB client & Query API instance
const influxDB = new InfluxDB({ url: influxConfig.url, token: influxConfig.token });
const queryApi = influxDB.getQueryApi(influxConfig.org);

/**
 * Query InfluxDB, with detailed debug logging for each callback.
 *
 * @param {string} query - The Flux query.
 * @param {number} timeoutMs - Timeout in milliseconds.
 * @returns {Promise<Array<any>>} Resolves when the query completes.
 */
function queryWithTimeout(query, timeoutMs = 60000) {
  return new Promise((resolve, reject) => {
    const results = [];
    const startTime = Date.now();
    console.debug("=== Starting query ===");
    console.debug("Timeout set to:", timeoutMs, "ms");
    console.debug("Query:", query);

    const timer = setTimeout(() => {
      const elapsed = Date.now() - startTime;
      console.error(`Query timed out after ${elapsed} ms`);
      reject(new Error('Query timed out'));
    }, timeoutMs);

    queryApi.queryRows(query, {
      next(row, tableMeta) {
        const elapsed = Date.now() - startTime;
        const obj = tableMeta.toObject(row);
        results.push(obj);
        console.debug(`Row received at ${elapsed} ms:`, obj);
      },
      error(err) {
        clearTimeout(timer);
        const elapsed = Date.now() - startTime;
        console.error(`Error after ${elapsed} ms:`, err);
        reject(err);
      },
      complete() {
        clearTimeout(timer);
        const duration = Date.now() - startTime;
        console.debug(`Query complete after ${duration} ms; total rows: ${results.length}`);
        resolve(results);
      },
    });
  });
}

// GET /api/metrics/losses
router.get('/losses', async (req, res, next) => {
  // Use the version parsed from __init__.py if not defined in environment variables.
  const minerVersion = influxConfig.version;
  const fluxQuery = `
    from(bucket: "${influxConfig.bucket}")
      |> range(start: -1h)
      |> filter(fn: (r) => r["_measurement"] == "templar_metrics")
      |> filter(fn: (r) => r["role"] == "miner" and r["version"] == "${minerVersion}")
      |> filter(fn: (r) => r["_field"] == "loss")
      |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
      |> yield(name: "mean")
  `;

  try {
    console.debug("Sending query from /losses endpoint...");
    const results = await queryWithTimeout(fluxQuery, 60000);
    console.debug("Returning results from /losses endpoint.");
    res.json(results);
  } catch (err) {
    console.error("Error in /losses route:", err);
    next(err);
  }
});

/**
 * GET /api/metrics/tokens-per-sec
 * Returns tokens per second metrics.
 * Query Parameters:
 * - uid (optional)
 * - timeRange (optional, default '-1h')
 * - aggregate (optional): if 'true', returns aggregated mean across all series, versions, and uids.
 */
router.get('/tokens-per-sec', async (req, res, next) => {
  const { uid, timeRange, aggregate } = req.query;
  const actualTimeRange = timeRange || '-1h';
  const minerVersion = influxConfig.version; // use parsed version from __init__.py if available
  let fluxQuery = '';

  if (aggregate === 'true') {
    // This query resets grouping and calculates the global mean
    fluxQuery = `
      from(bucket: "${influxConfig.bucket}")
        |> range(start: ${actualTimeRange})
        |> filter(fn: (r) => r["_measurement"] == "templar_metrics")
        |> filter(fn: (r) => r["_field"] == "tokens_per_sec")
        |> filter(fn: (r) => r["version"] == "${minerVersion}")
        |> group()
        |> mean()
        |> yield(name: "mean")
    `;
  } else if (uid) {
    fluxQuery = `
      from(bucket: "${influxConfig.bucket}")
        |> range(start: ${actualTimeRange})
        |> filter(fn: (r) => r["_measurement"] == "templar_metrics")
        |> filter(fn: (r) => r["uid"] == "${uid}")
        |> filter(fn: (r) => r["_field"] == "tokens_per_sec")
        |> group()   // keep original grouping to separate series
        |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
        |> yield(name: "mean")
    `;
  } else {
    // If no UID or aggregate specified, use the simple windowed query.
    fluxQuery = `
      from(bucket: "${influxConfig.bucket}")
        |> range(start: ${actualTimeRange})
        |> filter(fn: (r) => r["_measurement"] == "templar_metrics")
        |> filter(fn: (r) => r["_field"] == "tokens_per_sec")
        |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
        |> yield(name: "mean")
    `;
  }

  console.debug("Tokens-per-sec query:", fluxQuery);

  let results = [];
  try {
    await new Promise((resolve, reject) => {
      queryApi.queryRows(fluxQuery, {
        next(row, tableMeta) {
          results.push(tableMeta.toObject(row));
        },
        error(err) {
          console.error("Error in tokens-per-sec query:", err);
          reject(err);
        },
        complete() {
          resolve();
        }
      });
    });
    res.json({
      success: true,
      data: results,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    next(err);
  }
});

// GET /api/metrics/benchmark-scores
router.get('/benchmark-scores', async (req, res, next) => {
  const { timeRange } = req.query;
  const actualTimeRange = timeRange || '-30d';

  const fluxQuery = `
    from(bucket: "${influxConfig.bucket}")
      |> range(start: ${actualTimeRange})
      |> filter(fn: (r) => r._measurement == "templar_benchmark")
      |> filter(fn: (r) => r._field == "score")
      |> filter(fn: (r) => r.role == "evaluator")
      |> group(columns: ["task"])
      |> last()
      |> yield(name: "latest_scores")
  `;

  try {
    console.debug("Sending query from /benchmark-scores endpoint...");
    const results = await queryWithTimeout(fluxQuery, 60000);

    // Transform results into a more user-friendly format
    const formattedResults = results.map(row => ({
      task: row.task,
      score: row._value,
      global_step: row.global_step,
      window: row.window,
      timestamp: row._time
    }));

    console.debug("Returning results from /benchmark-scores endpoint.");
    res.json({
      success: true,
      data: formattedResults,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    console.error("Error in /benchmark-scores route:", err);
    next(err);
  }
});

module.exports = router;