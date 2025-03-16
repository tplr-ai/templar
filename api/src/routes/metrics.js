import express from 'express';
import influxConfig from '../config/influxdb.js';
import influxService from '../services/influxService.js';

const router = express.Router();

// GET /api/metrics/training-step
router.get('/training-step', async (req, res, next) => {
  const { timeRange, uid, window, global_step } = req.query;
  const actualTimeRange = timeRange || '-2h';
  
  try {
    const response = await influxService.queryMetrics('training_step', {
      uid,
      window,
      global_step
    }, actualTimeRange);
    
    res.json(response);
  } catch (err) {
    console.error("Error in /training-step route:", err);
    next(err);
  }
});

// GET /api/metrics/validator-scores
router.get('/validator-scores', async (req, res, next) => {
  const { timeRange, eval_uid, window, global_step } = req.query;
  const actualTimeRange = timeRange || '-6h';
  
  try {
    const response = await influxService.queryMetrics('validator_scores', {
      eval_uid,
      window,
      global_step
    }, actualTimeRange);
    
    res.json(response);
  } catch (err) {
    console.error("Error in /validator-scores route:", err);
    next(err);
  }
});

// GET /api/metrics/validator-window
router.get('/validator-window', async (req, res, next) => {
  const { timeRange, window, global_step } = req.query;
  const actualTimeRange = timeRange || '-6h';
  
  try {
    const response = await influxService.queryMetrics('validator_window', {
      window,
      global_step
    }, actualTimeRange);
    
    res.json(response);
  } catch (err) {
    console.error("Error in /validator-window route:", err);
    next(err);
  }
});

// GET /api/metrics/benchmark-task
router.get('/benchmark-task', async (req, res, next) => {
  const { timeRange, task, global_step, window, block } = req.query;
  const actualTimeRange = timeRange || '-30d';
  
  try {
    // Special case with custom query for benchmark task scores
    let fluxQuery = `
      from(bucket: "${influxConfig.bucket}")
        |> range(start: ${actualTimeRange})
        |> filter(fn: (r) => r["_measurement"] == "benchmark_task")
        |> filter(fn: (r) => r["_field"] == "score")
    `;
    
    if (task) {
      fluxQuery += `|> filter(fn: (r) => r["task"] == "${task}")`;
    }
    
    if (global_step) {
      fluxQuery += `|> filter(fn: (r) => r["global_step"] == "${global_step}")`;
    }
    
    if (window) {
      fluxQuery += `|> filter(fn: (r) => r["window"] == "${window}")`;
    }
    
    if (block) {
      fluxQuery += `|> filter(fn: (r) => r["block"] == "${block}")`;
    }
    
    fluxQuery += `
      |> group(columns: ["task"])
      |> last()
    `;
    
    const results = await influxService.queryWithTimeout(fluxQuery);
    
    const formattedResults = results.map(row => ({
      task: row.task,
      score: row._value,
      global_step: row.global_step,
      window: row.window,
      block: row.block,
      timestamp: row._time
    }));
    
    res.json({
      success: true,
      data: formattedResults,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    console.error("Error in /benchmark-task route:", err);
    next(err);
  }
});

// GET /api/metrics/benchmark-metrics
router.get('/benchmark-metrics', async (req, res, next) => {
  const { timeRange, global_step, window, block } = req.query;
  const actualTimeRange = timeRange || '-30d';
  
  try {
    const response = await influxService.queryMetrics('benchmark_metrics', {
      global_step,
      window,
      block
    }, actualTimeRange);
    
    res.json(response);
  } catch (err) {
    console.error("Error in /benchmark-metrics route:", err);
    next(err);
  }
});

// GET /api/metrics/benchmark-summary
router.get('/benchmark-summary', async (req, res, next) => {
  const { timeRange, global_step, window, block } = req.query;
  const actualTimeRange = timeRange || '-30d';
  
  try {
    const response = await influxService.queryMetrics('benchmark_summary', {
      global_step,
      window,
      block
    }, actualTimeRange);
    
    res.json(response);
  } catch (err) {
    console.error("Error in /benchmark-summary route:", err);
    next(err);
  }
});

// GET /api/metrics/similarity
router.get('/similarity', async (req, res, next) => {
  const { timeRange, peer_uid, step } = req.query;
  const actualTimeRange = timeRange || '-6h';
  
  try {
    const response = await influxService.queryMetrics('similarity', {
      peer_uid,
      step
    }, actualTimeRange);
    
    res.json(response);
  } catch (err) {
    console.error("Error in /similarity route:", err);
    next(err);
  }
});

// GET /api/metrics/gradient-analysis
router.get('/gradient-analysis', async (req, res, next) => {
  const { timeRange, peer_uid, window, step } = req.query;
  const actualTimeRange = timeRange || '-6h';
  
  try {
    const response = await influxService.queryMetrics('gradient_analysis', {
      peer_uid,
      window,
      step
    }, actualTimeRange);
    
    res.json(response);
  } catch (err) {
    console.error("Error in /gradient-analysis route:", err);
    next(err);
  }
});

// GET /api/metrics/validator-inactivity
router.get('/validator-inactivity', async (req, res, next) => {
  const { timeRange, uid, window } = req.query;
  const actualTimeRange = timeRange || '-6h';
  
  try {
    const response = await influxService.queryMetrics('validator_inactivity', {
      uid,
      window
    }, actualTimeRange);
    
    res.json(response);
  } catch (err) {
    console.error("Error in /validator-inactivity route:", err);
    next(err);
  }
});

// GET /api/metrics/validator-slash
router.get('/validator-slash', async (req, res, next) => {
  const { timeRange, eval_uid, window, global_step, reason_code } = req.query;
  const actualTimeRange = timeRange || '-6h';
  
  try {
    const response = await influxService.queryMetrics('validator_slash', {
      eval_uid,
      window,
      global_step,
      reason_code
    }, actualTimeRange);
    
    res.json(response);
  } catch (err) {
    console.error("Error in /validator-slash route:", err);
    next(err);
  }
});

// GET /api/metrics/templar-metrics-v2
router.get('/templar-metrics-v2', async (req, res, next) => {
  const { timeRange, uid, role, window, global_step, eval_uid, field } = req.query;
  const actualTimeRange = timeRange || '-6h';
  
  try {
    const response = await influxService.queryMetrics('templar_metrics_v2', {
      uid,
      role,
      window,
      global_step,
      eval_uid,
      field
    }, actualTimeRange);
    
    res.json(response);
  } catch (err) {
    console.error("Error in /templar-metrics-v2 route:", err);
    next(err);
  }
});

// For backward compatibility
// GET /api/metrics/losses
router.get('/losses', async (req, res, next) => {
  const { timeRange, uid, window } = req.query;
  const actualTimeRange = timeRange || '-2h';
  
  try {
    // Use legacy method in service
    const result = await influxService.getLosses({
      window,
      timeRange: actualTimeRange,
      uid
    });
    
    res.json(result);
  } catch (err) {
    console.error("Error in /losses route:", err);
    next(err);
  }
});

// For backward compatibility
// GET /api/metrics/tokens-per-sec
router.get('/tokens-per-sec', async (req, res, next) => {
  const { uid, timeRange, aggregate } = req.query;
  const actualTimeRange = timeRange || '-1h';
  
  try {
    // Use legacy method in service
    const result = await influxService.getTokensPerSec({
      uid,
      timeRange: actualTimeRange,
      aggregate: aggregate === 'true'
    });
    
    res.json(result);
  } catch (err) {
    console.error("Error in /tokens-per-sec route:", err);
    next(err);
  }
});

// For backward compatibility
// GET /api/metrics/benchmark-scores - Redirects to benchmark-task for compatibility
router.get('/benchmark-scores', async (req, res, next) => {
  const { timeRange, task, global_step, window, block } = req.query;
  const actualTimeRange = timeRange || '-30d';
  
  try {
    // Using the same implementation as benchmark-task
    let fluxQuery = `
      from(bucket: "${influxConfig.bucket}")
        |> range(start: ${actualTimeRange})
        |> filter(fn: (r) => r["_measurement"] == "benchmark_task")
        |> filter(fn: (r) => r["_field"] == "score")
    `;
    
    if (task) {
      fluxQuery += `|> filter(fn: (r) => r["task"] == "${task}")`;
    }
    
    if (global_step) {
      fluxQuery += `|> filter(fn: (r) => r["global_step"] == "${global_step}")`;
    }
    
    if (window) {
      fluxQuery += `|> filter(fn: (r) => r["window"] == "${window}")`;
    }
    
    if (block) {
      fluxQuery += `|> filter(fn: (r) => r["block"] == "${block}")`;
    }
    
    fluxQuery += `
      |> group(columns: ["task"])
      |> last()
    `;
    
    const results = await influxService.queryWithTimeout(fluxQuery);
    
    const formattedResults = results.map(row => ({
      task: row.task,
      score: row._value,
      global_step: row.global_step,
      window: row.window,
      timestamp: row._time
    }));
    
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

// GET /api/metrics/timing
router.get('/timing', async (req, res, next) => {
  const { timeRange, window, uid } = req.query;
  const actualTimeRange = timeRange || '-6h';
  
  try {
    const response = await influxService.queryMetrics('timing', {
      window,
      uid
    }, actualTimeRange);
    
    res.json(response);
  } catch (err) {
    console.error("Error in /timing route:", err);
    next(err);
  }
});

export default router;