const express = require('express');
const router = express.Router();
const influxService = require('../services/influxService');
const config = require('../config/influxdb');

router.get('/losses', async (req, res) => {
  try {
    const { window, version, timeRange } = req.query;
    
    const losses = await influxService.getLosses({
      window,
      version: version || config.version,
      timeRange: timeRange || '24h'
    });
    
    // If no data found, check available versions
    if (losses.length === 0) {
      const availableVersions = await influxService.getAvailableVersions();
      return res.json({
        losses: [],
        message: `No data found for version ${version || config.version}`,
        availableVersions
      });
    }
    
    return res.json({
      losses,
      version: version || config.version,
      params: {
        window,
        timeRange: timeRange || '24h'
      }
    });

  } catch (error) {
    console.error('API Error:', error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
});

router.get('/tokens-per-sec', async (req, res) => {
  try {
    const { uid, version, timeRange, window, aggregate } = req.query;
    const result = await influxService.getTokensPerSec({
      uid,
      version: version || config.version,
      timeRange: timeRange || '24h',
      window,
      aggregate: aggregate === 'true'
    });
    res.json(result);
  } catch (error) {
    res.status(500).json({ 
      success: false,
      error: error.message
    });
  }
});

module.exports = router;